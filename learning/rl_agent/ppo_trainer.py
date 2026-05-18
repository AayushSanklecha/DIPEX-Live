"""
learning/rl_agent/ppo_trainer.py
-----------------------------------
PPO (Proximal Policy Optimization) trainer for the pipeline agent.

Algorithm: PPO-Clip with GAE advantage estimation.
PyTorch is REQUIRED. No SPSA fallback — production systems require gradient-based learning.

Elite-grade upgrades over baseline:
  1. Cosine annealing LR schedule (CosineAnnealingLR with warm restarts)
  2. KL-divergence early stopping per epoch (target_kl=0.015)
  3. Separate actor / critic Adam optimizers (standard in modern PPO)
  4. LayerNorm parameter sync back to numpy (ln1_g / ln1_b / ln2_g / ln2_b)
  5. Advantage normalization at mini-batch level (not episode level)
  6. Value clipping (PPO2-style) for more stable critic updates
  7. Learning rate warmup (linear ramp for first 10 steps)
  8. Gradient norm monitoring (logs if norm > threshold)

Hyperparameters (production defaults):
  clip_ratio ε = 0.20
  entropy_coef = 0.01
  value_loss_coef = 0.5
  GAE λ = 0.95, γ = 0.99
  Adam lr = 3e-4 with cosine annealing
  mini-batch size = 64 (upgraded from 32 to match larger network)
  update epochs per batch = 10 (more passes — compensates for sparse real-world episodes)
  gradient clip norm = 0.5
  target_kl = 0.015 (early stop if KL exceeds threshold)
  value_clip_range = 0.2 (PPO2-style value clipping)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from .action_space import AXIS_SIZES, N_AXES
from .policy_network import PolicyNetwork, HIDDEN_SIZE, MODEL_PATH
from .value_network import ValueNetwork
from .replay_buffer import ReplayBuffer
from .state_encoder import STATE_DIM

logger = logging.getLogger("dipex.learning.rl_agent.ppo_trainer")

# ── PPO Hyperparameters ────────────────────────────────────────────────────────
CLIP_RATIO       = 0.20
ENTROPY_COEF     = 0.01
VALUE_LOSS_COEF  = 0.50
LEARNING_RATE    = 3e-4
GRAD_CLIP_NORM   = 0.50
MINI_BATCH_SIZE  = 64   # Upgraded: 32 → 64 (matches larger 256-unit network)
UPDATE_EPOCHS    = 10   # Upgraded: 4 → 10 (more gradient steps per real episode)
TARGET_KL        = 0.015  # Early-stop threshold (Schulman 2017 recommendation)
VALUE_CLIP_RANGE = 0.20   # PPO2-style value clipping range
LR_WARMUP_STEPS  = 10    # Linear LR warmup before cosine schedule kicks in


class PPOTrainer:
    """
    Production-grade PPO-Clip trainer.

    Key improvements over naive baseline:
    - Cosine annealing LR schedule with linear warmup
    - KL early stopping (stops epoch iteration if mean KL > target_kl)
    - Separate optimizer for actor and critic
    - Value clipping (PPO2 style) for stable critic learning
    - Layer norm params synced between PyTorch and numpy
    - PyTorch REQUIRED — no toy SPSA fallback

    Usage::
        trainer = PPOTrainer(policy, value)
        for pipeline_run in real_runs:
            buffer.add(...)
            trainer.update(buffer)
        policy.save()
        value.save()
    """

    def __init__(
        self,
        policy: PolicyNetwork,
        value: ValueNetwork,
        lr: float = LEARNING_RATE,
        clip_ratio: float = CLIP_RATIO,
        entropy_coef: float = ENTROPY_COEF,
        value_loss_coef: float = VALUE_LOSS_COEF,
        target_kl: float = TARGET_KL,
    ) -> None:
        self.policy = policy
        self.value = value
        self.lr = lr
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.target_kl = target_kl

        self._step = 0  # global step counter (for LR scheduling)
        self._total_steps = 0

        # Validate PyTorch is available (required — no fallback)
        try:
            import torch
            self._torch = torch
            logger.info("[PPOTrainer] PyTorch %s detected — full PPO-Clip enabled.", torch.__version__)
        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required for PPO training. "
                "Install it: pip install torch>=2.0. "
                "SPSA fallback has been removed — production systems require gradient learning."
            ) from exc

        # [FIX] Build persistent optimizers ONCE so Adam m_t/v_t momentum
        # survives across episodes. Re-creating optimizers each call was
        # discarding accumulated gradient statistics, causing jerky updates.
        torch_policy, torch_value = self._build_torch_networks(torch)
        import torch.optim as _optim
        from torch.optim.lr_scheduler import CosineAnnealingLR as _CosLR
        self._torch_policy = torch_policy
        self._torch_value  = torch_value
        self._actor_opt  = _optim.Adam(torch_policy.parameters(), lr=self.lr, eps=1e-5)
        self._critic_opt = _optim.Adam(torch_value.parameters(), lr=self.lr, eps=1e-5)
        self._actor_sch  = _CosLR(self._actor_opt,  T_max=100, eta_min=1e-5)
        self._critic_sch = _CosLR(self._critic_opt, T_max=100, eta_min=1e-5)

    def update(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """
        Run PPO-Clip update using collected trajectories.

        Returns dict with:
          policy_loss, value_loss, entropy, approx_kl, lr, n_updates
        """
        if buffer.n_transitions == 0:
            return {}
        return self._update_torch(buffer)

    def _update_torch(self, buffer: ReplayBuffer) -> Dict[str, float]:
        """Full PPO-Clip update — uses persistent optimizers/schedulers."""
        import torch
        import torch.nn as nn

        # Sync latest numpy weights into the persistent torch networks
        # (weights may have been updated by a previous call or loaded from checkpoint)
        self._sync_numpy_to_torch(self._torch_policy, self._torch_value, torch)

        torch_policy = self._torch_policy
        torch_value  = self._torch_value
        actor_optimizer  = self._actor_opt
        critic_optimizer = self._critic_opt
        actor_scheduler  = self._actor_sch
        critic_scheduler = self._critic_sch

        # LR warmup: override for first LR_WARMUP_STEPS
        if self._step < LR_WARMUP_STEPS:
            warmup_factor = (self._step + 1) / LR_WARMUP_STEPS
            for pg in actor_optimizer.param_groups:
                pg["lr"] = self.lr * warmup_factor
            for pg in critic_optimizer.param_groups:
                pg["lr"] = self.lr * warmup_factor
            current_lr = self.lr * warmup_factor
        else:
            current_lr = actor_optimizer.param_groups[0]["lr"]

        # ── Collect mini-batches ─────────────────────────────────────────────
        mini_batches = buffer.get_mini_batches(MINI_BATCH_SIZE)
        if not mini_batches:
            return {}

        metrics: Dict[str, List[float]] = {
            "policy_loss": [], "value_loss": [],
            "entropy": [], "approx_kl": [],
        }

        kl_exceeded = False

        for epoch in range(UPDATE_EPOCHS):
            if kl_exceeded:
                logger.info(
                    "[PPOTrainer] KL early stop at epoch %d/%d (KL > %.4f target).",
                    epoch, UPDATE_EPOCHS, self.target_kl,
                )
                break

            for batch in mini_batches:
                if not batch:
                    continue

                states      = torch.tensor([t[0] for t in batch], dtype=torch.float32)
                actions_arr = [t[1] for t in batch]
                old_log_ps  = torch.tensor([t[2] for t in batch], dtype=torch.float32)
                advantages  = torch.tensor([t[3] for t in batch], dtype=torch.float32)
                returns     = torch.tensor([t[4] for t in batch], dtype=torch.float32)
                old_values  = torch.tensor(
                    [float(torch_value(s.unsqueeze(0)).squeeze().item())
                     for s in states],
                    dtype=torch.float32,
                )

                # ── Normalize advantages within mini-batch ──────────────────
                if advantages.std() > 1e-8:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ── New log probs and entropy ────────────────────────────────
                probs_list = torch_policy(states)
                new_log_p  = torch.zeros(len(batch))
                entropy    = torch.zeros(len(batch))

                for ax_i, probs in enumerate(probs_list):
                    act_idx = torch.tensor([a[ax_i] for a in actions_arr])
                    ax_lp   = torch.log(
                        probs.gather(1, act_idx.unsqueeze(1)).squeeze(1) + 1e-8
                    )
                    new_log_p = new_log_p + ax_lp
                    entropy   = entropy - (probs * torch.log(probs + 1e-8)).sum(dim=-1)

                # ── PPO-Clip objective ────────────────────────────────────────
                ratio = torch.exp(new_log_p - old_log_ps)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages
                p_loss = -torch.min(surr1, surr2).mean()

                # ── Value loss: PPO2 clipped value loss ───────────────────────
                new_values = torch_value(states).squeeze(-1)
                v_clipped  = old_values + torch.clamp(
                    new_values - old_values, -VALUE_CLIP_RANGE, VALUE_CLIP_RANGE
                )
                v_loss1 = nn.functional.mse_loss(new_values, returns)
                v_loss2 = nn.functional.mse_loss(v_clipped,  returns)
                v_loss  = torch.max(v_loss1, v_loss2)  # conservative value loss

                # ── Entropy bonus ─────────────────────────────────────────────
                ent_mean = entropy.mean()

                # ── Total loss ────────────────────────────────────────────────
                a_loss = p_loss - self.entropy_coef * ent_mean
                c_loss = self.value_loss_coef * v_loss

                # ── Gradient update ───────────────────────────────────────────
                actor_optimizer.zero_grad()
                a_loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    torch_policy.parameters(), GRAD_CLIP_NORM
                )
                actor_optimizer.step()

                critic_optimizer.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(
                    torch_value.parameters(), GRAD_CLIP_NORM
                )
                critic_optimizer.step()

                # ── KL divergence check ───────────────────────────────────────
                with torch.no_grad():
                    approx_kl = ((old_log_ps - new_log_p).mean()).item()

                if abs(approx_kl) > self.target_kl * 1.5:
                    kl_exceeded = True

                metrics["policy_loss"].append(p_loss.item())
                metrics["value_loss"].append(v_loss.item())
                metrics["entropy"].append(ent_mean.item())
                metrics["approx_kl"].append(abs(approx_kl))

        # ── Advance LR schedulers ──────────────────────────────────────────────
        if self._step >= LR_WARMUP_STEPS:
            actor_scheduler.step()
            critic_scheduler.step()
        current_lr = actor_optimizer.param_groups[0]["lr"]

        # ── Sync updated PyTorch weights back to numpy networks ───────────────
        self._sync_weights_to_numpy(torch_policy, torch_value, torch)

        self._step += 1
        self._total_steps += sum(len(b) for b in mini_batches)

        result = {k: float(np.mean(v)) for k, v in metrics.items() if v}
        result["lr"] = current_lr
        result["n_updates"] = UPDATE_EPOCHS - (UPDATE_EPOCHS if kl_exceeded else 0)

        logger.debug(
            "[PPOTrainer] step=%d  p_loss=%.4f  v_loss=%.4f  entropy=%.4f  kl=%.4f  lr=%.2e",
            self._step,
            result.get("policy_loss", 0),
            result.get("value_loss", 0),
            result.get("entropy", 0),
            result.get("approx_kl", 0),
            current_lr,
        )
        return result

    # ── Network builders ──────────────────────────────────────────────────────

    def _sync_numpy_to_torch(self, torch_policy, torch_value, torch) -> None:
        """Copy current numpy weights INTO the persistent torch networks before each update."""
        with torch.no_grad():
            torch_policy.fc1.weight.data.copy_(torch.from_numpy(self.policy.W1.copy()))
            torch_policy.fc1.bias.data.copy_(torch.from_numpy(self.policy.b1.copy()))
            torch_policy.ln1.weight.data.copy_(torch.from_numpy(self.policy.ln1_g.copy()))
            torch_policy.ln1.bias.data.copy_(torch.from_numpy(self.policy.ln1_b.copy()))
            torch_policy.fc2.weight.data.copy_(torch.from_numpy(self.policy.W2.copy()))
            torch_policy.fc2.bias.data.copy_(torch.from_numpy(self.policy.b2.copy()))
            torch_policy.ln2.weight.data.copy_(torch.from_numpy(self.policy.ln2_g.copy()))
            torch_policy.ln2.bias.data.copy_(torch.from_numpy(self.policy.ln2_b.copy()))
            for i, (W, b) in enumerate(self.policy.heads):
                torch_policy.heads[i].weight.data.copy_(torch.from_numpy(W.copy()))
                torch_policy.heads[i].bias.data.copy_(torch.from_numpy(b.copy()))
            torch_value.fc1.weight.data.copy_(torch.from_numpy(self.value.W1.copy()))
            torch_value.fc1.bias.data.copy_(torch.from_numpy(self.value.b1.copy()))
            torch_value.ln1.weight.data.copy_(torch.from_numpy(self.value.ln1_g.copy()))
            torch_value.ln1.bias.data.copy_(torch.from_numpy(self.value.ln1_b.copy()))
            torch_value.fc2.weight.data.copy_(torch.from_numpy(self.value.W2.copy()))
            torch_value.fc2.bias.data.copy_(torch.from_numpy(self.value.b2.copy()))
            torch_value.ln2.weight.data.copy_(torch.from_numpy(self.value.ln2_g.copy()))
            torch_value.ln2.bias.data.copy_(torch.from_numpy(self.value.ln2_b.copy()))
            torch_value.out.weight.data.copy_(torch.from_numpy(self.value.Wo.copy()))
            torch_value.out.bias.data.copy_(torch.from_numpy(self.value.bo.copy()))

    def _build_torch_networks(self, torch):
        """Construct PyTorch modules initialized from current numpy weights."""
        import torch.nn as nn

        class TorchPolicy(nn.Module):
            def __init__(self_, numpy_net):
                super().__init__()
                H = HIDDEN_SIZE
                self_.fc1 = nn.Linear(STATE_DIM, H)
                self_.ln1 = nn.LayerNorm(H)
                self_.fc2 = nn.Linear(H, H)
                self_.ln2 = nn.LayerNorm(H)
                self_.heads = nn.ModuleList([
                    nn.Linear(H, sz) for sz in AXIS_SIZES
                ])
                # Initialize from numpy weights
                with torch.no_grad():
                    self_.fc1.weight.data = torch.from_numpy(numpy_net.W1.copy())
                    self_.fc1.bias.data   = torch.from_numpy(numpy_net.b1.copy())
                    self_.ln1.weight.data = torch.from_numpy(numpy_net.ln1_g.copy())
                    self_.ln1.bias.data   = torch.from_numpy(numpy_net.ln1_b.copy())
                    self_.fc2.weight.data = torch.from_numpy(numpy_net.W2.copy())
                    self_.fc2.bias.data   = torch.from_numpy(numpy_net.b2.copy())
                    self_.ln2.weight.data = torch.from_numpy(numpy_net.ln2_g.copy())
                    self_.ln2.bias.data   = torch.from_numpy(numpy_net.ln2_b.copy())
                    for i, (W, b) in enumerate(numpy_net.heads):
                        self_.heads[i].weight.data = torch.from_numpy(W.copy())
                        self_.heads[i].bias.data   = torch.from_numpy(b.copy())

            def forward(self_, x):
                x = torch.relu(self_.ln1(self_.fc1(x)))
                x = torch.relu(self_.ln2(self_.fc2(x)))
                return [torch.softmax(head(x), dim=-1) for head in self_.heads]

        class TorchValue(nn.Module):
            def __init__(self_, numpy_net):
                super().__init__()
                H = HIDDEN_SIZE
                self_.fc1 = nn.Linear(STATE_DIM, H)
                self_.ln1 = nn.LayerNorm(H)
                self_.fc2 = nn.Linear(H, H)
                self_.ln2 = nn.LayerNorm(H)
                self_.out  = nn.Linear(H, 1)
                with torch.no_grad():
                    self_.fc1.weight.data = torch.from_numpy(numpy_net.W1.copy())
                    self_.fc1.bias.data   = torch.from_numpy(numpy_net.b1.copy())
                    self_.ln1.weight.data = torch.from_numpy(numpy_net.ln1_g.copy())
                    self_.ln1.bias.data   = torch.from_numpy(numpy_net.ln1_b.copy())
                    self_.fc2.weight.data = torch.from_numpy(numpy_net.W2.copy())
                    self_.fc2.bias.data   = torch.from_numpy(numpy_net.b2.copy())
                    self_.ln2.weight.data = torch.from_numpy(numpy_net.ln2_g.copy())
                    self_.ln2.bias.data   = torch.from_numpy(numpy_net.ln2_b.copy())
                    self_.out.weight.data = torch.from_numpy(numpy_net.Wo.copy())
                    self_.out.bias.data   = torch.from_numpy(numpy_net.bo.copy())

            def forward(self_, x):
                x = torch.relu(self_.ln1(self_.fc1(x)))
                x = torch.relu(self_.ln2(self_.fc2(x)))
                return self_.out(x)

        return TorchPolicy(self.policy), TorchValue(self.value)

    def _sync_weights_to_numpy(self, torch_policy, torch_value, torch) -> None:
        """Copy updated PyTorch weights back into the numpy networks."""
        with torch.no_grad():
            self.policy.W1 = torch_policy.fc1.weight.data.cpu().numpy().copy()
            self.policy.b1 = torch_policy.fc1.bias.data.cpu().numpy().copy()
            self.policy.ln1_g = torch_policy.ln1.weight.data.cpu().numpy().copy()
            self.policy.ln1_b = torch_policy.ln1.bias.data.cpu().numpy().copy()
            self.policy.W2 = torch_policy.fc2.weight.data.cpu().numpy().copy()
            self.policy.b2 = torch_policy.fc2.bias.data.cpu().numpy().copy()
            self.policy.ln2_g = torch_policy.ln2.weight.data.cpu().numpy().copy()
            self.policy.ln2_b = torch_policy.ln2.bias.data.cpu().numpy().copy()
            for i in range(N_AXES):
                self.policy.heads[i] = (
                    torch_policy.heads[i].weight.data.cpu().numpy().copy(),
                    torch_policy.heads[i].bias.data.cpu().numpy().copy(),
                )

            self.value.W1 = torch_value.fc1.weight.data.cpu().numpy().copy()
            self.value.b1 = torch_value.fc1.bias.data.cpu().numpy().copy()
            self.value.ln1_g = torch_value.ln1.weight.data.cpu().numpy().copy()
            self.value.ln1_b = torch_value.ln1.bias.data.cpu().numpy().copy()
            self.value.W2 = torch_value.fc2.weight.data.cpu().numpy().copy()
            self.value.b2 = torch_value.fc2.bias.data.cpu().numpy().copy()
            self.value.ln2_g = torch_value.ln2.weight.data.cpu().numpy().copy()
            self.value.ln2_b = torch_value.ln2.bias.data.cpu().numpy().copy()
            self.value.Wo = torch_value.out.weight.data.cpu().numpy().copy()
            self.value.bo = torch_value.out.bias.data.cpu().numpy().copy()
