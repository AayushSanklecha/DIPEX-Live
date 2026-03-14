# ─────────────────────────────────────────────────────────────────
# DIPEX Frontend — Dockerfile
# Multi-stage build for React/Vite SPA
# ─────────────────────────────────────────────────────────────────

# Base stage for dependencies
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci

# Stage: Development Server (Vite plugin hot-reloading)
FROM deps AS dev
WORKDIR /app
# We expose port 5173 for Vite
EXPOSE 5173
# --host 0.0.0.0 is strictly required so Docker maps it externally
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# Stage: Builder (for production)
FROM deps AS builder
WORKDIR /app
COPY . .
RUN npm run build

# Stage: Serve with Nginx (Production)
FROM nginx:alpine AS production
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 80
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1
CMD ["nginx", "-g", "daemon off;"]
