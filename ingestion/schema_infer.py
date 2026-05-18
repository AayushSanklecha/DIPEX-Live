"""
ingestion/schema_infer.py
--------------------------
Production-grade ML Semantic Type Classifier — v7 aware.

Architecture
------------
3-layer cascade per column:
  Layer 1 — NLP column-name regex (21 labels, 2 tiers)
  Layer 2 — ML model with confidence gate (0.45)
             Auto-detects v7 (58-feat: 30 stat + 28 NLP embedding)
             vs old (25-feat: stat-only) via schema_feature_registry.pkl
  Layer 3 — Statistical heuristic engine (always available)

NLP Embedding at Inference (v7 only)
--------------------------------------
When the loaded model's registry declares nlp_method=sentence_transformers,
SmartSchemaInferer lazy-loads sentence-transformers and appends the 28-dim
NLP embedding to the stat feature vector before calling the model.
Falls back to keyword embedding if sentence-transformers unavailable.

Row names (DataFrame index)
----------------------------
Row index values are NOT semantically typed — they are structural, not
content columns. All columns (including integer-indexed ones like 0, 1, 2)
ARE processed: col_name is coerced to str() before any NLP/regex work.

Semantic Labels (21)
--------------------
  id, age, amount, date, category, text, phone, email, boolean, zipcode,
  percentage, score, count, name, unknown,
  url, ip_address, coordinates, duration, address, currency_code
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("dipex.ingestion.schema_infer")

# ── Artifact paths ─────────────────────────────────────────────────────────────
_MODEL_DIR        = os.path.join(os.path.dirname(__file__), "..", "models")
_CLASSIFIER_PATH  = os.path.join(_MODEL_DIR, "schema_classifier.pkl")
_ENCODER_PATH     = os.path.join(_MODEL_DIR, "schema_label_encoder.pkl")
_REGISTRY_PATH    = os.path.join(_MODEL_DIR, "schema_feature_registry.pkl")

# ── Semantic label set ─────────────────────────────────────────────────────────
SEMANTIC_LABELS = [
    "id", "age", "amount", "date", "category", "text",
    "phone", "email", "boolean", "zipcode", "percentage",
    "score", "count", "name", "unknown",
    "url", "ip_address", "coordinates", "duration", "address", "currency_code",
]

# NLP anchor phrases for keyword embedding (21 types + 7 domains = 28 dims)
_DOMAIN_LABELS = ["banking", "healthcare", "finance", "ecommerce",
                   "government", "insurance", "generic"]

_SEMANTIC_KW: Dict[str, List[str]] = {
    # Full acronym + short-form coverage for all 21 semantic types.
    # Covers: standard English, enterprise abbreviations (SAP/Oracle),
    # banking acronyms, healthcare codes, analytics shorthands.
    "id":            ["id","uuid","key","pk","identifier","ref","serial",
                      "cd","code","seq","acct","ref_no","nbr","nr","no"],
    "age":           ["age","dob","birth","yr","years","old","tenure","yrs",
                      "seniority","age_at","current_age"],
    "amount":        ["amount","amt","price","prc","cost","rev","revenue",
                      "salary","sal","income","fee","balance","bal","value","val",
                      "total","tot","payment","pmt","pymt","spend","budget",
                      "expense","exp","sales","profit","loss","margin","tax",
                      "wage","credit","debit","charge","invoice","inv","fare",
                      "premium","prm","subsidy","grant","bonus","dividend",
                      "txn","tran","trn","transaction","mkt_val","mktval",
                      "cogs","cgs","asst","liab","voltage","power","energy",
                      "temperature","pressure","weight","mass","distance",
                      "speed","dosage","concentration","sum"],
    "date":          ["date","dt","time","ts","tms","dtm","dttm","timestamp",
                      "created","updated","period","start","end","yr","mo",
                      "wk","qtr","year","month","day","born","expires",
                      "eff_dt","eff_date","exp_dt","txn_dt","tran_dt",
                      "val_dt","vldt","mat_dt","matdt","post_dt","pst_dt",
                      "recorded","reported","filed","posted","modified"],
    "category":      ["type","typ","tp","cat","ctg","category","class","cls",
                      "segment","seg","sgmt","group","grp","tier","status","sts",
                      "stat","gender","region","reg","country","cntry","city",
                      "state","department","dept","dpt","division","div",
                      "sector","industry","channel","platform","level","priority",
                      "mode","tag","genre","brand","model","product","prod","prd",
                      "variant","color","colour","material","method","outcome",
                      "result","diagnosis","treatment","condition","fault","src"],
    "text":          ["text","txt","note","comment","cmnt","cmt","description",
                      "desc","dscr","remark","rmk","narrative","narr","message",
                      "msg","review","feedback","summary","body","content",
                      "bio","info","reason","detail"],
    "phone":         ["phone","mobile","tel","cell","fax","whatsapp","mob",
                      "ph","phn","contact_no","ph_no","mob_no"],
    "email":         ["email","mail","inbox","e_mail","eml"],
    "boolean":       ["flag","flg","is_","has_","active","actv","enabled",
                      "verified","approved","valid","deleted","archived",
                      "visible","public","required","ind","yn","tf","sw",
                      "bit","bool","del_flg","del_ind"],
    "zipcode":       ["zip","zp","postal","pstl","pincode","pncd","postcode",
                      "plz","pin_cd"],
    "percentage":    ["pct","percent","ratio","rt","rate","proportion","fraction",
                      "share","utilization","util","efficiency","eff","occupancy",
                      "churn","conversion","conv","growth","inflation","yield",
                      "freq","loss_rt","dflt_rt","apr","apy","humidity"],
    "score":         ["score","scr","rating","rtg","grade","rank","gpa","idx",
                      "fico","cibil","nps","sentiment","quality","accuracy",
                      "confidence","conf","probability","prob","weight","wgt",
                      "importance","polarity"],
    "count":         ["count","cnt","qty","quantity","frequency","freq","n_",
                      "num","nbr","nr","nmbr","no_of","num_of","number","volume",
                      "tot","clicks","views","visits","sessions","orders",
                      "transactions","events","records","occurrences","occ"],
    "name":          ["name","nm","nme","fname","fn","fnm","lname","ln","lnm",
                      "fullname","srnm","surname","username","customer_name",
                      "cust_nm","cust_name","emp_nm","emp_name","org_nm",
                      "comp_nm","author","owner","employee","emp","cust",
                      "vendor","alias","display_name"],
    "url":           ["url","link","href","website","uri","endpoint","domain",
                      "homepage","permalink","source_url","image_url","avatar"],
    "ip_address":    ["ip","inet","ipv4","ipv6","addr","host","remote_addr",
                      "client_ip","server_ip"],
    "coordinates":   ["lat","lon","latitude","longitude","coord","gps",
                      "geo_x","geo_y","northing","easting"],
    "duration":      ["duration","dur","elapsed","seconds","mins","hours",
                      "ttl","tmo","latency","uptime","downtime","timeout",
                      "interval","session_length","call_duration","watch_time"],
    "address":       ["address","adr","addr","street","location","building",
                      "house_no","flat","suite","lane","road","avenue"],
    "currency_code": ["currency","ccy","curr","fx","forex","base_currency",
                      "quote_currency","iso_currency"],
    "unknown":       [],
}
_DOMAIN_KW: Dict[str, List[str]] = {
    "banking":    ["loan","account","aml","kyc","iban","repayment","ledger",
                   "collateral","transaction"],
    "healthcare": ["patient","diagnosis","drug","bmi","clinical","hospital",
                   "vital","icd","medication"],
    "finance":    ["stock","market_cap","ebitda","eps","portfolio","equity",
                   "bond","yield","nav"],
    "ecommerce":  ["sku","cart","checkout","product_id","basket","shipment",
                   "refund","order_item"],
    "government": ["census","voter","budget","taxpayer","regulation",
                   "municipality","policy_number"],
    "insurance":  ["policy_num","premium","claim","actuary","underwrite",
                   "beneficiary","coverage"],
    "generic":    [],
}

# NLP Tier A: column-name match is definitive → return immediately
_NLP_TIER_A = {
    "email", "phone", "date", "id", "boolean", "zipcode", "name",
    "url", "ip_address", "coordinates", "currency_code",
}
# NLP Tier B: column-name match is a strong hint used when ML is uncertain
_NLP_TIER_B = {
    "age", "amount", "score", "count", "percentage",
    "category", "text", "duration", "address",
}

# ── Regex tag map (column-name NLP, all 21 labels) ────────────────────────────
# ── Regex tag map (column-name NLP, all 21 labels) ────────────────────────────
# Uses word-boundary anchors to avoid false positives.
# Covers: standard English, short forms, SAP/Oracle enterprise naming,
# banking (txn, amt, bal, acct), analytics (scr, idx, rtg),
# healthcare (bmi, icd mapped via stats not name, diag_cd etc.)
_NLP_TAG_MAP: Dict[str, List[str]] = {
    "id": [
        r"\bid\b", r"_id$", r"^id_", r"uuid", r"guid",
        r"\bkey\b", r"identifier", r"ref$", r"^ref_",
        r"serial", r"code$", r"accession", r"barcode",
        r"sku", r"asin", r"ean", r"upc",
        r"\bcd$", r"_cd$", r"^cd_",        # cd = code (SAP/Oracle)
        r"\bseq\b", r"_seq$",              # seq = sequence
        r"\bacct\b", r"_acct$",            # acct = account id
        r"\bref_?no\b", r"_ref$",
        r"\bno\b", r"_no$", r"^no_",
        r"\bnbr\b", r"_nbr$",              # nbr = number (Oracle)
        r"\bnr\b", r"_nr$",                # nr = number (European DBs)
    ],
    "age": [
        r"\bage\b", r"years?_?old", r"dob", r"birth",
        r"age_", r"_age$", r"yrs", r"tenure", r"seniority",
        r"\byr\b", r"_yr$",                # yr = year-based age
        r"\bage_?at\b", r"\bcurrent_?age\b",
    ],
    "amount": [
        r"amount", r"price", r"revenue", r"cost", r"salary",
        r"income", r"fee", r"balance", r"value", r"total",
        r"payment", r"spend", r"budget", r"expense", r"sales",
        r"profit", r"loss", r"margin", r"tax", r"wage",
        r"credit", r"debit", r"charge", r"invoice", r"fare",
        r"premium", r"subsidy", r"grant", r"bonus", r"dividend",
        r"voltage", r"current", r"power", r"energy",
        r"temperature", r"pressure", r"flow", r"weight", r"mass",
        r"distance", r"length", r"height", r"width", r"depth",
        r"speed", r"velocity", r"concentration", r"dosage",
        r"\bamt\b", r"_amt$", r"^amt_",    # amt = amount
        r"\btxn\b", r"_txn$",              # txn = transaction
        r"\btrn\b", r"_trn$",
        r"\btran\b", r"_tran$",
        r"\bsal\b", r"_sal$",              # sal = salary
        r"\brev\b", r"_rev$",              # rev = revenue
        r"\bexp\b", r"_exp$",              # exp = expense
        r"\bpmt\b", r"_pmt$",              # pmt = payment
        r"\bpymt\b", r"_pymt$",
        r"\binv\b", r"_inv$",              # inv = invoice
        r"\bbal\b", r"_bal$",              # bal = balance
        r"\bval\b", r"_val$",              # val = value
        r"\btot\b", r"_tot$",              # tot = total
        r"\bprc\b", r"_prc$",              # prc = price
        r"\bmkt_?val\b", r"mktval",
        r"\bprm\b", r"_prm$",
        r"\bcgs\b", r"\bcogs\b",
        r"\basst\b", r"\bliab\b",
    ],
    "date": [
        r"date", r"time", r"_at$", r"_on$", r"timestamp",
        r"created", r"updated", r"modified", r"born", r"expires",
        r"start", r"end", r"period", r"year", r"month", r"day",
        r"datetime", r"recorded", r"reported", r"filed", r"posted",
        r"\bdt$", r"_dt$", r"^dt_",        # dt = date
        r"\bts$", r"_ts$",                 # ts = timestamp
        r"\btms$", r"_tms$",
        r"\bdtm$", r"_dtm$",
        r"\bdttm$", r"_dttm$",
        r"\byr$", r"_yr$", r"^yr_",        # yr = year
        r"\bmo$", r"_mo$",                 # mo = month
        r"\bwk$", r"_wk$",                 # wk = week
        r"\bqtr$", r"_qtr$",               # qtr = quarter
        r"\beff_?dt\b", r"\beff_?date\b",  # effective date
        r"\bexp_?dt\b", r"\bexpiry_?dt\b",
        r"\bpst_?dt\b", r"\bpost_?dt\b",
        r"\btxn_?dt\b", r"\btran_?dt\b",
        r"\bval_?dt\b", r"\bvldt\b",       # value date (banking)
        r"\bmat_?dt\b", r"\bmatdt\b",      # maturity date
    ],
    "phone": [
        r"phone", r"mobile", r"cell", r"tel", r"fax",
        r"contact_no", r"whatsapp",
        r"\bmob\b", r"_mob$",
        r"\bph\b", r"_ph$", r"^ph_",
        r"\bphn\b", r"_phn$",
        r"\bph_?no\b", r"\bmob_?no\b",
    ],
    "email": [r"email", r"e_?mail", r"mail", r"inbox", r"\beml\b"],
    "name": [
        r"\bname\b", r"first_?name", r"last_?name", r"full_?name",
        r"username", r"user_?name", r"customer_name", r"author",
        r"owner", r"employee", r"vendor", r"supplier", r"contact",
        r"alias", r"display_name", r"label_name",
        r"\bnm$", r"_nm$", r"^nm_",        # nm = name (enterprise standard)
        r"\bnme$", r"_nme$",
        r"\bfn$", r"_fn$",                 # fn = first name
        r"\bln$", r"_ln$",                 # ln = last name
        r"\bfnm$", r"_fnm$",
        r"\blnm$", r"_lnm$",
        r"\bsrnm\b", r"_srnm$",            # srnm = surname
        r"\bcust_?nm\b", r"\bcust_?name\b",
        r"\bemp_?nm\b", r"\bemp_?name\b",
        r"\borg_?nm\b", r"\bcomp_?nm\b",
        r"\bcust\b", r"_cust$",
        r"\bemp\b", r"_emp$",
    ],
    "zipcode": [
        r"zip", r"postal", r"pincode", r"postcode", r"plz",
        r"\bzp$", r"_zp$",
        r"\bpstl\b", r"_pstl$",
        r"\bpncd\b", r"pin_?cd",
    ],
    "category": [
        r"category", r"type", r"class", r"group", r"segment",
        r"label", r"status", r"gender", r"region", r"country",
        r"city", r"state", r"department", r"division", r"sector",
        r"industry", r"channel", r"platform", r"tier", r"level",
        r"priority", r"mode", r"flag", r"tag", r"genre",
        r"brand", r"model", r"product", r"variant", r"colour",
        r"color", r"material", r"method", r"outcome", r"result",
        r"diagnosis", r"treatment", r"condition", r"fault",
        r"\bcat\b", r"_cat$", r"^cat_",    # cat = category
        r"\bctg\b", r"_ctg$",
        r"\btyp\b", r"_typ$", r"^typ_",    # typ = type
        r"\btp\b", r"_tp$",
        r"\bgrp\b", r"_grp$", r"^grp_",    # grp = group
        r"\bseg\b", r"_seg$",              # seg = segment
        r"\bsgmt\b", r"_sgmt$",
        r"\bdept\b", r"_dept$",            # dept = department
        r"\bdpt\b", r"_dpt$",
        r"\bdiv\b", r"_div$",              # div = division
        r"\bcls\b", r"_cls$",              # cls = class
        r"\bsrc\b", r"_src$",              # src = source
        r"\bsts\b", r"_sts$",              # sts = status
        r"\bstat\b", r"_stat$",
        r"\breg\b", r"_reg$",              # reg = region
        r"\bcntry\b", r"_cntry$",
        r"\bprd\b", r"_prd$",              # prd = product/period
        r"\bprod\b", r"_prod$",
    ],
    "score": [
        r"score", r"rating", r"rank", r"grade", r"gpa",
        r"fico", r"cibil", r"nps", r"satisfaction", r"quality",
        r"accuracy", r"precision", r"recall", r"f1",
        r"confidence", r"probability", r"weight", r"importance",
        r"sentiment", r"polarity",
        r"\bscr\b", r"_scr$",              # scr = score
        r"\brtg\b", r"_rtg$",              # rtg = rating
        r"\bidx\b", r"_idx$",              # idx = index/score
        r"\bwgt\b", r"_wgt$",              # wgt = weight
        r"\bprob\b", r"_prob$",
        r"\bconf\b", r"_conf$",
    ],
    "percentage": [
        r"pct", r"percent", r"rate", r"ratio", r"proportion",
        r"fraction", r"share", r"utilization", r"utilisation",
        r"efficiency", r"occupancy", r"coverage", r"completion",
        r"error_rate", r"default_rate", r"churn", r"conversion",
        r"growth", r"inflation", r"yield", r"apr", r"apy",
        r"humidity", r"moisture",
        r"\brt\b", r"_rt$", r"^rt_",       # rt = rate
        r"\butil\b", r"_util$",
        r"\bconv\b", r"_conv$",
        r"\beff\b", r"_eff$",
        r"\bfreq\b", r"_freq$",
    ],
    "count": [
        r"count", r"num", r"number", r"qty", r"quantity",
        r"volume", r"n_", r"total_", r"_total$",
        r"clicks", r"views", r"visits", r"sessions", r"hits",
        r"orders", r"transactions", r"events", r"records",
        r"occurrences", r"frequency", r"attempts",
        r"\bcnt\b", r"_cnt$",
        r"\bnbr\b", r"_nbr$",              # nbr (Oracle/SAP)
        r"\bnr\b", r"_nr$",                # nr (European DBs)
        r"\bnmbr\b", r"_nmbr$",
        r"\bno_of\b", r"\bnum_of\b",
        r"\btot\b", r"_tot$",
        r"\bocc\b", r"_occ$",
    ],
    "boolean": [
        r"^is_", r"^has_", r"^flag", r"active", r"enabled",
        r"verified", r"approved", r"valid", r"deleted", r"archived",
        r"exists", r"available", r"visible", r"public", r"primary",
        r"default", r"required", r"mandatory", r"allowed",
        r"\bflg$", r"_flg$", r"^flg_",    # flg = flag (SAP/Oracle)
        r"\bind$", r"_ind$", r"^ind_",    # ind = indicator
        r"\byn$", r"_yn$",                 # yn = yes/no
        r"\btf$", r"_tf$",                 # tf = true/false
        r"\bsw$", r"_sw$",                 # sw = switch
        r"\bbit$", r"_bit$",
        r"\bbool$", r"_bool$",
        r"^actv\b", r"_actv$",
        r"^del_?flg\b", r"^del_?ind\b",
    ],
    "text": [
        r"description", r"notes?", r"comment", r"remarks", r"message",
        r"narrative", r"summary", r"body", r"content", r"text",
        r"review", r"feedback", r"reason", r"detail", r"bio",
        r"\bdesc\b", r"_desc$", r"^desc_",  # desc = description
        r"\btxt\b", r"_txt$",
        r"\bcmnt\b", r"_cmnt$",
        r"\bcmt\b", r"_cmt$",
        r"\brmk\b", r"_rmk$",
        r"\bnarr\b", r"_narr$",
        r"\bmsg\b", r"_msg$",
        r"\bdscr\b", r"_dscr$",
        r"\binfo\b", r"_info$",
    ],
    "url": [
        r"url", r"link", r"href", r"endpoint", r"uri", r"website",
        r"domain", r"homepage", r"permalink", r"source_url",
        r"image_url", r"avatar", r"thumbnail",
    ],
    "ip_address": [
        r"ip", r"ip_addr", r"ipv4", r"ipv6", r"host",
        r"remote_addr", r"client_ip", r"server_ip", r"origin_ip",
    ],
    "coordinates": [
        r"lat", r"latitude", r"lng", r"lon", r"longitude",
        r"geo_x", r"geo_y", r"coord", r"x_coord", r"y_coord",
        r"northing", r"easting",
    ],
    "duration": [
        r"duration", r"elapsed", r"latency", r"response_time",
        r"uptime", r"downtime", r"ttl", r"timeout", r"interval",
        r"session_length", r"call_duration", r"watch_time",
        r"\bdur\b", r"_dur$",
        r"\btmo\b", r"_tmo$",               # tmo = timeout
    ],
    "address": [
        r"address", r"street", r"addr", r"location",
        r"building", r"house_no", r"flat", r"suite",
        r"\badr\b", r"_adr$",               # adr = address
    ],
    "currency_code": [
        r"currency", r"ccy", r"iso_currency", r"currency_code",
        r"fx", r"forex", r"base_currency", r"quote_currency",
    ],
}



# =============================================================================
# Lightweight NLP embedder — mirrors ColabNLPAnalyzer from training scripts
# =============================================================================

class _InferenceNLPEmbedder:
    """
    28-dim NLP embedding for a column name at inference time.
    Matches exactly what the v7 training script produces via ColabNLPAnalyzer.
    Tries sentence-transformers first; keyword fallback if unavailable.

    Result: 21-dim semantic softmax + 7-dim domain scores = 28 dims total.
    These must match NLP_FEAT_NAMES from the training registry exactly.
    """

    def __init__(self, preferred_method: str = "auto") -> None:
        self._encoder    = None
        self._anchors: Dict[str, np.ndarray] = {}
        self._method     = "keyword"
        if preferred_method in ("sentence_transformers", "auto"):
            self._try_load_st()

    def _try_load_st(self) -> None:
        try:
            import os as _os
            _os.environ.setdefault("HF_TOKEN", "")
            _os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            self._precompute()
            self._method = "sentence_transformers"
            logger.info("[NLP-Infer] sentence-transformers loaded (384-dim anchors)")
        except Exception as exc:
            logger.debug("[NLP-Infer] sentence-transformers unavailable (%s) — keyword fallback", exc)
            self._method = "keyword"

    def _precompute(self) -> None:
        """Pre-compute mean anchor embeddings for all 21 semantic + 7 domain labels."""
        _SEMANTIC_ANCHORS = {
            "id":            ["unique identifier column","primary key field","record id","customer id"],
            "age":           ["age in years","person age","customer age","years old"],
            "amount":        ["monetary amount","transaction amount","payment amount",
                              "revenue figure","cost value","price column","balance amount"],
            "date":          ["date column timestamp","event date","transaction date","effective date"],
            "category":      ["categorical variable","class label","group type","product category"],
            "text":          ["free text description","notes field","comments column","narrative text"],
            "phone":         ["phone number","mobile number","telephone number","cell phone"],
            "email":         ["email address","email id column","user email","contact email"],
            "boolean":       ["binary flag indicator","yes no column","true false flag",
                              "boolean indicator","active inactive flag"],
            "zipcode":       ["zip code postal code","pin code","postal area code"],
            "percentage":    ["percentage value","ratio proportion","fractional rate",
                              "percent column","growth rate percentage"],
            "score":         ["credit score","risk score","model score prediction",
                              "rating value","performance score","grade point average"],
            "count":         ["count of occurrences","number of items","frequency count",
                              "quantity column","total number","visit count"],
            "name":          ["person name","customer name","full name",
                              "company name","organization name","entity name"],
            "url":           ["url web address","hyperlink column","website url"],
            "ip_address":    ["ip address inet","network address","ipv4 address","server ip"],
            "coordinates":   ["latitude longitude coordinate","gps coordinate","geographic point"],
            "duration":      ["duration in seconds","time elapsed","session duration",
                              "call duration","response time","processing time"],
            "address":       ["street address","mailing address","residential address"],
            "currency_code": ["currency code iso","payment currency","transaction currency"],
            "unknown":       ["unknown column type","unclassified column","miscellaneous field"],
        }
        _DOMAIN_ANCHORS = {
            "banking":    ["bank account transaction","loan repayment","aml kyc compliance",
                           "iban swift code","debit credit ledger"],
            "healthcare": ["patient diagnosis record","icd code clinical","drug dosage",
                           "bmi vital signs","hospital admission"],
            "finance":    ["stock price trading volume","eps earnings per share",
                           "market capitalization","ebitda profit loss"],
            "ecommerce":  ["product sku inventory","shopping cart order",
                           "customer checkout return","product review rating"],
            "government": ["census population data","government policy regulation",
                           "public expenditure budget","taxpayer national id"],
            "insurance":  ["insurance policy premium","claim settlement actuarial",
                           "underwriting risk assessment","beneficiary coverage"],
            "generic":    ["general purpose data","research dataset",
                           "scientific measurement tabular","generic numeric data"],
        }
        st = self._encoder
        for label, phrases in _SEMANTIC_ANCHORS.items():
            vecs = st.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
            self._anchors[f"type_{label}"] = vecs.mean(axis=0)
        for label, phrases in _DOMAIN_ANCHORS.items():
            vecs = st.encode(phrases, normalize_embeddings=True, show_progress_bar=False)
            self._anchors[f"domain_{label}"] = vecs.mean(axis=0)

    def _normalize_name(self, name: str) -> str:
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(name))
        s = re.sub(r"[_\-/.]", " ", s)
        return re.sub(r"\s+", " ", s).strip().lower()

    def embed(self, col_name: str) -> np.ndarray:
        """Return 28-dim NLP vector matching training script output."""
        col_name = str(col_name)   # handle integer-indexed columns
        if self._method == "sentence_transformers":
            return self._embed_st(col_name)
        return self._embed_keyword(col_name)

    def _embed_st(self, col_name: str) -> np.ndarray:
        readable = self._normalize_name(col_name)
        vec = self._encoder.encode([readable], normalize_embeddings=True,
                                   show_progress_bar=False)[0]
        sims = []
        for label in SEMANTIC_LABELS:
            anchor = self._anchors.get(f"type_{label}", np.zeros(384))
            sims.append(float(np.dot(vec, anchor) / (np.linalg.norm(anchor) + 1e-9)))
        for label in _DOMAIN_LABELS:
            anchor = self._anchors.get(f"domain_{label}", np.zeros(384))
            sims.append(float(np.dot(vec, anchor) / (np.linalg.norm(anchor) + 1e-9)))
        arr = np.array(sims, dtype=np.float32)
        t = arr[:len(SEMANTIC_LABELS)]
        t = np.exp(t * 5); t /= (t.sum() + 1e-9)
        arr[:len(SEMANTIC_LABELS)] = t
        return arr

    def _embed_keyword(self, col_name: str) -> np.ndarray:
        col_l = self._normalize_name(col_name)
        scores = {l: 0.0 for l in SEMANTIC_LABELS}
        for l, kws in _SEMANTIC_KW.items():
            for kw in kws:
                if kw in col_l:
                    scores[l] += 1.0
        total = max(sum(scores.values()), 1.0)
        type_arr = np.array([scores[l] / total for l in SEMANTIC_LABELS], dtype=np.float32)
        # Domain dims: check domain keywords
        dom_scores = []
        for dom, kws in _DOMAIN_KW.items():
            dom_scores.append(1.0 if any(k in col_l for k in kws) else 0.0)
        return np.concatenate([type_arr, np.array(dom_scores, dtype=np.float32)])


# =============================================================================
# Statistical feature extractor (v7-aligned: 30 features)
# =============================================================================

_STAT_FEAT_NAMES_V7 = [
    "null_rate", "unique_rate", "is_numeric", "is_string", "is_datetime",
    "mean_val", "std_val", "skew_val", "kurt_val", "iqr_val",
    "min_val", "max_val", "q25_val", "median_val", "q75_val",
    "all_integer", "max_lt_200", "max_lt_1", "all_positive", "log_n_distinct",
    "email_pattern", "phone_pattern", "mean_str_len", "high_cardinality",
    "low_cardinality", "url_pattern", "ip_pattern", "coord_range",
    "coord_precision", "cv_coeff",
]

# Legacy 25-feature order (pre-v7 models)
_STAT_FEAT_NAMES_LEGACY = [
    "null_rate", "unique_rate", "is_numeric", "is_string", "is_datetime",
    "mean_val", "std_val", "min_val", "max_val", "skew_val",
    "all_integer", "max_lt_200", "max_lt_1", "all_positive", "n_distinct",
    "email_pattern", "phone_pattern", "mean_str_len",
    "high_cardinality", "low_cardinality",
    "url_pattern", "ip_pattern", "coord_range", "coord_precision",
    "currency_pattern",
]


def _extract_stat_features_v7(series: pd.Series) -> Dict[str, float]:
    """30-feature statistical extraction matching v7 training script exactly."""
    s = series.dropna()
    is_num = pd.api.types.is_numeric_dtype(series)
    is_str = pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    is_dt  = pd.api.types.is_datetime64_any_dtype(series)
    nv = pd.to_numeric(s, errors="coerce").dropna() if not is_num else s.dropna()
    sv = s.astype(str) if is_str else pd.Series([], dtype=str)

    n_total     = max(len(series), 1)
    n_dist      = float(series.nunique(dropna=True))
    null_rate   = float(series.isnull().mean())
    unique_rate = n_dist / n_total
    mean_v = float(nv.mean())   if len(nv) else 0.0
    std_v  = float(nv.std())    if len(nv) > 1 else 0.0
    skew_v = float(nv.skew())   if len(nv) > 3 else 0.0
    kurt_v = float(nv.kurt())   if len(nv) > 3 else 0.0
    min_v  = float(nv.min())    if len(nv) else 0.0
    max_v  = float(nv.max())    if len(nv) else 0.0
    q25_v  = float(nv.quantile(0.25)) if len(nv) > 3 else 0.0
    med_v  = float(nv.median())       if len(nv) else 0.0
    q75_v  = float(nv.quantile(0.75)) if len(nv) > 3 else 0.0
    iqr_v  = q75_v - q25_v
    try:    all_int = float((nv == nv.astype(int)).all()) if len(nv) else 0.0
    except: all_int = 0.0

    ep = float(sv.str.contains(r"@.*\.", na=False).mean())                                 if is_str and len(sv) else 0.0
    pp = float(sv.str.contains(r"^\+?\d[\d\s\-()\\+]{7,}$", na=False, regex=True).mean()) if is_str and len(sv) else 0.0
    sl = float(sv.str.len().mean())                                                         if is_str and len(sv) else 0.0
    up = float(sv.str.contains(r"https?://|www\.", na=False).mean())                        if is_str and len(sv) else 0.0
    ip = float(sv.str.match(r"^(\d{1,3}\.){3}\d{1,3}$", na=False).mean())                 if is_str and len(sv) else 0.0
    cr = float(((nv >= -180) & (nv <= 180)).all()) if len(nv) else 0.0
    cp = float((nv % 1 != 0).mean() > 0.8)         if len(nv) else 0.0
    cv = min(float(std_v / (abs(mean_v) + 1e-9)), 100.0)

    return {
        "null_rate": null_rate, "unique_rate": unique_rate,
        "is_numeric": float(is_num), "is_string": float(is_str), "is_datetime": float(is_dt),
        "mean_val": min(max(mean_v, -1e6), 1e6), "std_val": min(std_v, 1e6),
        "skew_val": min(max(skew_v, -10), 10), "kurt_val": min(max(kurt_v, -10), 100),
        "iqr_val": min(iqr_v, 1e6), "min_val": min(max(min_v, -1e6), 1e6),
        "max_val": min(max(max_v, -1e6), 1e6), "q25_val": min(max(q25_v, -1e6), 1e6),
        "median_val": min(max(med_v, -1e6), 1e6), "q75_val": min(max(q75_v, -1e6), 1e6),
        "all_integer": all_int,
        "max_lt_200": float(max_v < 200) if len(nv) else 0.0,
        "max_lt_1": float(max_v <= 1.0) if len(nv) else 0.0,
        "all_positive": float((nv >= 0).all()) if len(nv) else 0.0,
        "log_n_distinct": float(np.log1p(n_dist)),
        "email_pattern": ep, "phone_pattern": pp, "mean_str_len": min(sl, 1000),
        "high_cardinality": float(unique_rate > 0.9),
        "low_cardinality": float(unique_rate < 0.05),
        "url_pattern": up, "ip_pattern": ip, "coord_range": cr,
        "coord_precision": cp, "cv_coeff": cv,
    }


def _extract_stat_features_legacy(series: pd.Series, col_name: str) -> Dict[str, float]:
    """Legacy 25-feature extraction for pre-v7 models."""
    f = _extract_stat_features_v7(series)
    sv = series.dropna().astype(str) if (
        pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)
    ) else pd.Series([], dtype=str)
    currency_p = (
        float(sv.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7)
        if len(sv) else 0.0
    )
    return {
        "null_rate": f["null_rate"], "unique_rate": f["unique_rate"],
        "is_numeric": f["is_numeric"], "is_string": f["is_string"],
        "is_datetime": f["is_datetime"],
        "mean_val": f["mean_val"], "std_val": f["std_val"],
        "min_val": f["min_val"], "max_val": f["max_val"],
        "skew_val": f["skew_val"], "all_integer": f["all_integer"],
        "max_lt_200": f["max_lt_200"], "max_lt_1": f["max_lt_1"],
        "all_positive": f["all_positive"],
        "n_distinct": float(series.nunique(dropna=True)),
        "email_pattern": f["email_pattern"], "phone_pattern": f["phone_pattern"],
        "mean_str_len": f["mean_str_len"],
        "high_cardinality": f["high_cardinality"], "low_cardinality": f["low_cardinality"],
        "url_pattern": f["url_pattern"], "ip_pattern": f["ip_pattern"],
        "coord_range": f["coord_range"], "coord_precision": f["coord_precision"],
        "currency_pattern": currency_p,
    }


# ── NLP tag helpers ────────────────────────────────────────────────────────────

def _nlp_tags_from_name(col_name: str) -> List[str]:
    """Return matching semantic tag hints from column name regex patterns."""
    # Integer-indexed columns (0, 1, 2...) → coerce to string first
    lower = str(col_name).lower().replace(" ", "_")
    tags: List[str] = []
    for tag, patterns in _NLP_TAG_MAP.items():
        for pat in patterns:
            if re.search(pat, lower):
                tags.append(tag)
                break
    return tags


# ── Heuristic fallback ─────────────────────────────────────────────────────────

def _heuristic_infer(series: pd.Series, col_name: str) -> str:
    """Rule-based semantic type inference (21-label). Always available."""
    col_name = str(col_name)
    tags = _nlp_tags_from_name(col_name)
    if tags:
        return tags[0]

    f = _extract_stat_features_v7(series)
    sv = series.dropna().astype(str) if f["is_string"] else pd.Series([], dtype=str)

    # Pattern-first checks
    if f["url_pattern"] > 0.7: return "url"
    if f["ip_pattern"] > 0.7:  return "ip_address"
    if len(sv) > 0 and sv.str.match(r"^[A-Z]{3}$", na=False).mean() > 0.7:
        return "currency_code"
    if f["is_datetime"] == 1.0: return "date"
    if f["email_pattern"] > 0.7: return "email"
    if f["phone_pattern"] > 0.7: return "phone"

    if (f["is_numeric"] == 1.0 and f["coord_range"] == 1.0
            and f["coord_precision"] == 1.0 and f["min_val"] < 0):
        return "coordinates"

    if f["is_numeric"] == 1.0:
        if f["max_lt_1"] and f["all_positive"]: return "percentage"
        if f["max_lt_200"] and f["all_integer"] and f["min_val"] >= 0: return "age"
        if f["high_cardinality"]: return "id"
        if f["all_integer"] and f["low_cardinality"]: return "count"
        return "amount"

    if f["is_string"] == 1.0:
        if f["low_cardinality"]: return "category"
        if len(sv) > 0:
            dur_pat = r"^\d+:\d{2}(:\d{2})?$|^\d+\s*(s|sec|min|h|hr|hours?)$"
            if sv.str.match(dur_pat, na=False).mean() > 0.5: return "duration"
            if f["mean_str_len"] > 20:
                addr_pat = r"\d+.*\b(st|street|rd|road|ave|avenue|blvd|ln|lane|dr|drive|nagar|colony)\b"
                if sv.str.contains(addr_pat, case=False, na=False, regex=True).mean() > 0.3:
                    return "address"
        if f["mean_str_len"] > 60: return "text"
        return "name"
    return "unknown"


# =============================================================================
# SmartSchemaInferer — main class
# =============================================================================

class SmartSchemaInferer:
    """
    Production-grade semantic type classifier for DataFrame columns.

    Automatically detects v7 models (58-feature: 30 stat + 28 NLP) vs
    legacy models (25-feature: stat-only) via schema_feature_registry.pkl.

    Column names: ALL names processed, including integer-indexed (0, 1, 2...)
    Row names: Not semantically typed (index = structural, not content)
    Extreme missingness: Handled — null_rate, is_numeric etc. work at 0-100% null

    3-layer cascade:
      Layer 1 — NLP regex on column name (Tier A: definitive, Tier B: hint)
      Layer 2 — ML model (v7 NLP-augmented or legacy stat-only)
      Layer 3 — Statistical heuristic engine (always available)
    """

    _ML_CONF_THRESHOLD = 0.45

    def __init__(self) -> None:
        self._model:    Any = None
        self._encoder:  Any = None
        self._registry: Dict[str, Any] = {}
        self._method:   str = "heuristic"
        self._is_v7:    bool = False
        self._nlp:      Optional[_InferenceNLPEmbedder] = None
        self._n_stat:   int = 30
        self._n_nlp:    int = 28
        self._stat_feat_names: List[str] = _STAT_FEAT_NAMES_V7
        self._load_model()

    def _load_model(self) -> None:
        """Load ML artifact + registry. Detect v7 vs legacy automatically."""
        try:
            import joblib  # type: ignore

            if not (os.path.exists(_CLASSIFIER_PATH) and os.path.exists(_ENCODER_PATH)):
                logger.info("SmartSchemaInferer: model artifacts not found — heuristic mode.")
                return

            self._model   = joblib.load(_CLASSIFIER_PATH)
            self._encoder = joblib.load(_ENCODER_PATH)
            self._method  = "ml"

            # ── Detect v7 via feature registry ───────────────────────────────
            if os.path.exists(_REGISTRY_PATH):
                self._registry = joblib.load(_REGISTRY_PATH)
                n_stat = self._registry.get("n_stat", 25)
                n_nlp  = self._registry.get("n_nlp", 0)
                nlp_m  = self._registry.get("nlp_method", "keyword")

                if n_stat == 30 and n_nlp == 28:
                    # v7 model: need to embed column name and append NLP dims
                    self._is_v7          = True
                    self._n_stat         = 30
                    self._n_nlp          = 28
                    self._stat_feat_names = _STAT_FEAT_NAMES_V7
                    self._nlp            = _InferenceNLPEmbedder(
                        preferred_method=nlp_m
                    )
                    logger.info(
                        "SmartSchemaInferer: v7 model loaded  "
                        "(30 stat + 28 NLP = 58 feat)  NLP=%s",
                        self._nlp._method,
                    )
                else:
                    # Legacy model (25 stat, no NLP dims)
                    self._is_v7          = False
                    self._n_stat         = n_stat
                    self._stat_feat_names = _STAT_FEAT_NAMES_LEGACY
                    logger.info(
                        "SmartSchemaInferer: legacy model loaded (%d stat features).",
                        n_stat,
                    )
            else:
                # No registry → assume legacy 25-feature model
                self._is_v7          = False
                self._stat_feat_names = _STAT_FEAT_NAMES_LEGACY
                logger.info(
                    "SmartSchemaInferer: no registry found — assuming legacy 25-feat model."
                )

        except Exception as exc:
            logger.warning("SmartSchemaInferer: model load failed (%s) — heuristic mode.", exc)
            self._model = None

    # ── Feature vector builder ─────────────────────────────────────────────────

    def _build_feature_vector(
        self, series: pd.Series, col_name: str
    ) -> np.ndarray:
        """
        Build the feature vector that matches what the model was trained on.

        v7 model  → 58 dims: [30 stat | 28 NLP embedding]
        Legacy    → 25 dims: [25 stat]

        Handles all column types including:
        - Integer-indexed columns (col_name=0 → str("0"))
        - 90-95% null columns (null_rate=0.95, stats fallback gracefully)
        - String/datetime/numeric/mixed columns
        """
        col_name = str(col_name)   # integer-indexed column guard

        if self._is_v7:
            stat = _extract_stat_features_v7(series)
            stat_vec = np.array([stat.get(k, 0.0) for k in _STAT_FEAT_NAMES_V7],
                                 dtype=np.float32)
            nlp_vec  = self._nlp.embed(col_name) if self._nlp else np.zeros(28, dtype=np.float32)
            return np.concatenate([stat_vec, nlp_vec])
        else:
            stat = _extract_stat_features_legacy(series, col_name)
            feat_vec = np.array([stat.get(k, 0.0) for k in _STAT_FEAT_NAMES_LEGACY],
                                 dtype=np.float32)
            # Pad if model expects more features than we have
            n_expected = getattr(self._model, "n_features_in_", len(feat_vec))
            if len(feat_vec) < n_expected:
                feat_vec = np.pad(feat_vec, (0, n_expected - len(feat_vec)))
            return feat_vec[:n_expected]

    # ── Public API ─────────────────────────────────────────────────────────────

    def infer(self, series: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Infer semantic type of a single column via 3-layer cascade.

        Processes ALL column names including:
        - Normal string names: "customer_id", "transaction_amount"
        - CamelCase: "CustomerAge" → normalised to "customer age"
        - Integer indices: 0, 1, 2 → treated as "0", "1", "2"
        - Deeply missing columns: 90-95% null → handled, null_rate feature captures it

        Returns
        -------
        {
          "semantic_type": str,   # one of 21 SEMANTIC_LABELS
          "confidence":    float, # 0.0 – 1.0
          "method":        str,   # nlp_name | nlp_hint | ml | heuristic | nlp_rescue
          "nlp_tags":      list,  # all regex tags matched from column name
          "model_version": str,   # "v7" | "legacy" | "heuristic"
        }
        """
        col_name  = str(col_name)  # integer-indexed column safety
        nlp_tags  = _nlp_tags_from_name(col_name)
        nlp_hint  = nlp_tags[0] if nlp_tags else None
        model_ver = "v7" if self._is_v7 else ("legacy" if self._model else "heuristic")

        # ── Layer 1 Tier A: definitive NLP name match ─────────────────────────
        if nlp_hint and nlp_hint in _NLP_TIER_A:
            logger.debug("SmartSchemaInferer [NLP-A] col='%s' → '%s'", col_name, nlp_hint)
            return {
                "semantic_type": nlp_hint, "confidence": 0.95,
                "method": "nlp_name", "nlp_tags": nlp_tags,
                "model_version": model_ver,
            }

        # ── Layer 2: ML model ─────────────────────────────────────────────────
        if self._model is not None:
            try:
                fv       = self._build_feature_vector(series, col_name)
                proba    = self._model.predict_proba(fv.reshape(1, -1))[0]
                pred_idx = int(np.argmax(proba))
                sem_type = str(self._encoder.inverse_transform([pred_idx])[0])
                conf     = float(proba[pred_idx])

                if sem_type != "unknown" and conf >= self._ML_CONF_THRESHOLD:
                    logger.debug(
                        "SmartSchemaInferer [ML/%s] col='%s' → '%s' (conf=%.2f)",
                        model_ver, col_name, sem_type, conf,
                    )
                    return {
                        "semantic_type": sem_type, "confidence": round(conf, 4),
                        "method": "ml", "nlp_tags": nlp_tags,
                        "model_version": model_ver,
                    }

                # ML uncertain → try Tier-B NLP hint before heuristic
                if nlp_hint and nlp_hint in _NLP_TIER_B:
                    logger.debug(
                        "SmartSchemaInferer [NLP-B] col='%s' ML uncertain → '%s'",
                        col_name, nlp_hint,
                    )
                    return {
                        "semantic_type": nlp_hint, "confidence": 0.75,
                        "method": "nlp_hint", "nlp_tags": nlp_tags,
                        "model_version": model_ver,
                    }

            except Exception as exc:
                logger.warning(
                    "SmartSchemaInferer ML inference failed for col='%s': %s", col_name, exc
                )
                if nlp_hint and nlp_hint in _NLP_TIER_B:
                    return {
                        "semantic_type": nlp_hint, "confidence": 0.75,
                        "method": "nlp_hint", "nlp_tags": nlp_tags,
                        "model_version": "heuristic",
                    }
        else:
            # No ML model — Tier-B hint before heuristic
            if nlp_hint and nlp_hint in _NLP_TIER_B:
                return {
                    "semantic_type": nlp_hint, "confidence": 0.75,
                    "method": "nlp_hint", "nlp_tags": nlp_tags,
                    "model_version": "heuristic",
                }

        # ── Layer 3: Statistical heuristic ────────────────────────────────────
        heuristic_type = _heuristic_infer(series, col_name)

        # Last resort: NLP rescue if heuristic gives 'unknown'
        if heuristic_type == "unknown" and nlp_hint:
            logger.debug(
                "SmartSchemaInferer [NLP-rescue] col='%s' → '%s'", col_name, nlp_hint
            )
            return {
                "semantic_type": nlp_hint, "confidence": 0.55,
                "method": "nlp_rescue", "nlp_tags": nlp_tags,
                "model_version": "heuristic",
            }

        return {
            "semantic_type": heuristic_type,
            "confidence":    0.70 if heuristic_type != "unknown" else 0.0,
            "method":        "heuristic" if self._model is None else "heuristic_fallback",
            "nlp_tags":      nlp_tags,
            "model_version": "heuristic",
        }

    def enrich_schema(
        self,
        df:          pd.DataFrame,
        schema_dict: Dict[str, str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Enrich EVERY column in schema_dict with semantic type, confidence, NLP tags.

        Handles:
        - All column names including integer-indexed (0, 1, 2 ...)
        - Columns that are 90-95% null (null_rate is a valid feature)
        - Columns absent from df (graceful unknown result)

        Parameters
        ----------
        df          : DataFrame being analysed
        schema_dict : {column_name: dtype_str}

        Returns
        -------
        {
          column_name: {
            "dtype":         str,
            "semantic_type": str,
            "confidence":    float,
            "method":        str,
            "nlp_tags":      List[str],
            "model_version": str,
          }
        }
        """
        enriched: Dict[str, Dict[str, Any]] = {}
        for col, dtype in schema_dict.items():
            col_str = str(col)   # integer-indexed column safety
            if col not in df.columns and col_str not in df.columns:
                enriched[col_str] = {
                    "dtype": dtype, "semantic_type": "unknown",
                    "confidence": 0.0, "method": "not_found",
                    "nlp_tags": [], "model_version": "n/a",
                }
                continue
            actual_col = col if col in df.columns else col_str
            result = self.infer(df[actual_col], col_str)
            enriched[col_str] = {
                "dtype":         dtype,
                "semantic_type": result["semantic_type"],
                "confidence":    result["confidence"],
                "method":        result["method"],
                "nlp_tags":      result["nlp_tags"],
                "model_version": result.get("model_version", "unknown"),
            }
        logger.info(
            "SmartSchemaInferer: enriched %d columns  method=%s  v7=%s",
            len(enriched), self._method, self._is_v7,
        )
        return enriched
