# -*- coding: utf-8 -*-
"""
BERTopic DTM Runner + Life2Vec Export
-------------------------------------
End-to-end pipeline to:
  1) Load a RUN folder of participant CSVs (synthetic or real),
  2) Train a BERTopic model at the COHORT level,
  3) Optionally build "MetaTopics" (clusters of leaf topics) for cross-person consistency,
  4) Apply the cohort model to each participant and export Life2Vec-ready CSVs,
  5) Concatenate a cohort-level Life2Vec CSV,
  6) Save model artifacts, plots, and manifests under <repo>/models and <repo>/outputs.

How to run (Spyder):
  1) Edit the CONFIG section below (paths, modes, toggles).
  2) Run this script. If ASK_FOR_RUN_DIR=True, pick the RUN folder (e.g., data/TRAIN_*).
  3) Watch the console. On success, we print the absolute paths to outputs/models folders.

Input EXPECTATIONS:
  - A RUN folder contains per-participant CSV files (e.g., P001.csv, P002.csv, ...),
  - Each CSV has at least a text column (TEXT_COL). If a time column (TIME_COL) exists,
    topics-over-time and Life2Vec RECORD_DATE will be derived from it.

Life2Vec CSV format (per participant and cohort):
  USER_ID              int     (e.g., P001 -> 1)
  RECORD_DATE          str     YYYY-MM-DD (date only)
  Text                 str     original text (cleaned internally for modeling)
  MetaTopic1..M        {0,1}   hard assignment (1 if doc’s best leaf topic falls in MetaTopic m)
  MetaTopic1_prob..M   float   soft score, sum of leaf-topic confidences in cluster m (not normalized)

Version history:
  v1.0  2025-10-21  Initial pipeline with BERTopic fit, per-participant transform, basic outputs.
  v1.1  2025-10-22  Added manifests, folder conventions, cohort/participant modes, helpers.
  v1.2  2025-10-23  Per-participant Life2Vec export; added extra plots; improved error handling.
  v1.3  2025-10-24  Master error log; robust guards for empty/outlier docs; path fixes (repo outputs/models).
  v1.4  2025-10-27  MetaTopics:
                    - Exact-M clustering option (META_N_CLUSTERS) via c-TF-IDF or hierarchy cutoff (META_CUTOFF).
                    - Cohort-level Life2Vec CSV (vertical concat of participant L2V CSVs).
                    - L2V columns renamed to USER_ID (numeric) and RECORD_DATE (YYYY-MM-DD).
  v1.5  2025-10-29  Scale & stability:
                    - **SVD-wrapped embedder** so both docs & words are in the same reduced space
                      (keeps KeyBERT labels + speeds UMAP/HDBSCAN).
                    - **Hierarchical-docs plot downsampling** for very large cohorts (plot only).
                    - Clearer config docs and inline comments with valid options.

Dependencies:
  - Python 3.10+
  - pandas, numpy, scikit-learn, joblib
  - sentence-transformers, torch (GPU optional), bertopic, hdbscan, umap-learn
  - (Our helpers) io_utils.py in <repo>/src

Notes:
  - MetaTopic probabilities are **aggregated leaf-topic confidences**, not a normalized distribution.
    If you need normalized probabilities, we can add an optional row-normalization or remainder bucket.
  - Downsampling the *hierarchical documents plot* does not affect the trained model or MetaTopics —
    it only speeds up figure generation.
"""
#%% ===================== Imports =====================
from __future__ import annotations

import os, sys, re, json, traceback, platform, warnings
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd

from pathlib import Path

# BERTopic stack
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.backend import BaseEmbedder

# Embeddings / ML utils
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from joblib import dump, load

# Our repo helpers (adjust import names here if your io_utils differs)
from io_utils import (
    ensure_dir, choose_open_file, choose_save_dir,
    open_folder_in_explorer, build_default_basename, build_timestamped_folder,
    write_manifest, get_git_info
)

# Silence pandas performance warning; we bulk-concat to avoid fragmentation anyway
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


#%% ===================== CONFIG =====================
"""
Every variable below has:
  - a brief description,
  - valid options/range,
  - and a sensible default for synthetic data testing.
Change them here and re-run.
"""
# ---- Run mode & paths ----
RUN_MODE              = "both"   # {"cohort","participants","both"} – which pipeline parts to run
ASK_FOR_RUN_DIR       = True     # {True,False} – if True, shows a dialog to pick RUN folder
INPUT_RUN_DIR         = None     # {None or str/Path} – if not asking, path to RUN folder (contains per-participant CSVs)

# Repo-rooted outputs/models folders (computed below; do not change unless you know what you’re doing)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
OUTPUTS_ROOT = REPO_ROOT / "outputs"  # where figures/manifests/L2V CSVs go
MODELS_ROOT  = REPO_ROOT / "models"   # where BERTopic models/checkpoints go

# ---- Data schema ----
TEXT_COL              = "text"        # {str} – column name with the raw text
TIME_COL              = "timestamp"   # {str or None} – if present, used for ToT & RECORD_DATE in L2V (YYYY-MM-DD)

# ---- Modeling: embeddings & representation ----
EMBEDDING_NAME        = "sentence-transformers/all-MiniLM-L6-v2"  # {HF model name} – 384D sentence embeddings
USE_KEYBERT_REP       = True   # {True,False} – use KeyBERTInspired to label topics with readable words
MIN_TOPIC_SIZE        = 10     # {int >= 2} – HDBSCAN cluster size threshold (fewer => more topics; more => fewer)

# ---- Topics over time (cohort) ----
RUN_TOPICS_OVER_TIME  = True   # {True,False} – compute/save topics-over-time for cohort
NR_BINS               = 20     # {int >= 2} – number of time bins in ToT
TOT_MIN_VALID         = 50     # {int >= 1} – minimum valid timestamps to run ToT

# ---- MetaTopics (merge leaf topics for cross-person consistency) ----
META_N_CLUSTERS       = 10     # {int >=1 or None} – exact number of MetaTopics; if None, use META_CUTOFF below
META_CUTOFF           = 0.30   # {float >0} – dendrogram distance cutoff when META_N_CLUSTERS is None

# ---- Life2Vec export ----
BUILD_COHORT_L2V      = True   # {True,False} – after per-participant L2V, also save a cohort-level L2V CSV
L2V_INCLUDE_SOFT      = True   # {True,False} – include MetaTopic*_prob columns
L2V_OUTLIER_POLICY    = "zero" # {"zero","drop","keep_all"} – how to handle docs with no assigned leaf topic

# ---- Scale knobs ----
PRE_REDUCE_EMBEDDINGS = True   # {True,False} – fit TruncatedSVD and wrap embedder so docs/words use reduced space
SVD_N_COMPONENTS      = 150    # {int 50..256} – reduced dim; 100–200 good for MiniLM(384D)
SVD_RANDOM_STATE      = 42     # {int} – reproducible SVD

HIER_DOCS_MAX_N       = 10000  # {int >=1 or None} – show at most this many docs in hierarchical-docs plot (plot only)

# ---- Misc / Device ----
EMBED_DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"  # {"cuda","cpu"}
BASE_PREFIX           = "bertopic_run"  # {str} – prefix for run basenames/manifests

#%% ===================== Small helpers =====================
def get_env_versions():
    info = {
        "python": sys.version.split()[0] if hasattr(sys, "version") else None,
        "platform": platform.platform(),
    }
    try:
        import bertopic; info["bertopic"] = getattr(bertopic, "__version__", "unknown")
    except Exception:
        pass
    for pkg in ["torch","sentence_transformers","umap","hdbscan","pandas","numpy","plotly","scikit_learn"]:
        try:
            mod = __import__(pkg if pkg!="sentence_transformers" else "sentence_transformers")
            ver = getattr(mod,"__version__",None) or getattr(mod,"version",None)
            info[pkg] = ver
        except Exception:
            info[pkg] = None
    try:
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_version"] = getattr(torch.version,"cuda",None)
        info["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    return info

def clean_text(s: str) -> str:
    """Very light cleaning; keep it conservative so we don't distort meaning."""
    if not isinstance(s, str):
        s = str(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def pid_from_filename(p: Path) -> Optional[str]:
    """Try to extract a participant id like 'P001' from filename; else return stem."""
    m = re.search(r"(P\d+)", p.stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else p.stem

def numeric_user_id(pid: str) -> Optional[int]:
    """Map 'P001' -> 1, '007' -> 7, else None if no digits found."""
    m = re.search(r"\d+", str(pid))
    return int(m.group(0)) if m else (int(pid) if str(pid).isdigit() else None)

def collect_run_csvs(run_dir: Path) -> List[Path]:
    return sorted([p for p in run_dir.glob("*.csv") if p.is_file()])

def concat_participants(csv_paths: List[Path]) -> pd.DataFrame:
    frames = []
    for p in csv_paths:
        try:
            dfp = pd.read_csv(p)
            dfp["__pid__"] = pid_from_filename(p)
            frames.append(dfp)
        except Exception as e:
            print(f"   WARN: cannot read {p.name}: {e}")
    return pd.concat(frames, axis=0, ignore_index=True) if frames else pd.DataFrame()

#%% ========== SVD-wrapped embedder (single reduced space for docs & words) ==========
class SVDBackend(BaseEmbedder):
    """
    Wrap a SentenceTransformer so .embed (docs) and .embed_words (tokens) return SVD-reduced vectors.
    Ensures KeyBERTInspired (word embeddings) matches the document embedding dimensionality.
    """
    def __init__(self, base_model: SentenceTransformer, svd: TruncatedSVD):
        self.base = base_model
        self.svd = svd

    def embed(self, documents, verbose: bool = False):
        X = self.base.encode(
            documents,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=verbose
        )
        return self.svd.transform(X)

    def embed_words(self, words, verbose: bool = False):
        X = self.base.encode(
            words,
            batch_size=512,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        return self.svd.transform(X)

#%% ===================== MetaTopic utilities =====================
class DSU:
    """Disjoint Set Union for merging topics under a distance cutoff."""
    def __init__(self, items: List[int]):
        self.parent = {i: i for i in items}
        self.rank   = {i: 0   for i in items}
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def build_meta_topics_from_hierarchy(topic_model: BERTopic,
                                     docs: List[str],
                                     cutoff: float) -> Tuple[List[List[int]], Dict[int, str]]:
    """Cluster by cutting the dendrogram at 'cutoff' distance. Returns (clusters, labels)."""
    h_df = topic_model.hierarchical_topics(docs)
    if h_df is None or h_df.empty:
        return [], {}
    leaf_topics = sorted(t for t in topic_model.get_topics().keys() if isinstance(t, int) and t >= 0)
    if not leaf_topics:
        return [], {}

    dsu = DSU(leaf_topics)
    for _, r in h_df.sort_values("Distance").iterrows():
        try:
            d = float(r["Distance"])
            if d > cutoff:
                break
            a = int(r["Topic_1"]); b = int(r["Topic_2"])
            if a >= 0 and b >= 0:
                dsu.union(a, b)
        except Exception:
            continue

    buckets: Dict[int, List[int]] = defaultdict(list)
    for t in leaf_topics:
        buckets[dsu.find(t)].append(t)

    clusters = [sorted(v) for _, v in sorted(buckets.items(), key=lambda kv: (len(kv[1]), min(kv[1])), reverse=True)]

    labels: Dict[int, str] = {}
    for i, cl in enumerate(clusters, start=1):
        counter = Counter()
        for tid in cl:
            words = topic_model.get_topic(tid) or []
            for w, wgt in words[:10]:
                try: counter[w] += float(wgt)
                except Exception: counter[w] += 1.0
        top_words = [w for w, _ in counter.most_common(3)]
        labels[i] = " | ".join(top_words) if top_words else f"Meta_{i}"
    return clusters, labels


def find_cutoff_for_n_clusters(h_df: pd.DataFrame,
                               leaf_topics: List[int],
                               target_m: int) -> float:
    """Choose a cutoff so that using links with Distance <= cutoff yields ~target_m clusters."""
    if not isinstance(target_m, int) or target_m < 1:
        return 0.0
    if h_df is None or h_df.empty or not leaf_topics:
        return 0.0

    dsu = DSU(leaf_topics)
    n_clusters = len(leaf_topics)

    for _, row in h_df.sort_values("Distance").iterrows():
        try:
            d = float(row["Distance"])
            a = int(row["Topic_1"]); b = int(row["Topic_2"])
        except Exception:
            continue
        dsu.union(a, b)
        n_clusters = len({dsu.find(t) for t in leaf_topics})
        if n_clusters <= target_m:
            return max(0.0, d - 1e-9)
    try:
        return float(h_df["Distance"].max())
    except Exception:
        return 1.0


def cluster_topics_exact_M(topic_model: BERTopic, M: int) -> Tuple[List[List[int]], Dict[int, str]]:
    """
    Produce exactly M MetaTopics by clustering the per-topic c-TF-IDF rows.
    Returns (clusters, labels) with clusters as lists of leaf topic IDs.
    """
    leaf_ids = sorted([t for t in topic_model.get_topics().keys() if isinstance(t, int) and t >= 0])
    if not leaf_ids:
        return [], {}

    ctfidf = topic_model.c_tf_idf_
    # Ensure sparse CSR for normalization and clustering
    from scipy import sparse as sp
    X = ctfidf if sp.issparse(ctfidf) else sp.csr_matrix(ctfidf)
    Xn = normalize(X, norm="l2", axis=1)

    try:
        agg = AgglomerativeClustering(n_clusters=max(1, int(M)), metric="cosine", linkage="average")
    except TypeError:
        agg = AgglomerativeClustering(n_clusters=max(1, int(M)), affinity="cosine", linkage="average")

    labels_arr = agg.fit_predict(Xn.toarray() if hasattr(Xn, "toarray") else Xn)
    clusters_map: Dict[int, List[int]] = defaultdict(list)
    for tid, lab in zip(leaf_ids, labels_arr):
        clusters_map[int(lab)].append(int(tid))

    clusters: List[List[int]] = [sorted(v) for _, v in sorted(
        clusters_map.items(), key=lambda kv: (len(kv[1]), min(kv[1])), reverse=True
    )]

    labels: Dict[int, str] = {}
    for idx, cl in enumerate(clusters, start=1):
        counter = Counter()
        for tid in cl:
            words = topic_model.get_topic(tid) or []
            for w, wgt in words[:10]:
                try: counter[w] += float(wgt)
                except Exception: counter[w] += 1.0
        top_words = [w for w, _ in counter.most_common(3)]
        labels[idx] = " | ".join(top_words) if top_words else f"Meta_{idx}"

    return clusters, labels


#%% ===================== Plots (with hierarchical-docs downsampling) =====================

def run_hierarchy_and_save(topic_model: BERTopic,
                           docs: List[str],
                           out_dir: Path,
                           run_base: str,
                           topics_present: Optional[List[int]] = None) -> Dict[str, Union[str, List, Dict]]:
    """Compute & save hierarchical topics/plots. Never abort the run; returns paths and any warnings/errors."""
    out: Dict[str, Union[str, List, Dict]] = {}

    if topics_present is None:
        try:
            topics_present = [t for t in topic_model.get_topics().keys() if isinstance(t, int) and t >= 0]
        except Exception:
            topics_present = []
    if len(topics_present) < 2:
        out["warning"] = "skipped_hierarchy: fewer than 2 topics"
        return out

    try:
        h_topics = topic_model.hierarchical_topics(docs)
    except Exception as e:
        err_path = out_dir / f"{run_base}_hierarchy_ERROR.json"
        write_manifest({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, err_path)
        out["error"] = f"see {err_path.name}"
        return out

    if h_topics is None or len(h_topics) == 0:
        out["warning"] = "skipped_hierarchy: no hierarchical links found"
        h_csv = out_dir / f"{run_base}_hierarchical_topics.csv"
        pd.DataFrame().to_csv(h_csv, index=False)
        out["hierarchical_topics_csv"] = str(h_csv.resolve())
        return out

    h_csv = out_dir / f"{run_base}_hierarchical_topics.csv"
    h_topics.to_csv(h_csv, index=False)
    out["hierarchical_topics_csv"] = str(h_csv.resolve())

    try:
        fig_h = topic_model.visualize_hierarchy(hierarchical_topics=h_topics)
        h_html = out_dir / f"{run_base}_hierarchy.html"
        fig_h.write_html(h_html)
        out["hierarchy_html"] = str(h_html.resolve())
    except Exception as e:
        out["hierarchy_error"] = f"{type(e).__name__}: {e}"

    try:
        sample_rate = 1.0
        if HIER_DOCS_MAX_N is not None and len(docs) > HIER_DOCS_MAX_N:
            sample_rate = max(0.0, min(1.0, HIER_DOCS_MAX_N / float(len(docs))))
        fig_hd = topic_model.visualize_hierarchical_documents(
            docs, hierarchical_topics=h_topics, topics=topics_present, sample=sample_rate
        )
        hd_html = out_dir / f"{run_base}_hier_docs.html"
        fig_hd.write_html(hd_html)
        out["hierarchical_docs_html"] = str(hd_html.resolve())
        if sample_rate < 0.999:
            out["hier_docs_note"] = f"downsampled to {sample_rate:.4f} of documents for plotting only"
    except Exception as e:
        out["hier_docs_error"] = f"{type(e).__name__}: {e}"

    try:
        tree_txt = topic_model.get_topic_tree(h_topics)
        tree_path = out_dir / f"{run_base}_topic_tree.txt"
        with open(tree_path, "w", encoding="utf-8") as f:
            f.write(tree_txt)
        out["topic_tree_txt"] = str(tree_path.resolve())
    except Exception as e:
        out["topic_tree_error"] = f"{type(e).__name__}: {e}"

    return out


#%% ===================== Life2Vec export =====================

def export_l2v_for_participant(cohort_model: BERTopic,
                               meta_clusters: List[List[int]],
                               meta_labels: Dict[int, str],
                               pid_csv: Path,
                               out_path: Path,
                               text_col: str = "text",
                               time_col: str = "timestamp",
                               include_soft: bool = True,
                               outlier_policy: str = "zero") -> Path:
    """
    Transform one participant with the trained cohort_model and export Life2Vec CSV.
    Assumes cohort_model.embedding_model is already the correct backend (SVD-wrapped if enabled).
    """
    df = pd.read_csv(pid_csv)
    if df.empty:
        raise ValueError(f"{pid_csv.name}: no rows")
    if text_col not in df.columns:
        raise ValueError(f"{pid_csv.name}: TEXT_COL '{text_col}' not found.")

    docs = df[text_col].astype(str).map(clean_text).tolist()
    topics, probs = cohort_model.transform(docs)

    # Build mapping from leaf topic -> MetaTopic index (1-based)
    leaf_to_meta: Dict[int, int] = {}
    for m_idx, members in enumerate(meta_clusters, start=1):
        for t in members:
            leaf_to_meta[int(t)] = m_idx

    n = len(docs)
    M = len(meta_clusters)

    # Hard (0/1): 1 where the best leaf topic falls inside the meta cluster
    hard = np.zeros((n, M), dtype=int)
    for i, t in enumerate(topics):
        if t == -1:
            continue
        m = leaf_to_meta.get(int(t), None)
        if m is not None:
            hard[i, m-1] = 1

    # Soft (prob): sum of leaf-topic confidences per cluster
    soft = None
    if include_soft and probs is not None and isinstance(probs, np.ndarray) and probs.ndim == 2:
        soft = np.zeros((n, M), dtype=float)
        for t_leaf, m in leaf_to_meta.items():
            if t_leaf < probs.shape[1]:
                soft[:, m-1] += probs[:, t_leaf]

    # Build base Life2Vec DataFrame (bulk, with required renames/formats)
    pid = pid_from_filename(pid_csv)
    user_id = numeric_user_id(pid)

    if time_col in df.columns:
        dt_series = pd.to_datetime(df[time_col], errors="coerce")
        record_date = dt_series.dt.strftime("%Y-%m-%d")
    else:
        record_date = pd.Series([pd.NA] * len(df))

    base = pd.DataFrame({
        "USER_ID": user_id,
        "RECORD_DATE": record_date,
        "Text": df[text_col].astype(str)
    })

    bin_df  = pd.DataFrame({f"MetaTopic{i+1}": hard[:, i] for i in range(M)})
    frames = [base, bin_df]

    if include_soft and soft is not None:
        prob_df = pd.DataFrame({f"MetaTopic{i+1}_prob": soft[:, i] for i in range(M)})
        frames.append(prob_df)

    out_df = pd.concat(frames, axis=1)

    if outlier_policy == "drop":
        keep = (hard.sum(axis=1) > 0)
        out_df = out_df.loc[keep].reset_index(drop=True)

    ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    return out_path


#%% ===================== Main cohort processing =====================

def process_cohort(run_dir: Path,
                   outputs_run_dir: Path,
                   models_run_dir: Path,
                   embedding_model: SentenceTransformer,
                   rep_model: Optional[KeyBERTInspired],
                   config: dict) -> Dict[str, Union[str, int]]:
    out_dir_cohort = ensure_dir(outputs_run_dir / "COHORT")
    model_dir_cohort_root = ensure_dir(models_run_dir / "COHORT")

    csv_paths = collect_run_csvs(run_dir)
    if not csv_paths:
        raise FileNotFoundError("No CSV files found in RUN folder for cohort mode.")

    df_all = concat_participants(csv_paths)
    if df_all.empty:
        raise ValueError("Cohort DataFrame is empty after concatenation.")

    timestamps = df_all[TIME_COL] if TIME_COL in df_all.columns else None
    docs = df_all[TEXT_COL].astype(str).map(clean_text).tolist()

    base_suffix = f"{run_dir.name}_COHORT"
    run_base = build_default_basename(BASE_PREFIX, suffix=base_suffix, with_timestamp=True)

    # ---- SVD-wrapped embedder so docs & words live in the same reduced space ----
    if PRE_REDUCE_EMBEDDINGS:
        print(f"[SVD] Fitting TruncatedSVD to {len(docs)} cohort docs at {SVD_N_COMPONENTS} dims…")
        full_emb = embedding_model.encode(docs, batch_size=64, convert_to_numpy=True, show_progress_bar=True)
        svd_obj = TruncatedSVD(n_components=SVD_N_COMPONENTS, random_state=SVD_RANDOM_STATE).fit(full_emb)
        svd_backend = SVDBackend(embedding_model, svd_obj)

        topic_model = BERTopic(
            embedding_model=svd_backend,
            representation_model=rep_model,
            min_topic_size=max(2, MIN_TOPIC_SIZE),
            calculate_probabilities=True,
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(docs)
    else:
        svd_obj = None
        topic_model = BERTopic(
            embedding_model=embedding_model,
            representation_model=rep_model,
            min_topic_size=max(2, MIN_TOPIC_SIZE),
            calculate_probabilities=True,
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(docs)

    # Save model (and SVD if used)
    model_dir = ensure_dir(model_dir_cohort_root / f"{run_base}_model")
    if PRE_REDUCE_EMBEDDINGS and svd_obj is not None:
        dump(svd_obj, model_dir / "svd.joblib")

    # ---- Topic info & doc info ----
    topic_info = topic_model.get_topic_info()
    n_topics_present = int((topic_info["Topic"] >= 0).sum())
    topic_info_csv = out_dir_cohort / f"{run_base}_topic_info.csv"
    topic_info.to_csv(topic_info_csv, index=False)

    doc_info = topic_model.get_document_info(docs)
    doc_info.insert(0, "row_id", df_all.index.values)
    doc_info_csv = out_dir_cohort / f"{run_base}_document_info.csv"
    doc_info.to_csv(doc_info_csv, index=False)

    # ---- Visualizations ----
    if n_topics_present >= 2:
        try:
            fig_overview = topic_model.visualize_topics()
            overview_html = out_dir_cohort / f"{run_base}_topics_overview.html"
            fig_overview.write_html(overview_html)
        except Exception as e:
            err = out_dir_cohort / f"{run_base}_overview_ERROR.json"
            write_manifest({"error": f"{type(e).__name__}: {e}"}, err)
    else:
        with open(out_dir_cohort / f"{run_base}_overview_SKIPPED.txt", "w", encoding="utf-8") as f:
            f.write(f"Skipped visualize_topics: only {n_topics_present} topic(s).")

    if n_topics_present >= 2:
        try:
            fig_heatmap = topic_model.visualize_heatmap()
            heatmap_html = out_dir_cohort / f"{run_base}_topics_heatmap.html"
            fig_heatmap.write_html(heatmap_html)
        except Exception as e:
            err = out_dir_cohort / f"{run_base}_heatmap_ERROR.json"
            write_manifest({"error": f"{type(e).__name__}: {e}"}, err)
    else:
        with open(out_dir_cohort / f"{run_base}_heatmap_SKIPPED.txt", "w", encoding="utf-8") as f:
            f.write(f"Skipped visualize_heatmap: only {n_topics_present} topic(s).")

    hierarchy_paths = {}
    if n_topics_present >= 2:
        try:
            topics_present = [t for t in topic_model.get_topics().keys() if isinstance(t, int) and t >= 0]
        except Exception:
            topics_present = []
        try:
            hierarchy_paths = run_hierarchy_and_save(topic_model, docs, out_dir_cohort, run_base, topics_present)
        except Exception as e:
            err_path = out_dir_cohort / f"{run_base}_hierarchy_ERROR.json"
            write_manifest({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, err_path)
            hierarchy_paths = {"error": f"see {err_path.name}"}

    # Topics over time (optional)
    if RUN_TOPICS_OVER_TIME and timestamps is not None:
        ts = pd.to_datetime(timestamps, errors="coerce")
        if ts.notna().sum() >= TOT_MIN_VALID:
            try:
                tot = topic_model.topics_over_time(
                    docs, ts.tolist(),
                    nr_bins=min(NR_BINS, max(2, ts.notna().sum()//2)),
                    global_tuning=True,
                    evolution_tuning=True
                )
                tot_csv = out_dir_cohort / f"{run_base}_topics_over_time.csv"
                tot.to_csv(tot_csv, index=False)
                fig_tot = topic_model.visualize_topics_over_time(tot, top_n_topics=10)
                tot_html = out_dir_cohort / f"{run_base}_topics_over_time.html"
                fig_tot.write_html(tot_html)
            except Exception as e:
                err = out_dir_cohort / f"{run_base}_topics_over_time_ERROR.json"
                write_manifest({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, err)
        else:
            with open(out_dir_cohort / f"{run_base}_topics_over_time_SKIPPED.txt", "w", encoding="utf-8") as f:
                f.write(f"Skipped topics_over_time: only {ts.notna().sum()} valid timestamps (min {TOT_MIN_VALID}).")

    # Save model folder
    topic_model.save(
        model_dir,
        serialization="pytorch",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_NAME
    )

    # Manifest
    manifest = {
        "run_base": run_base,
        "timestamps": {"ended": datetime.now().isoformat(timespec="seconds")},
        "input": {"rows": int(len(df_all)), "text_column": TEXT_COL, "time_column_present": TIME_COL in df_all.columns},
        "config": config,
        "environment": get_env_versions(),
        "git": get_git_info(),
        "model_summary": {"n_documents": len(docs), "n_topics": n_topics_present},
        "hierarchy": hierarchy_paths if n_topics_present >= 2 else None,
        "outputs": [{"path": str(p.resolve())} for p in [topic_info_csv, doc_info_csv] if p.exists()]
    }
    write_manifest(manifest, out_dir_cohort / f"{run_base}_manifest.json")

    return {
        "mode": "cohort",
        "rows": int(len(df_all)),
        "n_topics": n_topics_present,
        "out_dir": str(out_dir_cohort.resolve()),
        "model_dir": str(model_dir.resolve()),
        "run_base": run_base
    }


#%% ===================== Main =====================

def main():
    # --- Resolve RUN dir ---
    if ASK_FOR_RUN_DIR or not INPUT_RUN_DIR:
        run_dir = choose_save_dir(title="Select RUN folder (contains participant CSVs)")
    else:
        run_dir = Path(INPUT_RUN_DIR)
    if not run_dir or not Path(run_dir).exists():
        raise FileNotFoundError("No RUN folder selected/found.")

    # --- Make sibling run folders under outputs/ and models/ with SAME timestamp ---
    run_folder_name = build_timestamped_folder(prefix="Out")
    outputs_run_dir = ensure_dir(OUTPUTS_ROOT / run_folder_name)
    models_run_dir  = ensure_dir(MODELS_ROOT  / run_folder_name)

    print("\n=== COHORT MODE ===" if RUN_MODE in ("cohort","both") else
          "=== PARTICIPANTS MODE ===" if RUN_MODE=="participants" else
          "=== MODE: both (cohort + participants) ===")
    print(f"RUN folder    : {Path(run_dir).resolve()}")
    print(f"OUTPUTS folder: {outputs_run_dir.resolve()}")
    print(f"MODELS  folder: {models_run_dir.resolve()}\n")

    # --- Embedding & representation models ---
    print(f"Loading embedding model: {EMBEDDING_NAME} on device: {EMBED_DEVICE}")
    embedding_model = SentenceTransformer(EMBEDDING_NAME, device=EMBED_DEVICE)
    rep_model = KeyBERTInspired() if USE_KEYBERT_REP else None

    # --- Collect basic config snapshot for manifest ---
    cfg = {
        "RUN_MODE": RUN_MODE,
        "TEXT_COL": TEXT_COL,
        "TIME_COL": TIME_COL,
        "EMBEDDING_NAME": EMBEDDING_NAME,
        "USE_KEYBERT_REP": USE_KEYBERT_REP,
        "MIN_TOPIC_SIZE": MIN_TOPIC_SIZE,
        "RUN_TOPICS_OVER_TIME": RUN_TOPICS_OVER_TIME,
        "NR_BINS": NR_BINS,
        "TOT_MIN_VALID": TOT_MIN_VALID,
        "META_N_CLUSTERS": META_N_CLUSTERS,
        "META_CUTOFF": META_CUTOFF,
        "BUILD_COHORT_L2V": BUILD_COHORT_L2V,
        "L2V_INCLUDE_SOFT": L2V_INCLUDE_SOFT,
        "L2V_OUTLIER_POLICY": L2V_OUTLIER_POLICY,
        "PRE_REDUCE_EMBEDDINGS": PRE_REDUCE_EMBEDDINGS,
        "SVD_N_COMPONENTS": SVD_N_COMPONENTS,
        "SVD_RANDOM_STATE": SVD_RANDOM_STATE,
        "HIER_DOCS_MAX_N": HIER_DOCS_MAX_N,
        "EMBED_DEVICE": EMBED_DEVICE,
    }

    cohort_artifacts = None

    # ---- COHORT ----
    if RUN_MODE in ("cohort","both"):
        try:
            cohort_artifacts = process_cohort(
                run_dir=Path(run_dir),
                outputs_run_dir=outputs_run_dir,
                models_run_dir=models_run_dir,
                embedding_model=embedding_model,
                rep_model=rep_model,
                config=cfg
            )
        except Exception as e:
            err = outputs_run_dir / f"COHORT_ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            write_manifest({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}, err)
            print(f"[COHORT ERROR] see {err}")
            # If only cohort mode was requested, bail out:
            if RUN_MODE == "cohort":
                return

    # ---- PARTICIPANTS (Life2Vec export using trained cohort model) ----
    if RUN_MODE in ("participants","both"):
        # We need the trained cohort model dir
        if cohort_artifacts is None:
            # Try to locate the most recent COHORT model under models_run_dir/COHORT
            cand = sorted((models_run_dir / "COHORT").glob("*_model"), key=lambda p: p.stat().st_mtime, reverse=True)
            if not cand:
                raise FileNotFoundError("No cohort model found. Run in 'cohort' or 'both' mode first.")
            cohort_model_dir = cand[0]
            run_base = "unknown"
        else:
            cohort_model_dir = Path(cohort_artifacts["model_dir"])
            run_base = cohort_artifacts.get("run_base", "cohort_model")

        # Load the cohort model; if SVD was used, wrap the embedder again
        try:
            cohort_model = BERTopic.load(cohort_model_dir, embedding_model=embedding_model)
        except Exception:
            print("   WARN: Failed to load cohort model with embedding; trying default load.")
            cohort_model = BERTopic.load(cohort_model_dir)

        svd_path = Path(cohort_model_dir) / "svd.joblib"
        if PRE_REDUCE_EMBEDDINGS and svd_path.exists():
            svd_obj = load(svd_path)
            svd_backend = SVDBackend(embedding_model, svd_obj)
            try:
                cohort_model = BERTopic.load(cohort_model_dir, embedding_model=svd_backend)
            except Exception:
                cohort_model.embedding_model = svd_backend

        # Build MetaTopics (exact-M if specified; else cutoff)
        # We need docs to access hierarchy when using cutoff:
        csv_paths = collect_run_csvs(Path(run_dir))
        df_all = concat_participants(csv_paths)
        docs_cohort = df_all[TEXT_COL].astype(str).map(clean_text).tolist()

        if META_N_CLUSTERS is not None:
            clusters, labels = cluster_topics_exact_M(cohort_model, int(META_N_CLUSTERS))
            print(f"   Formed {len(clusters)} MetaTopics (target={META_N_CLUSTERS}) via exact-M clustering.")
        else:
            h_df = cohort_model.hierarchical_topics(docs_cohort)
            leaf_topics = sorted([t for t in cohort_model.get_topics().keys() if isinstance(t, int) and t >= 0])
            cutoff = find_cutoff_for_n_clusters(h_df, leaf_topics, int(META_N_CLUSTERS)) if META_N_CLUSTERS else META_CUTOFF
            clusters, labels = build_meta_topics_from_hierarchy(cohort_model, docs_cohort, cutoff=float(cutoff))
            print(f"   Formed {len(clusters)} MetaTopics via cutoff={float(cutoff):.4f}.")

        # Export per-participant L2V CSVs
        l2v_dir = ensure_dir(Path(cohort_artifacts["out_dir"] if cohort_artifacts else (outputs_run_dir / "COHORT")) / "L2V")
        part_dir = ensure_dir(l2v_dir / "participants")

        for csv_path in csv_paths:
            pid = pid_from_filename(csv_path) or csv_path.stem
            out_csv = part_dir / f"{pid}_l2v.csv"
            try:
                export_l2v_for_participant(
                    cohort_model, clusters, labels,
                    pid_csv=csv_path, out_path=out_csv,
                    text_col=TEXT_COL, time_col=TIME_COL,
                    include_soft=L2V_INCLUDE_SOFT,
                    outlier_policy=L2V_OUTLIER_POLICY
                )
                print(f"   Saved L2V: {out_csv.name}")
            except Exception as e:
                err = part_dir / f"ERROR_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                write_manifest({
                    "participant": pid, "csv": str(csv_path),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc()
                }, err)
                print(f"   WARN: L2V export failed for {pid}: see {err.name}")

        # Cohort-level L2V CSV (vertical concat of participant L2V CSVs)
        if BUILD_COHORT_L2V:
            frames = []
            for f in sorted(part_dir.glob("*_l2v.csv")):
                try:
                    frames.append(pd.read_csv(f))
                except Exception as e:
                    print(f"   WARN: could not read {f.name} for cohort L2V concat: {e}")
            if frames:
                cohort_l2v = pd.concat(frames, axis=0, ignore_index=True)
                # (Optional) sort for readability
                if {"USER_ID","RECORD_DATE"}.issubset(cohort_l2v.columns):
                    cohort_l2v = cohort_l2v.sort_values(["USER_ID","RECORD_DATE"], kind="stable").reset_index(drop=True)
                cohort_l2v_path = l2v_dir / "COHORT_l2v.csv"
                cohort_l2v.to_csv(cohort_l2v_path, index=False, encoding="utf-8")
                print(f"   Saved cohort L2V: {cohort_l2v_path.name}")
            else:
                print("   WARN: No participant L2V CSVs found; skipping cohort L2V file.")

    print("\n=== Outputs folder ===")
    print(outputs_run_dir.resolve())
    try:
        open_folder_in_explorer(outputs_run_dir)
    except Exception:
        pass


#%% ===================== Entrypoint =====================
if __name__ == "__main__":
    main()
