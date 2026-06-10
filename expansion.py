"""
Kernel Expansion
================

Grows a kernel dictionary into a full diachronic dictionary by running
Local Graph-based Dictionary Expansion (LGDE) — or its SIWO variant —
on per-decade word2vec embeddings, then integrating the per-decade
results into a single deduplicated CSV.

Pipeline position
-----------------
    kernel.py            ->  expansion.py  ->  relevance.py  ->  scoring.py
    (build the kernel)        (this file)       (score it)        (apply)

Method (one decade at a time)
-----------------------------
    1. Load per-decade Google Ngram embeddings.
    2. For each foundation, grow a neighbourhood of each kernel seed,
       using one of three methods:
         - severability  (graph-based; CkNN, parameters `k` and `t`),
         - SIWO          (graph-based; CkNN, parameter `k`),
         - cosine        (graph-free; keep every vocabulary word whose
                          cosine similarity to a seed exceeds
                          `cosine_threshold`).
    3. Write one CSV per decade.

Integration (after all decades finish)
--------------------------------------
    5. Concatenate the per-decade CSVs.
    6. Drop duplicates.
    7. Drop stop words, words containing digits, and short words.
    8. Write a two-column CSV (`foundation`, `word`).
    9. Delete the intermediate per-decade CSVs.

The LGDE backend lives in `lgde_cluster.py`, vendored from
https://github.com/barahona-research-group/LGDE.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import pickle
import shutil
import tempfile
import time
import warnings
from collections import namedtuple
from multiprocessing import Pool
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd

from lgde_cluster import LGDE

warnings.simplefilter(action='ignore', category=FutureWarning)

PathLike = Union[str, Path]


# ----------------------------------------------------------------------
# Defaults and constants
# ----------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEFAULT_EMBEDDING_FOLDER = _HERE / "aux" / "embeddings"
_VALID_METHODS = {"severability", "siwo", "cosine"}


# ----------------------------------------------------------------------
# Embedding-year detection
# ----------------------------------------------------------------------

def _detect_embedding_years(folder: Path) -> list[int]:
    """Infer available decades from embedding files present in `folder`.

    Scans for files matching ``{year}-vocab.pkl`` and returns the
    corresponding years as a sorted list.  Raises ``FileNotFoundError``
    if the folder exists but contains no matching files.
    """
    if not folder.exists():
        raise FileNotFoundError(
            f"Embedding folder not found: {folder}. "
            f"Place '{{year}}-vocab.pkl' and '{{year}}-w.npy' files there "
            f"or pass an alternative path via `embedding_folder`."
        )
    years = sorted(
        int(p.stem.split("-")[0])
        for p in folder.glob("*-vocab.pkl")
        if p.stem.split("-")[0].isdigit()
    )
    if not years:
        raise FileNotFoundError(
            f"No embedding files found in {folder}. "
            f"Expected files named '{{year}}-vocab.pkl'."
        )
    return years
_GRAPH_METHODS = {"severability", "siwo"}

# Stop-word list mirrors `extended_stopwords` from CreateLGDEDictionaries.R:
# the SMART/sklearn stop-word list plus a manual expansion of common
# function words (be-verbs, do-verbs, modals, prepositions, conjunctions,
# articles, determiners, personal pronouns).
_MANUAL_STOPWORDS = (
    "am is are was were be been being "
    "have has had having do does did doing "
    "will would shall should can could may might must "
    "aboard about above across after around as at before behind below "
    "beneath beside besides between beyond but by considering despite "
    "down during except excepting excluding for from in inside into "
    "like minus near of off on onto over past per plus regarding since "
    "than through to toward towards under underneath unlike until up "
    "upon via with within without "
    "and or nor for so yet after although as because before if since "
    "than though unless until when whenever where whereas wherever while "
    "a an the this that these those my your his her its our their "
    "few little much many lot most some any enough all both half either "
    "neither each every other another such what which whose "
    "he she it his hers its they we us them they their theirs our ours"
).split()


def _default_stopwords() -> set[str]:
    """Return the default stop-word set (SMART/sklearn ∪ manual list)."""
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        smart = set(ENGLISH_STOP_WORDS)
    except ImportError:
        smart = set()
    return {w.lower() for w in smart.union(_MANUAL_STOPWORDS)}


# A pickleable bundle for the per-year worker. Named for clarity at the
# call sites (avoids a long positional tuple).
_YearJob = namedtuple(
    "_YearJob",
    [
        "year",
        "embedding_folder",
        "foundation_dict",
        "workdir",
        "method",
        "k",
        "t",
        "cosine_threshold",
        "siwo_timeout",
        "n_jobs_knn",
        "n_workers_community",
    ],
)


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def _read_table(path: PathLike) -> pd.DataFrame:
    """Read a CSV or XLSX file into a DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .xlsx.")


def _load_kernel(path: PathLike) -> dict[str, list[str]]:
    """Load a (foundation, word) table into {foundation: [words]}."""
    df = _read_table(path)
    missing = [c for c in ("foundation", "word") if c not in df.columns]
    if missing:
        raise ValueError(
            f"Kernel file is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    return {
        foundation: df.loc[df["foundation"] == foundation, "word"]
                      .astype(str).tolist()
        for foundation in df["foundation"].unique()
    }


def _load_embeddings(vocab_path: Path, vecs_path: Path):
    """Load a (vocabulary, embedding matrix) pair from one decade."""
    with open(vocab_path, "rb") as f:
        word_list = pickle.load(f)
    word_vecs = np.load(vecs_path)
    return word_list, word_vecs


# ----------------------------------------------------------------------
# Per-decade expansion (called once per decade, possibly in parallel)
# ----------------------------------------------------------------------

def _build_lgde_graph(word_list, word_vecs, k: int, n_jobs: int) -> LGDE:
    """Construct the shared CkNN graph for one decade."""
    base = LGDE(seed_dict=[], word_list=word_list, word_vecs=word_vecs)
    base.construct_network(k=k, n_jobs=n_jobs)
    return base


def _grow_one_foundation_graph(
    foundation: str,
    seeds: list[str],
    base: LGDE,
    method: str,
    t: int,
    siwo_timeout: float,
    n_workers_community: int,
) -> list[dict]:
    """Run LGDE for a single foundation and return rows for the year CSV.

    Used for the graph-based methods (`severability` and `siwo`).
    """
    lgde = LGDE(
        seed_dict=seeds,
        word_list=base.word_list,
        word_vecs=base.word_vecs,
    )
    # Reuse the shared graph instead of rebuilding it.
    lgde.semantic_sim = base.semantic_sim
    lgde.adjacency_ = base.adjacency_

    if method == "siwo":
        lgde.detect_local_communities_siwo(
            timeout=siwo_timeout, disable_tqdm=True
        )
    else:  # severability
        lgde.detect_local_communities_severability(
            t=t, disable_tqdm=True, n_jobs=n_workers_community
        )

    rows: list[dict] = []
    seed_set = set(lgde.seed_dict)
    for w in seed_set:
        rows.append({
            "foundation": foundation, "word": w,
            "is_seed": True, "discovered_from": None,
        })
    for seed_word, community in lgde.semantic_communities_.items():
        for w in community:
            if w not in seed_set:
                rows.append({
                    "foundation": foundation, "word": w,
                    "is_seed": False, "discovered_from": seed_word,
                })
    return rows


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Return a row-normalised copy of `matrix` (zero rows stay zero)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0, 1.0, norms)
    return matrix / safe


def _grow_one_foundation_cosine(
    foundation: str,
    seeds: list[str],
    word_list: list[str],
    normalized_vecs: np.ndarray,
    word_to_idx: dict[str, int],
    threshold: float,
) -> list[dict]:
    """For each seed, return every vocabulary word with cosine >= threshold.

    Used for `method='cosine'`. No graph is built; the row-normalised
    matrix of all decadal embeddings is multiplied against each seed's
    vector and thresholded.
    """
    rows: list[dict] = []
    seed_set = set(seeds)
    for w in seed_set:
        rows.append({
            "foundation": foundation, "word": w,
            "is_seed": True, "discovered_from": None,
        })

    seeds_in_vocab = [(w, word_to_idx[w]) for w in seed_set if w in word_to_idx]
    if not seeds_in_vocab:
        return rows

    seed_matrix = np.vstack([normalized_vecs[i] for _, i in seeds_in_vocab])
    sims = seed_matrix @ normalized_vecs.T   # (n_seeds, n_vocab)

    for row_idx, (seed_word, _) in enumerate(seeds_in_vocab):
        # >= threshold and not the seed itself (its self-similarity is 1).
        matches = np.where(sims[row_idx] >= threshold)[0]
        for idx in matches:
            word = word_list[idx]
            if word not in seed_set:
                rows.append({
                    "foundation": foundation, "word": word,
                    "is_seed": False, "discovered_from": seed_word,
                })
    return rows


def _expand_one_year(job: _YearJob) -> dict:
    """Expand all foundations for a single decade and write the per-year CSV.

    Skips work if the per-year CSV already exists (checkpointing).
    Returns a small status dict so the caller can report progress.
    """
    out_path = job.workdir / f"lgde_expanded_{job.year}.csv"
    if out_path.exists():
        return {"year": job.year, "status": "cached", "path": str(out_path)}

    vocab_path = job.embedding_folder / f"{job.year}-vocab.pkl"
    vecs_path = job.embedding_folder / f"{job.year}-w.npy"
    if not (vocab_path.exists() and vecs_path.exists()):
        return {"year": job.year, "status": "missing"}

    word_list, word_vecs = _load_embeddings(vocab_path, vecs_path)
    word_list = [w.lower() for w in word_list]

    rows: list[dict] = []
    if job.method == "cosine":
        # Graph-free path: row-normalise once, then dot-product per seed.
        normalized_vecs = _normalize_rows(word_vecs)
        word_to_idx = {w: i for i, w in enumerate(word_list)}
        for foundation, seeds in job.foundation_dict.items():
            rows.extend(_grow_one_foundation_cosine(
                foundation=foundation,
                seeds=[s.lower() for s in seeds],
                word_list=word_list,
                normalized_vecs=normalized_vecs,
                word_to_idx=word_to_idx,
                threshold=job.cosine_threshold,
            ))
    else:
        # Graph-based path: build the shared CkNN graph and reuse it.
        base = _build_lgde_graph(
            word_list, word_vecs, k=job.k, n_jobs=job.n_jobs_knn,
        )
        for foundation, seeds in job.foundation_dict.items():
            rows.extend(_grow_one_foundation_graph(
                foundation=foundation,
                seeds=seeds,
                base=base,
                method=job.method,
                t=job.t,
                siwo_timeout=job.siwo_timeout,
                n_workers_community=job.n_workers_community,
            ))

    df = pd.DataFrame(rows)
    df["year"] = job.year
    df.to_csv(out_path, index=False)
    return {"year": job.year, "status": "ok",
            "path": str(out_path), "entries": len(df)}


# ----------------------------------------------------------------------
# Integration (after all per-decade files exist)
# ----------------------------------------------------------------------

def _integrate_year_csvs(
    workdir: Path,
    stopwords: set[str],
    min_word_length: int,
) -> pd.DataFrame:
    """Concatenate per-decade CSVs and apply the cleaning rules.

    Mirrors the R script's `unique() + filter + select(foundation, word)`,
    plus a final dedup on (foundation, word).
    """
    csvs = sorted(workdir.glob("lgde_expanded_*.csv"))
    if not csvs:
        raise RuntimeError(f"No per-year LGDE files found in {workdir}.")

    combined = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)
    combined = combined.drop_duplicates().reset_index(drop=True)

    word = combined["word"].astype(str)
    keep = (
        ~word.str.lower().isin(stopwords)
        & ~word.str.contains(r"\d", regex=True)
        & (word.str.len() >= min_word_length)
    )
    cleaned = combined.loc[keep, ["foundation", "word"]]
    return cleaned.drop_duplicates().reset_index(drop=True)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def expand_kernel_lgde(
    kernel_file: PathLike,
    output_file: PathLike,
    method: str = "severability",
    k: int = 15,
    t: int = 5,
    cosine_threshold: float = 0.3,
    embedding_folder: Optional[PathLike] = None,
    years: Optional[Iterable[int]] = None,
    workdir: Optional[PathLike] = None,
    keep_intermediates: bool = False,
    stopwords: Optional[Iterable[str]] = None,
    min_word_length: int = 3,
    siwo_timeout: float = 60.0,
    n_parallel_years: int = 1,
    n_jobs_knn: int = -1,
    n_workers_community: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Expand a kernel dictionary into a full diachronic dictionary.

    Parameters
    ----------
    kernel_file : str or Path
        CSV or XLSX with `foundation` and `word` columns. Typically the
        output of `get_kernel_dictionary`.
    output_file : str or Path
        Destination CSV for the integrated dictionary.
    method : {'severability', 'siwo', 'cosine'}
        Expansion method. `severability` and `siwo` are graph-based and
        use the LGDE backend; `cosine` is graph-free and includes every
        vocabulary word whose cosine similarity to a seed exceeds
        `cosine_threshold`.
    k : int
        Number of nearest neighbours used to build the CkNN graph.
        Only used when `method` is graph-based.
    t : int
        Markov time for the severability method (ignored otherwise).
    cosine_threshold : float, default 0.3
        Minimum cosine similarity for a vocabulary word to be added to
        a seed's neighbourhood. Only used when `method='cosine'`.
    embedding_folder : str or Path, optional
        Directory containing `{year}-vocab.pkl` and `{year}-w.npy`.
        Defaults to `<repo>/aux/embeddings`.
    years : iterable of int, optional
        Decades for which embeddings are available.
        Defaults to all decades detected in `embedding_folder`.
    workdir : str or Path, optional
        Directory for intermediate per-year CSVs. If None, a temporary
        directory is created and removed at the end.
    keep_intermediates : bool, default False
        If True, the per-year CSVs are preserved in `workdir`.
    stopwords : iterable of str, optional
        Override the default stop-word list.
    min_word_length : int, default 3
        Drop words with fewer than this many characters.
    siwo_timeout : float, default 60
        Timeout (seconds) passed to the SIWO detector.
    n_parallel_years : int, default 1
        Number of decades processed in parallel via multiprocessing.
    n_jobs_knn : int, default -1
        Cores for CkNN construction within a single decade.
    n_workers_community : int, default 1
        Threads for community detection within a single decade.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    pandas.DataFrame
        The integrated, deduplicated dictionary with `foundation` and
        `word` columns. The same content is written to `output_file`.
    """
    # ---- argument validation ----------------------------------------
    if method not in _VALID_METHODS:
        raise ValueError(
            f"Unknown method '{method}'. Choose from {sorted(_VALID_METHODS)}."
        )
    if method in _GRAPH_METHODS and k <= 0:
        raise ValueError("`k` must be a positive integer.")
    if method == "severability" and t <= 0:
        raise ValueError("`t` must be a positive integer for severability.")
    if method == "cosine" and not -1.0 <= cosine_threshold <= 1.0:
        raise ValueError("`cosine_threshold` must lie in [-1, 1].")

    embedding_folder = Path(embedding_folder or _DEFAULT_EMBEDDING_FOLDER)
    years = list(years) if years is not None else _detect_embedding_years(embedding_folder)
    if not embedding_folder.exists():
        raise FileNotFoundError(
            f"Embedding folder not found: {embedding_folder}. "
            f"Provide an alternative via `embedding_folder`."
        )

    stopwords = (
        set(map(str.lower, stopwords))
        if stopwords is not None
        else _default_stopwords()
    )

    # ---- load kernel ------------------------------------------------
    foundation_dict = _load_kernel(kernel_file)
    if verbose:
        n_words = sum(len(v) for v in foundation_dict.values())
        print(f"Kernel loaded: {n_words} words across "
              f"{len(foundation_dict)} foundations.")

    # ---- set up working directory -----------------------------------
    user_supplied_workdir = workdir is not None
    if user_supplied_workdir:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        if method == "cosine":
            prefix = f"lgde_cosine_th{cosine_threshold:g}_"
        else:
            prefix = f"lgde_{method}_k{k}_t{t}_"
        workdir = Path(tempfile.mkdtemp(prefix=prefix))
    if verbose:
        print(f"Working directory: {workdir}")

    # ---- expand each decade -----------------------------------------
    jobs = [
        _YearJob(
            year=year,
            embedding_folder=embedding_folder,
            foundation_dict=foundation_dict,
            workdir=workdir,
            method=method,
            k=k,
            t=t,
            cosine_threshold=cosine_threshold,
            siwo_timeout=siwo_timeout,
            n_jobs_knn=n_jobs_knn,
            n_workers_community=n_workers_community,
        )
        for year in years
    ]

    try:
        t_start = time.time()
        if verbose:
            if method == "cosine":
                method_str = f"cosine (threshold={cosine_threshold:g})"
            elif method == "severability":
                method_str = f"severability (k={k}, t={t})"
            else:
                method_str = f"SIWO (k={k})"
            mode_str = (f"{n_parallel_years} in parallel"
                        if n_parallel_years > 1 else "serial")
            print(f"Expanding across {len(years)} decades using "
                  f"{method_str} ({mode_str})...")
        if n_parallel_years > 1:
            with Pool(processes=n_parallel_years) as pool:
                results = pool.map(_expand_one_year, jobs)
        else:
            results = [_expand_one_year(job) for job in jobs]

        missing = [r["year"] for r in results if r["status"] == "missing"]
        if missing:
            warnings.warn(
                f"Skipped years (missing embeddings): {missing}",
                RuntimeWarning,
            )
        if verbose:
            print(f"Per-year expansion finished in "
                  f"{(time.time() - t_start) / 60:.2f} minutes.")

        # ---- integrate and clean ------------------------------------
        if verbose:
            print("Integrating per-year results...")
        integrated = _integrate_year_csvs(workdir, stopwords, min_word_length)

        out_path = Path(output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        integrated.to_csv(out_path, index=False)
        if verbose:
            print(f"Wrote {len(integrated)} words "
                  f"({integrated['foundation'].nunique()} foundations) "
                  f"to {out_path}")

        return integrated

    finally:
        # Clean up intermediates unless asked to keep them.
        if not keep_intermediates:
            if user_supplied_workdir:
                for p in workdir.glob("lgde_expanded_*.csv"):
                    p.unlink(missing_ok=True)
            else:
                shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Expand a kernel dictionary via LGDE and integrate "
                    "the per-decade outputs into a single CSV."
    )
    parser.add_argument("kernel_file", help="Kernel CSV or XLSX.")
    parser.add_argument("output_file", help="Destination CSV.")
    parser.add_argument("--method", default="severability",
                        choices=sorted(_VALID_METHODS))
    parser.add_argument("--k", type=int, default=15,
                        help="CkNN neighbours (graph methods only).")
    parser.add_argument("--t", type=int, default=5,
                        help="Markov time (severability only).")
    parser.add_argument("--cosine-threshold", type=float, default=0.3,
                        help="Cosine similarity cutoff (cosine method only).")
    parser.add_argument("--embedding-folder", default=None)
    parser.add_argument("--workdir", default=None)
    parser.add_argument("--keep-intermediates", action="store_true")
    parser.add_argument("--n-parallel-years", type=int, default=1)
    parser.add_argument("--n-jobs-knn", type=int, default=-1)
    parser.add_argument("--n-workers-community", type=int, default=1)
    args = parser.parse_args()

    expand_kernel_lgde(
        kernel_file=args.kernel_file,
        output_file=args.output_file,
        method=args.method,
        k=args.k,
        t=args.t,
        cosine_threshold=args.cosine_threshold,
        embedding_folder=args.embedding_folder,
        workdir=args.workdir,
        keep_intermediates=args.keep_intermediates,
        n_parallel_years=args.n_parallel_years,
        n_jobs_knn=args.n_jobs_knn,
        n_workers_community=args.n_workers_community,
    )
