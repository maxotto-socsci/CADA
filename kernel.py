"""
Kernel Dictionary Construction
==============================

Reduces a seed word list (e.g. the Moral Foundations Dictionary) to a
diachronically stable *kernel* dictionary by applying two filters:

  1. **Frequency filter.** Each candidate word must reach a minimum
     Google Ngram frequency in every decade considered.

  2. **Semantic-change filter.** Each candidate's cumulative and
     decade-to-decade cosine drift must lie inside a null distribution
     of construct-neutral words (by default, stop words).

Pipeline position
-----------------
    kernel.py            ->  expansion.py  ->  relevance.py  ->  scoring.py
    (this file)               (expand)          (score)           (apply)

Defaults reproduce the paper's thresholds:
    freq_cutoff       = 1e-6
    change_percentile = 0.95
    years             = [1800, 1810, ..., 1990]

Inputs are CSV or XLSX files with at least a `foundation` and a `word`
column. Words ending in `*` are treated as stems and expanded via
WordNet (and the `english_words` package when available).

Reference distribution
----------------------
The null distribution is derived on the fly from a plain word list
(default: ``aux/stopwords.csv``, which needs only a ``word`` column).
For each reference word, ``total_change`` (cosine distance between the
first and last supplied year) and ``avg_decade_change`` (mean pairwise
change across consecutive years) are computed from the same embeddings
used for seed scoring.  The thresholds are therefore always calibrated
to whatever year range you supply.

Outputs are written as a CSV containing only `foundation` and `word`.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import pickle
import time
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_distances

import nltk
from nltk.corpus import wordnet

warnings.simplefilter(action='ignore', category=FutureWarning)

PathLike = Union[str, Path]


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEFAULT_REFERENCE = _HERE / "aux" / "stopwords.csv"
_DEFAULT_EMBEDDING_FOLDER = _HERE / "aux" / "embeddings"


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


# ----------------------------------------------------------------------
# WordNet / english_words bootstrap
# ----------------------------------------------------------------------

def _ensure_wordnet() -> None:
    """Download WordNet on first use (idempotent)."""
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)


try:
    from english_words import english_words_set as _EW_SET
    _ENGLISH_WORDS_AVAILABLE = True
except ImportError:
    try:
        from english_words import english_words_lower_alpha_set as _EW_SET
        _ENGLISH_WORDS_AVAILABLE = True
    except ImportError:
        _EW_SET = set()
        _ENGLISH_WORDS_AVAILABLE = False


# ----------------------------------------------------------------------
# Stem expansion (only used for input rows ending in '*')
# ----------------------------------------------------------------------

class _StemExpander:
    """Expand wildcard stems like 'kind*' into a list of concrete words.

    Prefers the `english_words` package; otherwise falls back to a
    vocabulary built from WordNet.
    """

    def __init__(self) -> None:
        _ensure_wordnet()
        if _ENGLISH_WORDS_AVAILABLE and _EW_SET:
            self.vocab = {w.lower() for w in _EW_SET if isinstance(w, str)}
        else:
            self.vocab = self._build_wordnet_vocab()

    @staticmethod
    def _build_wordnet_vocab() -> set[str]:
        vocab: set[str] = set()
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                w = lemma.name().replace('_', '').lower()
                if w.isalpha() and len(w) > 1:
                    vocab.add(w)
        return vocab

    def expand(
        self,
        stem: str,
        max_words: int = 15,
        max_length_add: int = 6,
    ) -> list[str]:
        """Return up to `max_words` concrete words matching the stem."""
        base = stem.rstrip('*').lower()
        expansions: set[str] = {base}

        # WordNet lemmas that start with the stem.
        for synset in wordnet.all_synsets():
            for lemma in synset.lemmas():
                w = lemma.name().replace('_', '').lower()
                if (w.startswith(base)
                        and len(w) <= len(base) + max_length_add
                        and w.isalpha()):
                    expansions.add(w)

        # Plus matches from the english_words vocabulary.
        for w in self.vocab:
            w = w.lower()
            if (w.startswith(base)
                    and len(w) <= len(base) + max_length_add
                    and w.isalpha()):
                expansions.add(w)

        return sorted(expansions)[:max_words]

    def expand_dataframe(
        self,
        df: pd.DataFrame,
        word_col: str = 'word',
        foundation_col: str = 'foundation',
    ) -> pd.DataFrame:
        """Expand stems row-by-row, preserving foundation assignments."""
        rows = []
        for _, row in df.iterrows():
            word = str(row[word_col])
            if word.endswith('*'):
                for expanded in self.expand(word):
                    new_row = row.copy()
                    new_row[word_col] = expanded
                    rows.append(new_row)
            else:
                rows.append(row)
        return pd.DataFrame(rows).reset_index(drop=True)


# ----------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------

def _read_table(path: PathLike) -> pd.DataFrame:
    """Read a CSV or XLSX file into a DataFrame."""
    suffix = Path(path).suffix.lower()
    if suffix == '.csv':
        return pd.read_csv(path)
    if suffix in {'.xlsx', '.xls'}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .xlsx.")


def _validate_columns(
    df: pd.DataFrame,
    word_col: str,
    foundation_col: str,
) -> None:
    """Raise if required columns are missing."""
    missing = [c for c in (word_col, foundation_col) if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _load_decadal_embeddings(
    embedding_folder: PathLike,
    years: Iterable[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Load per-decade Google Ngram embeddings (no alignment).

    Expects `{year}-vocab.pkl` and `{year}-w.npy` per decade. Returns a
    dict mapping each year to `{word_lower: vector}`.
    """
    folder = Path(embedding_folder)
    if not folder.exists():
        raise FileNotFoundError(
            f"Embedding folder not found: {folder}. "
            f"Place '{{year}}-vocab.pkl' and '{{year}}-w.npy' files there "
            f"or pass an alternative path via `embedding_folder`."
        )

    embeddings: dict[int, dict[str, np.ndarray]] = {}
    for year in years:
        vocab_path = folder / f"{year}-vocab.pkl"
        vecs_path = folder / f"{year}-w.npy"
        if not (vocab_path.exists() and vecs_path.exists()):
            raise FileNotFoundError(
                f"Missing embedding files for year {year} in {folder}."
            )
        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        vectors = np.load(vecs_path)
        embeddings[year] = {
            w.lower(): vectors[i] for i, w in enumerate(vocab)
        }
    return embeddings


# ----------------------------------------------------------------------
# Google Ngram frequency lookup
# ----------------------------------------------------------------------

_NGRAM_URL = (
    "https://books.google.com/ngrams/json"
    "?content={word}&year_start={start}&year_end={end}&corpus=en&smoothing=0"
)


def _fetch_ngram_series(
    word: str,
    start: int,
    end: int,
    max_retries: int = 5,
) -> Optional[pd.Series]:
    """Fetch a word's Google Ngram series, with simple linear backoff."""
    backoff = 2
    for _attempt in range(max_retries):
        try:
            url = _NGRAM_URL.format(word=word, start=start, end=end)
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if not data or 'timeseries' not in data[0]:
                return None
            return pd.Series(data[0]['timeseries'])
        except Exception:
            time.sleep(backoff)
            backoff += 2
    return None


# ----------------------------------------------------------------------
# Per-word scoring
# ----------------------------------------------------------------------

def _avg_dist_to_centroid(
    word: str,
    foundation: str,
    year: int,
    embeddings: dict,
    foundation_words: dict,
) -> float:
    """Cosine distance from `word` to its foundation's centroid in `year`."""
    members = [
        w for w in foundation_words[foundation]
        if w != word and w in embeddings[year]
    ]
    if not members:
        return np.nan
    member_vecs = [embeddings[year][w] for w in members]
    centroid = np.mean(member_vecs, axis=0).reshape(1, -1)
    word_vec = embeddings[year][word].reshape(1, -1)
    return float(cosine_distances(word_vec, centroid)[0][0])


def _score_word(
    word: str,
    foundation: str,
    embeddings: dict,
    foundation_words: dict,
    years: list[int],
    fetch_ngrams: bool,
) -> Optional[dict]:
    """Compute diachronic stability metrics for a single word.

    Returns None if the word appears in fewer than two of the decades.
    """
    present = [y for y in years if word in embeddings[y]]
    if len(present) < 2:
        return None

    # Per-decade cosine drift and distance-to-centroid.
    decade_changes: list[float] = []
    centroid_dists: list[float] = []
    for j in range(len(years) - 1):
        y1, y2 = years[j], years[j + 1]
        if word in embeddings[y1] and word in embeddings[y2]:
            v1 = embeddings[y1][word].reshape(1, -1)
            v2 = embeddings[y2][word].reshape(1, -1)
            decade_changes.append(float(cosine_distances(v1, v2)[0][0]))

            cd = _avg_dist_to_centroid(
                word, foundation, y1, embeddings, foundation_words
            )
            if not np.isnan(cd):
                centroid_dists.append(cd)

    # Cumulative drift between the first and last supplied year.
    y_first, y_last = years[0], years[-1]
    if word in embeddings[y_first] and word in embeddings[y_last]:
        v0 = embeddings[y_first][word].reshape(1, -1)
        vN = embeddings[y_last][word].reshape(1, -1)
        total_change = float(cosine_distances(v0, vN)[0][0])
    else:
        total_change = np.nan

    stats: dict = {
        'word': word,
        'total_change': total_change,
        'avg_decade_change':
            float(np.mean(decade_changes)) if decade_changes else np.nan,
        'n_decades_present': len(decade_changes),
        'avg_dist_centroid':
            float(np.mean(centroid_dists)) if centroid_dists else np.nan,
    }

    if fetch_ngrams:
        series = _fetch_ngram_series(word, y_first, y_last)
        if series is not None and not series.empty:
            stats['ngram_min_freq'] = float(series.min())
            stats['ngram_freq_change'] = float(
                series.iloc[-1] - series.iloc[0]
            )
        else:
            stats['ngram_min_freq'] = np.nan
            stats['ngram_freq_change'] = np.nan

    return stats


def _score_wordlist(
    df: pd.DataFrame,
    embeddings: dict,
    years: list[int],
    fetch_ngrams: bool,
    word_col: str,
    foundation_col: str,
    verbose: bool,
) -> pd.DataFrame:
    """Score every unique word in `df`."""
    foundation_words = (
        df.groupby(foundation_col)[word_col]
          .apply(lambda s: s.str.lower().tolist())
          .to_dict()
    )
    targets = df[word_col].str.lower().unique().tolist()

    records = []
    for i, w in enumerate(targets):
        if verbose and i % 25 == 0:
            print(f"  scoring {i + 1}/{len(targets)}: {w}")
        foundation = df.loc[
            df[word_col].str.lower() == w, foundation_col
        ].iloc[0]
        rec = _score_word(
            w, foundation, embeddings, foundation_words,
            years, fetch_ngrams,
        )
        if rec is not None:
            records.append(rec)

    return pd.DataFrame(records)


# ----------------------------------------------------------------------
# Reference distribution
# ----------------------------------------------------------------------

def _compute_reference_distribution(
    words: list[str],
    embeddings: dict[int, dict[str, np.ndarray]],
    years: list[int],
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute ``total_change`` and ``avg_decade_change`` for a reference word list.

    Uses the same embeddings and year range as seed scoring, so the
    resulting null distribution is always calibrated to the specific
    time window supplied.  Words present in fewer than two years are
    silently skipped.

    Parameters
    ----------
    words : list of str
        Reference words (e.g. stop words).
    embeddings : dict
        Mapping ``{year: {word: vector}}`` as returned by
        ``_load_decadal_embeddings``.
    years : list of int
        Sorted list of years present in ``embeddings``.
    verbose : bool
        Print progress.

    Returns
    -------
    pd.DataFrame
        Columns: ``word``, ``total_change``, ``avg_decade_change``.
    """
    if verbose:
        print(f"  Computing reference distribution from {len(words)} words "
              f"over years {years[0]}–{years[-1]}...")

    records = []
    for word in words:
        present = [y for y in years if word in embeddings[y]]
        if len(present) < 2:
            continue

        decade_changes: list[float] = []
        for j in range(len(years) - 1):
            y1, y2 = years[j], years[j + 1]
            if word in embeddings[y1] and word in embeddings[y2]:
                v1 = embeddings[y1][word].reshape(1, -1)
                v2 = embeddings[y2][word].reshape(1, -1)
                decade_changes.append(float(cosine_distances(v1, v2)[0][0]))

        y_first, y_last = years[0], years[-1]
        if word in embeddings[y_first] and word in embeddings[y_last]:
            v0 = embeddings[y_first][word].reshape(1, -1)
            vN = embeddings[y_last][word].reshape(1, -1)
            total_change = float(cosine_distances(v0, vN)[0][0])
        else:
            total_change = np.nan

        records.append({
            'word': word,
            'total_change': total_change,
            'avg_decade_change':
                float(np.mean(decade_changes)) if decade_changes else np.nan,
        })

    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(
            "None of the reference words were found in the supplied embeddings. "
            "Check that the reference word list and embedding vocabulary overlap."
        )
    if verbose:
        print(f"  Reference distribution built from "
              f"{len(result)}/{len(words)} words with sufficient coverage.")
    return result


# ----------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------

def _apply_kernel_filters(
    scored: pd.DataFrame,
    reference: pd.DataFrame,
    freq_cutoff: float,
    change_percentile: float,
    apply_freq_filter: bool = True,
) -> pd.DataFrame:
    """Apply frequency and semantic-change filters.

    Mirrors the reduction rules in ``Create_MFT_BASE.r``:
        ngram_min_freq    >= freq_cutoff          (skipped when apply_freq_filter=False)
        total_change      <= P(change_percentile, reference)
        avg_decade_change <= P(change_percentile, reference)
    """
    total_cutoff = np.nanquantile(reference['total_change'], change_percentile)
    decade_cutoff = np.nanquantile(reference['avg_decade_change'], change_percentile)

    mask = (
        (scored['total_change'] <= total_cutoff)
        & (scored['avg_decade_change'] <= decade_cutoff)
    )
    if apply_freq_filter:
        mask = mask & (scored['ngram_min_freq'] >= freq_cutoff)

    return scored.loc[mask].copy()


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def get_kernel_dictionary(
    input_file: PathLike,
    output_file: PathLike,
    embedding_folder: Optional[PathLike] = None,
    reference_file: Optional[PathLike] = None,
    freq_cutoff: float = 1e-6,
    change_percentile: float = 0.95,
    years: Optional[Iterable[int]] = None,
    word_col: str = 'word',
    foundation_col: str = 'foundation',
    expand_stems: bool = True,
    fetch_ngrams: bool = True,
    return_scored: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a diachronic kernel dictionary from a seed word list.

    Parameters
    ----------
    input_file : str or Path
        CSV or XLSX file with at least `foundation_col` and `word_col`.
        Words ending in '*' are treated as stems and expanded.
    output_file : str or Path
        Destination CSV. The reduced two-column dictionary
        (foundation, word) is written here.
    embedding_folder : str or Path, optional
        Directory containing '{year}-vocab.pkl' and '{year}-w.npy' per
        decade. Defaults to `<repo>/aux/embeddings`.
    reference_file : str or Path, optional
        CSV with a ``word`` column listing construct-neutral reference
        words (e.g. stop words).  ``total_change`` and
        ``avg_decade_change`` are computed on the fly from the same
        embeddings and year range used for seed scoring, so the null
        distribution is always calibrated to your specific setup.
        Defaults to ``<repo>/aux/stopwords.csv``.
    freq_cutoff : float, default 1e-6
        Minimum Google Ngram frequency a word must reach in every
        decade between `years[0]` and `years[-1]`.
    change_percentile : float, default 0.95
        Percentile of the reference distribution used as the upper
        bound for both cumulative and decadal cosine change.
    years : iterable of int, optional
        Decade endpoints used for diachronic scoring.
        Defaults to all decades detected in `embedding_folder`.
    word_col, foundation_col : str
        Column names in the input file.
    expand_stems : bool, default True
        Whether to expand wildcard stems before scoring.
    fetch_ngrams : bool, default True
        Whether to query the Google Ngram API for frequency data. Set
        to False to skip the frequency filter entirely.
    return_scored : bool, default False
        If True, also return the full per-word score table.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    kernel_df : pandas.DataFrame
        The reduced two-column kernel dictionary.
    scored_df : pandas.DataFrame, optional
        Only returned when `return_scored=True`; the full pre-filter
        score table.
    """
    # ---- resolve defaults -------------------------------------------
    embedding_folder = Path(embedding_folder or _DEFAULT_EMBEDDING_FOLDER)
    years = list(years) if years is not None else _detect_embedding_years(embedding_folder)
    reference_file = Path(reference_file or _DEFAULT_REFERENCE)

    # ---- load and prepare seed wordlist -----------------------------
    seed_df = _read_table(input_file)
    _validate_columns(seed_df, word_col, foundation_col)
    seed_df[word_col] = seed_df[word_col].astype(str)

    if expand_stems and seed_df[word_col].str.endswith('*').any():
        if verbose:
            print("Expanding wildcard stems...")
        seed_df = _StemExpander().expand_dataframe(
            seed_df, word_col=word_col, foundation_col=foundation_col,
        )
    seed_df[word_col] = seed_df[word_col].str.lower()
    seed_df = seed_df.drop_duplicates(subset=[foundation_col, word_col])

    # ---- load embeddings and score seed words -----------------------
    if verbose:
        print(f"Loading embeddings from {embedding_folder}...")
    embeddings = _load_decadal_embeddings(embedding_folder, years)

    if verbose:
        print(f"Scoring {seed_df[word_col].nunique()} unique seed words...")
    new_scores = _score_wordlist(
        seed_df, embeddings, years, fetch_ngrams,
        word_col, foundation_col, verbose,
    )
    scored = seed_df.merge(
        new_scores, left_on=word_col, right_on='word',
        how='left', suffixes=('', '_dup'),
    )
    if 'word_dup' in scored.columns:
        scored = scored.drop(columns=['word_dup'])

    # ---- build reference (null) distribution ------------------------
    if not reference_file.exists():
        raise FileNotFoundError(
            f"Reference file not found: {reference_file}. "
            f"Supply a CSV with a 'word' column listing reference words "
            f"(e.g. stop words) via `reference_file`."
        )
    ref_df = pd.read_csv(reference_file)
    if 'word' not in ref_df.columns:
        raise ValueError(
            f"Reference file '{reference_file}' must contain a 'word' column."
        )
    ref_words = ref_df['word'].str.lower().dropna().unique().tolist()
    reference = _compute_reference_distribution(
        ref_words, embeddings, years, verbose=verbose
    )

    # ---- apply filters and write output -----------------------------
    kernel = _apply_kernel_filters(
        scored, reference, freq_cutoff, change_percentile,
        apply_freq_filter=fetch_ngrams,
    )

    if verbose:
        print(f"Kernel reduction: {len(scored)} -> {len(kernel)} words "
              f"across {kernel[foundation_col].nunique()} foundations.")

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kernel[[foundation_col, word_col]].to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote kernel dictionary to {out_path}")

    if return_scored:
        return kernel[[foundation_col, word_col]], scored
    return kernel[[foundation_col, word_col]]


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reduce a seed word list to a diachronic kernel dictionary."
    )
    parser.add_argument("input_file", help="CSV or XLSX seed word list.")
    parser.add_argument("output_file", help="Destination CSV.")
    parser.add_argument("--embedding-folder", default=None)
    parser.add_argument("--reference-file", default=None,
                        help="CSV with a 'word' column (stop words / reference list).")
    parser.add_argument("--freq-cutoff", type=float, default=1e-6)
    parser.add_argument("--change-percentile", type=float, default=0.95)
    parser.add_argument("--no-stems", action="store_true",
                        help="Disable wildcard-stem expansion.")
    parser.add_argument("--no-ngrams", action="store_true",
                        help="Skip Google Ngram frequency lookup and filter.")
    args = parser.parse_args()

    get_kernel_dictionary(
        input_file=args.input_file,
        output_file=args.output_file,
        embedding_folder=args.embedding_folder,
        reference_file=args.reference_file,
        freq_cutoff=args.freq_cutoff,
        change_percentile=args.change_percentile,
        expand_stems=not args.no_stems,
        fetch_ngrams=not args.no_ngrams,
    )
