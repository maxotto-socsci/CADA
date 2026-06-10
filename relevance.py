"""
Moral-Relevance Trajectories
============================

Computes per-decade — and optionally per-year — moral-relevance scores
for every (foundation, word) pair in a supplied dictionary.

Pipeline position
-----------------
    kernel.py            ->  expansion.py  ->  relevance.py  ->  scoring.py
    (build kernel)            (expand)          (this file)       (apply)

Method
------
For each decade, every word is compared in the corresponding embedding
space to:

    - the centroid of the moral-foundation kernel for its foundation, and
    - one centroid per category in the neutral reference dictionary.

The number of neutral categories is inferred from the unique values in
the ``foundation`` column of the neutral reference file.  The default
file (``aux/neutral_dictionary.csv``) provides three subspaces
(``neutral.world``, ``neutral.mind``, ``neutral.society``), but any
CSV with ``foundation`` and ``word`` columns works — a single category,
two categories, or more.

Proximity is the temperature-scaled exponential of the Euclidean
distance between the word vector and the centroid:

    proximity(w, c; T) = exp( - || w - c ||_2 / T )

The moral-relevance score is the soft-max share of the moral proximity
relative to the sum of all proximities:

    score(w, F; T) = prox_moral / ( prox_moral + sum_n prox_neutral_n )

This module deliberately omits the eMFD-based temperature optimisation
present in the original calibration script: the caller supplies the
temperature directly via `temp`. Euclidean distance is hard-coded.

If `smooth_interpolate=True` (default), each (word, foundation) decade
trajectory is LOESS-smoothed at bandwidth `loess_span` (default 0.3)
and linearly interpolated to a yearly grid covering `year_start..year_end`.
Otherwise raw per-decade scores are returned.
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import pickle
import warnings
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess

warnings.simplefilter(action='ignore', category=FutureWarning)

PathLike = Union[str, Path]


# ----------------------------------------------------------------------
# Defaults
# ----------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_DEFAULT_EMBEDDING_FOLDER = _HERE / "aux" / "embeddings"
_DEFAULT_KERNEL_FILE = _HERE / "aux" / "CADA_KERNEL.csv"
_DEFAULT_NEUTRAL_FILE = _HERE / "aux" / "neutral_dictionary.csv"


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
            f"Supply an alternative via `embedding_folder`."
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


def _load_two_col_dictionary(path: PathLike) -> pd.DataFrame:
    """Load and tidy a (foundation, word) dictionary."""
    df = _read_table(path)
    missing = [c for c in ('foundation', 'word') if c not in df.columns]
    if missing:
        raise ValueError(
            f"{Path(path).name} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df[['foundation', 'word']].copy()
    df['word'] = df['word'].astype(str).str.lower()
    return df


def _load_decadal_embeddings(
    folder: PathLike,
    years: Iterable[int],
) -> dict[int, dict[str, np.ndarray]]:
    """Load per-decade embeddings as {year: {word_lower: vector}}."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(
            f"Embedding folder not found: {folder}. "
            f"Supply an alternative via `embedding_folder`."
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
# Core scoring
# ----------------------------------------------------------------------

def _proximity(
    word_vec: np.ndarray,
    centroid_vec: np.ndarray,
    temp: float,
) -> float:
    """Temperature-scaled exponential of the Euclidean distance."""
    dist = float(np.linalg.norm(word_vec - centroid_vec))
    return float(np.exp(-dist / temp))


def _centroid(
    words: Iterable[str],
    embeddings: dict[str, np.ndarray],
) -> Optional[np.ndarray]:
    """Mean of available embedding vectors for the listed words."""
    vecs = [embeddings[w] for w in words if w in embeddings]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


def _moral_score(
    word: str,
    foundation: str,
    embeddings: dict[str, np.ndarray],
    moral_words: list[str],
    neutral_centroids: dict[str, Optional[np.ndarray]],
    temp: float,
) -> float:
    """Soft-max share of moral proximity vs. neutral proximities.

    Returns NaN if the word, the foundation kernel, or all neutral
    centroids are unavailable in this embedding space.
    """
    if word not in embeddings:
        return np.nan
    v_word = embeddings[word]

    moral_centroid = _centroid(moral_words, embeddings)
    if moral_centroid is None:
        return np.nan
    prox_moral = _proximity(v_word, moral_centroid, temp)

    prox_neutrals = [
        _proximity(v_word, c, temp)
        for c in neutral_centroids.values() if c is not None
    ]
    if not prox_neutrals or prox_moral <= 0 or np.isnan(prox_moral):
        return np.nan

    total = prox_moral + sum(prox_neutrals)
    if total < 1e-10:
        return np.nan
    return float(prox_moral / total)


def _score_dictionary(
    dictionary: pd.DataFrame,
    embeddings_by_year: dict[int, dict[str, np.ndarray]],
    foundation_groups: dict[str, list[str]],
    neutral_groups: dict[str, list[str]],
    temp: float,
    years: list[int],
) -> pd.DataFrame:
    """Score every (foundation, word) row across every decade."""
    rows = []
    for _, row in dictionary.iterrows():
        word = row['word']
        foundation = row['foundation']
        moral_words = foundation_groups.get(foundation, [])

        scored_row: dict = {'foundation': foundation, 'word': word}
        for year in years:
            emb = embeddings_by_year[year]
            neutral_centroids = {
                f: _centroid(words, emb)
                for f, words in neutral_groups.items()
            }
            score = _moral_score(
                word, foundation, emb,
                moral_words, neutral_centroids, temp,
            )
            scored_row[f'p{year}'] = (
                np.clip(score, 0.0, 1.0) if not np.isnan(score) else np.nan
            )
        rows.append(scored_row)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Smoothing and interpolation
# ----------------------------------------------------------------------

def _smooth_and_interpolate(
    scored: pd.DataFrame,
    loess_span: float,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """LOESS-smooth each trajectory, then interpolate to a yearly grid.

    Each decade is anchored at its midpoint (e.g. p1800 -> 1805) so the
    smoother sees evenly spaced observations.
    """
    decade_cols = sorted(
        (c for c in scored.columns if c.startswith('p')),
        key=lambda c: int(c[1:]),
    )
    decade_midpoints = np.array(
        [int(c[1:]) + 5 for c in decade_cols], dtype=float
    )
    yearly_grid = np.arange(year_start, year_end + 1)

    smoothed_rows = []
    for _, row in scored.iterrows():
        y_vals = row[decade_cols].values.astype(float)
        valid = ~np.isnan(y_vals)
        if valid.sum() < 3:
            continue   # too few points to smooth

        x_valid = decade_midpoints[valid]
        y_valid = y_vals[valid]

        loess_fit = lowess(
            y_valid, x_valid, frac=loess_span, return_sorted=False,
        )
        y_interp = np.clip(
            np.interp(yearly_grid, x_valid, loess_fit), 0.0, 1.0
        )

        smoothed_row: dict = {
            'foundation': row['foundation'],
            'word': row['word'],
        }
        for year, val in zip(yearly_grid, y_interp):
            smoothed_row[f'p{year}'] = float(val)
        smoothed_rows.append(smoothed_row)

    return pd.DataFrame(smoothed_rows)


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def compute_moral_relevance(
    dictionary_file: PathLike,
    output_file: PathLike,
    temp: float = 0.05,
    smooth_interpolate: bool = True,
    loess_span: float = 0.3,
    kernel_file: Optional[PathLike] = None,
    neutral_file: Optional[PathLike] = None,
    embedding_folder: Optional[PathLike] = None,
    years: Optional[Iterable[int]] = None,
    year_start: Optional[int] = None,
    year_end: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute moral-relevance trajectories for a (expanded) dictionary.

    Parameters
    ----------
    dictionary_file : str or Path
        CSV or XLSX with `foundation` and `word` columns; one row per
        word-foundation pair to be scored. Typically the integrated
        output of `expand_kernel_lgde`.
    output_file : str or Path
        Destination CSV for the scored dictionary.
    temp : float, default 0.05
        Soft-max temperature applied to the Euclidean proximities. This
        module does not optimise the temperature; the caller specifies
        it directly.
    smooth_interpolate : bool, default True
        If True, each per-decade trajectory is LOESS-smoothed and
        linearly interpolated to a yearly grid covering
        `year_start..year_end`. If False, raw per-decade scores are
        returned.
    loess_span : float, default 0.3
        LOESS bandwidth (fraction of decades used per local fit).
        Ignored when `smooth_interpolate=False`.
    kernel_file : str or Path, optional
        CSV or XLSX defining the moral-foundation centroids
        (`foundation`, `word`). Defaults to the bundled kernel.
    neutral_file : str or Path, optional
        CSV or XLSX with `foundation` and `word` columns defining the
        neutral reference subspaces. The number of neutral categories is
        inferred from the unique values in `foundation` — any number of
        categories works (one, three, or more). Each category contributes
        one centroid to the soft-max denominator.
        Defaults to the bundled ``aux/neutral_dictionary.csv``, which
        provides three subspaces: ``neutral.world``, ``neutral.mind``,
        and ``neutral.society``.
    embedding_folder : str or Path, optional
        Directory with `{year}-vocab.pkl` and `{year}-w.npy` files.
        Defaults to `<repo>/aux/embeddings`.
    years : iterable of int, optional
        Decade endpoints used for scoring.
        Defaults to all decades detected in `embedding_folder`.
    year_start, year_end : int, optional
        Range of the yearly interpolation grid (inclusive).
        Default to the first and last detected embedding decade
        respectively, so the output never extrapolates beyond the
        supplied embeddings.
    verbose : bool, default True
        Print progress messages.

    Returns
    -------
    pandas.DataFrame
        The scored (and optionally smoothed) dictionary, with columns
        `foundation`, `word`, and one probability column per decade or
        year (`p{YYYY}`). The same content is written to `output_file`.
    """
    # ---- argument validation ----------------------------------------
    if temp <= 0:
        raise ValueError("`temp` must be a positive float.")
    if smooth_interpolate and not (0 < loess_span <= 1):
        raise ValueError("`loess_span` must lie in (0, 1].")

    embedding_folder = Path(embedding_folder or _DEFAULT_EMBEDDING_FOLDER)
    years = list(years) if years is not None else _detect_embedding_years(embedding_folder)
    if year_start is None:
        year_start = years[0]
    if year_end is None:
        year_end = years[-1]
    kernel_file = Path(kernel_file or _DEFAULT_KERNEL_FILE)
    neutral_file = Path(neutral_file or _DEFAULT_NEUTRAL_FILE)

    # ---- load inputs ------------------------------------------------
    dictionary = _load_two_col_dictionary(dictionary_file)
    if verbose:
        print(f"Loaded dictionary: {len(dictionary)} rows, "
              f"{dictionary['foundation'].nunique()} foundations.")

    kernel = _load_two_col_dictionary(kernel_file)
    foundation_groups = (
        kernel.groupby('foundation')['word'].apply(list).to_dict()
    )

    neutral = _load_two_col_dictionary(neutral_file)
    neutral_groups = (
        neutral.groupby('foundation')['word'].apply(list).to_dict()
    )
    if not neutral_groups:
        raise ValueError(
            f"Neutral reference file '{neutral_file}' contains no usable "
            f"foundation categories. Check that it has 'foundation' and "
            f"'word' columns with at least one row."
        )
    if verbose:
        print(f"Neutral reference: {len(neutral_groups)} category/ies — "
              f"{list(neutral_groups.keys())}.")

    if verbose:
        print(f"Loading embeddings from {embedding_folder}...")
    embeddings_by_year = _load_decadal_embeddings(embedding_folder, years)

    # ---- score per decade -------------------------------------------
    if verbose:
        print(f"Scoring with temp={temp} across "
              f"{len(years)} decades ({min(years)}-{max(years)})...")
    scored = _score_dictionary(
        dictionary, embeddings_by_year,
        foundation_groups, neutral_groups,
        temp=temp, years=years,
    )

    # ---- smooth and interpolate (optional) --------------------------
    if smooth_interpolate:
        if verbose:
            print(f"LOESS smoothing (span={loess_span}) and "
                  f"yearly interpolation ({year_start}-{year_end})...")
        result = _smooth_and_interpolate(
            scored, loess_span, year_start, year_end
        )
    else:
        result = scored

    # ---- write output -----------------------------------------------
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    if verbose:
        print(f"Wrote {len(result)} rows to {out_path}.")
    return result


# ----------------------------------------------------------------------
# Command-line entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute moral-relevance trajectories for a dictionary."
    )
    parser.add_argument("dictionary_file", help="Dictionary CSV or XLSX.")
    parser.add_argument("output_file", help="Destination CSV.")
    parser.add_argument("--temp", type=float, default=0.05)
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable LOESS smoothing and yearly interpolation.")
    parser.add_argument("--loess-span", type=float, default=0.3)
    parser.add_argument("--kernel-file", default=None)
    parser.add_argument("--neutral-file", default=None)
    parser.add_argument("--embedding-folder", default=None)
    parser.add_argument("--year-start", type=int, default=None,
                        help="First year of the interpolation grid "
                             "(default: first available embedding decade).")
    parser.add_argument("--year-end", type=int, default=None,
                        help="Last year of the interpolation grid "
                             "(default: last available embedding decade).")
    args = parser.parse_args()

    compute_moral_relevance(
        dictionary_file=args.dictionary_file,
        output_file=args.output_file,
        temp=args.temp,
        smooth_interpolate=not args.no_smooth,
        loess_span=args.loess_span,
        kernel_file=args.kernel_file,
        neutral_file=args.neutral_file,
        embedding_folder=args.embedding_folder,
        year_start=args.year_start,
        year_end=args.year_end,
    )
