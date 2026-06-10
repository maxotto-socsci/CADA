"""
Document Scoring
================

Applies a scored CADA-style dictionary (one with `p{YYYY}` columns) to
a corpus of documents and returns moral-foundation scores per document.

Pipeline position
-----------------
    kernel.py            ->  expansion.py  ->  relevance.py  ->  scoring.py
    (build kernel)            (expand)          (score)          (this file)

Input
-----
A `pandas.DataFrame` with at least:
    * a `text` column (any capitalisation, string-typed),
    * a `year` column (any capitalisation, numeric).

For every document, the closest year present in the dictionary is used
when the exact year is unavailable.

Output
------
The input frame concatenated with one column per vice/virtue foundation
plus `moral_nonmoral_ratio`. Each foundation column is the sum of the
dictionary's per-word probability for that foundation in the relevant
year, weighted by token counts and normalised according to the chosen
`aggregation` rule:

    aggregation = "per_moral"   (default)
        score[F] = (sum over moral tokens of count * P(F | word)) / n_moral_tokens
        -> mirrors the rule used in eMFDscore
           (https://github.com/medianeuroscience/emfdscore).
    aggregation = "per_total"
        score[F] = (sum over moral tokens of count * P(F | word)) / n_tokens
        -> closer to a classic LIWC-style rate (modified numerator).
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------

import warnings
from collections import Counter
from pathlib import Path
from typing import Tuple

import pandas as pd
import progressbar

import nltk
from nltk.corpus import stopwords as nltk_sw
from sklearn.feature_extraction import text as sk_text
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_sw
from spacy.tokens import Doc

warnings.simplefilter(action='ignore', category=FutureWarning)


# ----------------------------------------------------------------------
# Default foundations (paper's MFT dictionary — kept for reference)
# ----------------------------------------------------------------------

# These are the foundations used in the original paper. They are NOT
# enforced by the scoring functions; foundations are detected from
# whatever dictionary is supplied at runtime.
VICE_VIRTUE_FOUNDATIONS: list[str] = [
    'care.virtue',     'fairness.virtue', 'loyalty.virtue',
    'authority.virtue', 'sanctity.virtue',
    'care.vice',       'fairness.vice',   'loyalty.vice',
    'authority.vice',   'sanctity.vice',
]

VALID_AGGREGATIONS: set[str] = {"per_moral", "per_total"}


# ----------------------------------------------------------------------
# Column validation
# ----------------------------------------------------------------------

def _find_required_column(df: pd.DataFrame, target: str) -> str:
    """Return the actual column name for `target` (case-insensitive)."""
    colmap = {c.lower(): c for c in df.columns}
    if target.lower() not in colmap:
        raise ValueError(
            f"Required column '{target}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )
    return colmap[target.lower()]


def _validate_input_dataframe(df: pd.DataFrame) -> Tuple[str, str]:
    """Ensure the input has text and year columns; return their names."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    text_col = _find_required_column(df, "text")
    year_col = _find_required_column(df, "year")

    if not pd.api.types.is_string_dtype(df[text_col]):
        raise ValueError(f"Column '{text_col}' must contain strings.")
    if not pd.api.types.is_numeric_dtype(df[year_col]):
        raise ValueError(f"Column '{year_col}' must be numeric.")

    return text_col, year_col


# ----------------------------------------------------------------------
# Stop-word set used by the spaCy tokeniser
# ----------------------------------------------------------------------

try:
    _nltk_stopwords = nltk_sw.words('english')
except Exception:
    print("NLTK stopwords missing, downloading now.")
    nltk.download('stopwords')
    _nltk_stopwords = nltk_sw.words('english')

STOPWORDS_SET: set[str] = set(_nltk_stopwords) \
    | set(sk_text.ENGLISH_STOP_WORDS) \
    | set(spacy_sw)


# ----------------------------------------------------------------------
# spaCy extensions / tokeniser
# ----------------------------------------------------------------------

def _ensure_doc_extensions() -> None:
    """Register the custom Doc extensions used to attach token / score info."""
    if not Doc.has_extension("mf_tokens"):
        Doc.set_extension("mf_tokens", default=None)
    if not Doc.has_extension("score_CADA"):
        Doc.set_extension("score_CADA", default={})


_ensure_doc_extensions()


@spacy.language.Language.component("mf_tokenizer")
def mf_tokenizer(doc):
    """Keep lower-cased content tokens; drop stop words, punctuation, etc."""
    filtered_tokens = [
        t for t in doc
        if t.lower_ not in STOPWORDS_SET
        and not t.is_punct
        and not t.is_digit
        and not t.is_quote
        and not t.like_num
        and not t.is_space
    ]
    doc._.mf_tokens = [t.lower_ for t in filtered_tokens]
    return doc


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def _safe_ratio(numerator: int, denominator: int) -> float:
    """Numerator / denominator, with a 0-denominator falling back to 1."""
    return numerator / (denominator if denominator != 0 else 1)


def get_closest_year(year: int, available_years: list[int]) -> int:
    """Return the dictionary year nearest `year`, warning on mismatch."""
    if year in available_years:
        return year
    closest = min(available_years, key=lambda y: abs(y - year))
    warnings.warn(f"Year {year} not in dictionary. Using closest: {closest}.")
    return closest


# ----------------------------------------------------------------------
# Per-document scoring
# ----------------------------------------------------------------------

def score_CADA_doc(
    doc,
    dic_indexed: pd.DataFrame,
    dic_word_set: set[str],
    year: int,
    foundations: list[str],
    aggregation: str = "per_moral",
):
    """Annotate a single spaCy Doc with foundation scores for `year`.

    Parameters
    ----------
    doc : spacy.tokens.Doc
        Document already tagged by the `mf_tokenizer` pipeline component.
    dic_indexed : pandas.DataFrame
        Dictionary indexed on (word, foundation).
    dic_word_set : set of str
        Fast-lookup set of all dictionary words.
    year : int
        Dictionary year column (`p{year}`) to read probabilities from.
    foundations : list of str
        All foundation labels present in the dictionary. One output
        column is produced per entry.
    aggregation : {'per_moral', 'per_total'}, default 'per_moral'
        Denominator used when normalising the foundation sums:
          - 'per_moral' divides by the number of moral tokens
            (matches eMFDscore);
          - 'per_total' divides by the total number of content tokens
            (LIWC-style rate, with the eMFD probability as numerator).
    """
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Choose from {sorted(VALID_AGGREGATIONS)}."
        )

    tokens = doc._.mf_tokens
    year_col = f"p{year}"

    score = {foundation: 0.0 for foundation in foundations}

    if not tokens:
        score['moral_nonmoral_ratio'] = 0
        doc._.score_CADA = score
        return doc

    token_counts = Counter(t for t in tokens if t in dic_word_set)
    if not token_counts:
        score['moral_nonmoral_ratio'] = 0
        doc._.score_CADA = score
        return doc

    total_moral_tokens = sum(token_counts.values())

    # Add weighted contribution from every matched token.
    for word, count in token_counts.items():
        try:
            rows_for_word = dic_indexed.loc[word]
        except KeyError:
            continue
        if isinstance(rows_for_word, pd.Series):
            rows_for_word = rows_for_word.to_frame().T
        for foundation, row in rows_for_word.iterrows():
            if foundation in score:
                score[foundation] += row[year_col] * count

    # Normalise according to the chosen aggregation rule.
    denom = total_moral_tokens if aggregation == "per_moral" else len(tokens)
    for foundation in foundations:
        score[foundation] /= denom

    score['moral_nonmoral_ratio'] = _safe_ratio(
        total_moral_tokens, len(tokens) - total_moral_tokens
    )
    doc._.score_CADA = score
    return doc


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def score_docs(
    data: pd.DataFrame,
    dic_type: str,
    aggregation: str = "per_moral",
) -> pd.DataFrame:
    """Score a DataFrame of documents using one of the bundled dictionaries.

    Parameters
    ----------
    data : pandas.DataFrame
        Must contain a `text` column (string) and a `year` column
        (numeric), in any capitalisation.
    dic_type : str
        Stem (filename minus '.csv') of a dictionary in
        `<repo>/dictionaries/`. The file must contain `foundation`,
        `word`, and one `p{YYYY}` column per year.
    aggregation : {'per_moral', 'per_total'}, default 'per_moral'
        Denominator used when normalising the foundation sums:
          - 'per_moral' divides by the number of moral tokens
            (eMFDscore-style);
          - 'per_total' divides by the total number of content tokens
            (LIWC-style rate).

    Returns
    -------
    pandas.DataFrame
        The original DataFrame concatenated with one column per
        foundation present in the dictionary plus `moral_nonmoral_ratio`.
        The foundation columns are determined entirely by the supplied
        dictionary and require no prior configuration.
    """
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f"Unknown aggregation '{aggregation}'. "
            f"Choose from {sorted(VALID_AGGREGATIONS)}."
        )

    # ---- validate input --------------------------------------------
    text_col, year_col = _validate_input_dataframe(data)

    # ---- load dictionary -------------------------------------------
    dict_folder = Path(__file__).resolve().parent / "dictionaries"
    available = {p.stem: p for p in dict_folder.glob("*.csv")}
    if dic_type not in available:
        raise ValueError(
            f"Dictionary '{dic_type}' not found. "
            f"Available dictionaries: {list(available)}"
        )

    dictionary = pd.read_csv(available[dic_type])
    dictionary["word"] = dictionary["word"].astype(str)

    year_cols = [c for c in dictionary.columns if c.startswith("p")]
    available_years = sorted(int(c[1:]) for c in year_cols)

    # Detect foundations from the dictionary — no hardcoded list required.
    foundations = sorted(dictionary["foundation"].dropna().unique().tolist())

    dic_indexed = dictionary.set_index(["word", "foundation"])
    dic_word_set = set(dictionary["word"])

    # ---- prepare spaCy pipeline ------------------------------------
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
    nlp.add_pipe("mf_tokenizer")

    # ---- progress bar widgets --------------------------------------
    widgets = [
        'Processed: ', progressbar.Counter(), ' ',
        progressbar.Percentage(), ' ',
        progressbar.Bar(marker="⏱"), ' ',
        progressbar.Timer(), ' ',
        progressbar.ETA(),
    ]

    # ---- score documents -------------------------------------------
    scored_docs: list[dict] = []
    with progressbar.ProgressBar(max_value=len(data), widgets=widgets) as bar:
        for i, (doc_text, doc_year) in enumerate(
                zip(data[text_col], data[year_col]), start=1):

            closest_year = get_closest_year(int(doc_year), available_years)
            doc = nlp(doc_text)
            doc = score_CADA_doc(
                doc, dic_indexed, dic_word_set, closest_year,
                foundations=foundations,
                aggregation=aggregation,
            )
            scored_docs.append(doc._.score_CADA)
            bar.update(i)

    scores_df = pd.DataFrame(scored_docs)
    return pd.concat([data.reset_index(drop=True), scores_df], axis=1)
