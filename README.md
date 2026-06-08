# Context Adaptive Dictionary Analysis (CADA)

This repository contains the scripts used to build and apply the
**CADA** dictionaries for the paper *"Context Adaptive Dictionary Analysis"* (TODO: eventual citation).

To run the pre-specified pipeline, you can clone the repository,
install the dependencies, point the scripts at your own data, and
either run them from the command line or copy the functions into your
own code.


## General structure

The full pipeline has four stages, each implemented in a single Python file:

```
kernel.py        ->  expansion.py  ->  relevance.py  ->  scoring.py
build a kernel       expand dict.       score the         apply to
of diachronically    on diachronic      expanded          documents
stable seed words    embedding splits   dictionary        in a corpus
                                        per decade /
                                        year
```

Two further files support this pipeline:

- `lgde_cluster.py` — the LGDE backend, vendored from
  https://github.com/barahona-research-group/LGDE (severability
  primitives in turn from
  https://github.com/barahona-research-group/severability). No
  external installation of LGDE is required.
- `example_pipeline.ipynb` — a Jupyter notebook wiring the four steps together on a tiny
  example in order to illustrate the order of operations.

The `aux/` folder holds bundled defaults used in the original paper: a stop-word reference for
the kernel filter, the kernel dictionary built for *Moral Foundations* centroids, and a
neutral reference dictionary. The `dictionaries/` folder holds pre-computed CADA MFT dictionaries that
can be applied to diachronic corpora via `scoring.py`.

## Setup

```bash
git clone <repo-url> CADAscore
cd CADAscore
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # used by scoring.py
```

The pipeline runs natively on word2vec embeddings
(`{year}-vocab.pkl` and `{year}-w.npy` for each period/year) 
which are **not** distributed alongside the repository (see 
[Hamilton et al. 2018](https://doi.org/10.48550/arXiv.1605.09096) )

They can be obtained via https://nlp.stanford.edu/projects/histwords/ and 
placed in `aux/embeddings/` or 
passed via an alternative path to the relevant function via `embedding_folder=...`.

To allow for a minimal reproducible example, `aux/sample_embeddings` holds a unsystematic 
sample of Google Books derived embeddings from three decades. Please note that these are for illustrative 
purposes only and do not allow for genuine analyses. 

## Using the scripts

Each script can be used in three ways:

1. **As a function inside your own Python code.** Sit in the
   repository directory (or add it to `sys.path`) and import:
   ```python
   from kernel import get_kernel_dictionary
   from expansion import expand_kernel_lgde
   from relevance import compute_moral_relevance
   from scoring import score_docs
   ```

2. **From the command line.** Every script exposes a small
   `argparse` interface, so:
   ```bash
   python kernel.py seeds.xlsx kernel.csv --freq-cutoff 1e-6
   python expansion.py kernel.csv expanded.csv --method severability --k 15 --t 5
   python relevance.py expanded.csv scored.csv --temp 0.05 --loess-span 0.3
   ```

3. **By copying the code into your project** and adapting it.

The sections below describe each stage in detail.

## 1. Kernel Dictionary Construction (`kernel.py`)

### Purpose

The script takes an expert-curated seed dictionary and reduces it to
a *kernel* of words that are (i) sufficiently frequent in every
historical period considered and (ii) semantically stable relative to
a null distribution of construct-neutral reference terms. The kernel
is intended to be representative of the underlying construct while
remaining usable across the full historical window.

### Method

Two filters are applied:

1. **Frequency filter.** Each candidate's minimum Google Ngram
   frequency across all years must reach `freq_cutoff` (default
   `1e-6`). Pass `fetch_ngrams=False` (or `--no-ngrams` on the CLI)
   to skip this filter entirely.
2. **Semantic-change filter.** Two cosine-distance statistics are
   computed per candidate (`total_change` and `avg_decade_change`) on
   the supplied embeddings and benchmarked against the same statistics
   on a reference set of neutral words (by default, stop words).
   A candidate is retained only if both fall below the configurable
   percentile (`change_percentile`, default `0.95`) of this null
   distribution.

The script also allows passing words ending in `*`, which are treated as stems and expanded
via WordNet (and the `english_words` package when available) prior to scoring.

### Reference distribution

The null distribution is computed from the same embeddings
used for seed scoring through a user supplied .csv file with a `word` column (these should be
words assumed to be semantically stable). If none is supplied, the script defaults to `aux/stopwords.csv` 
which is a list of stopwords. For each reference word, the script computes:

- `total_change` — cosine distance between the first and last supplied year.
- `avg_decade_change` — mean cosine change across consecutive year steps.

The thresholds are always calibrated to whatever year range
supplied. For example, with embeddings for 1800, 1850, and 1950,
`total_change` spans 1800→1950 and `avg_decade_change` averages
1800→1850 and 1850→1950.

### Example

```python
from kernel import get_kernel_dictionary

# Default: stopwords.csv is scored against all embeddings in aux/embeddings/.
get_kernel_dictionary(
    input_file="seeds/curated_seed_dictionary.xlsx",
    output_file="kernel_dictionary.csv",
    freq_cutoff=1e-6,
    change_percentile=0.95,
)

# Custom year range: thresholds are calibrated to 1800, 1850, 1950 only.
get_kernel_dictionary(
    input_file="seeds/curated_seed_dictionary.xlsx",
    output_file="kernel_dictionary.csv",
    years=[1800, 1850, 1950],
    fetch_ngrams=False,   # skip frequency filter
)

# Custom reference word list.
get_kernel_dictionary(
    input_file="seeds/curated_seed_dictionary.xlsx",
    output_file="kernel_dictionary.csv",
    reference_file="my_reference_words.csv",  # needs a 'word' column
)
```

### Auxiliary data

- `aux/stopwords.csv` — plain stop-word list (default reference). 
- `aux/embeddings/{year}-vocab.pkl` and `aux/embeddings/{year}-w.npy` (embeddings).

## 2. Kernel Expansion (`expansion.py`)

### Purpose

Building on the diachronically stable kernel dictionary, this step 
recovers construct-coverage across historical periods. Each kernel term is used 
as a seed for a local expansion in the embedding space of each decade. The
per-decade expansions are then integrated into a single, deduplicated
dictionary that preserves the construct labels of the kernel.

### Method

For each decade in `years`, `expand_kernel_lgde` loads the embeddings
and, for every foundation, grows a local neighborhood around each
kernel seed using one of three methods:

- **`severability`** (graph-based; default). Builds a continuous
  k-nearest-neighbour graph and detects the local community via
  severability. Parameters: `k` (neighbours) and `t` (Markov time). See [Yu et al. (2020)](https://doi.org/10.48550/arXiv.2006.02972).
- **`siwo`** (graph-based). Same CkNN graph, SIWO community detector.
  Parameter: `k`. See [Zafarmand et al. (2023)](https://doi.org/10.1007/s13278-023-01115-7)
- **`cosine`** (graph-free). For each seed, includes every vocabulary
  word whose cosine similarity to the seed in that decade's embedding
  space is at least `cosine_threshold`. No graph is built.

Each decade writes its expansion to a per-decade CSV in a working
directory. Once all decades are done, the per-decade CSVs are
concatenated, stop words / words containing digits / words shorter
than `min_word_length` characters are dropped (this can guard against OCR noise)
and the result is deduplicated on (foundation, word). Note, that depending on the use case, 
these filters may want to be adapted. The integrated dictionary is
written to `output_file`; the per-decade intermediates are deleted
unless `keep_intermediates=True`.

Note, that using the `method="severability"` can be computationally expensive 
depending on the size of the kernel dictionary and the type of word embeddings. 
`method="cosine"` is fastest and, in many cases, may deliver acceptable results.

### Example

```python
from expansion import expand_kernel_lgde

# Graph-based (default): severability with k=15 and Markov time t=5.
expand_kernel_lgde(
    kernel_file="kernel.csv",
    output_file="expanded.csv",
    method="severability",
    k=15,
    t=5,
)

# Graph-free: keep everything within cosine similarity 0.3 of any seed.
expand_kernel_lgde(
    kernel_file="kernel.csv",
    output_file="expanded_cos03.csv",
    method="cosine",
    cosine_threshold=0.3,
)
```

The same options are available from the command line:

```bash
python expansion.py kernel.csv expanded_cos03.csv \
    --method cosine --cosine-threshold 0.3
```

### LGDE dependency and attribution

The graph-based methods rely on a community-detection backend vendored
as `lgde_cluster.py`, adapted from
[https://github.com/barahona-research-group/LGDE](https://github.com/barahona-research-group/LGDE)
(with severability primitives adapted in turn from
[https://github.com/barahona-research-group/severability](https://github.com/barahona-research-group/severability)).
No external installation of LGDE is required. The `cosine` method does
not use the LGDE backend.

## 3. Moral-Relevance Trajectories (`relevance.py`)

### Purpose

This calculates moral-relevance trajectories for words in a given
(expanded) dictionary: for every (foundation, word) pair, it produces
a per-embedding-period — and, optionally, per-year — series quantifying the
proximity of that word's diachronic embedding to its foundation
relative to a neutral baseline. The output is the input format
expected by `scoring.py`, so the function bridges the dictionary-
construction stage with the document-scoring stage.

### Method

For each decade and each word, the script computes the centroid of the
foundation kernel for that word's foundation, plus one centroid per
category in the neutral reference dictionary.  Proximity is the
temperature-scaled exponential of the Euclidean distance,

    proximity(w, c; T) = exp(- || w - c ||_2 / T),

and the moral-relevance score is the soft-max share of the moral
proximity in the sum of moral and all neutral/reference proximities.

The number of neutral/reference categories is inferred automatically from the
unique values in the `foundation` column of the neutral reference file,
so a file with one category, three, or more all work without any code
changes.  The default file (`aux/neutral_dictionary.csv`) provides three
subspaces: `neutral.world`, `neutral.mind`, and `neutral.society`.

When `smooth_interpolate=True` (default), each (word, foundation) trajectory
is LOESS-smoothed at bandwidth `loess_span` (default 0.3) and linearly
interpolated to a yearly grid covering `year_start..year_end`.

### Example

```python
from relevance import compute_moral_relevance

compute_moral_relevance(
    dictionary_file="expanded.csv",
    output_file="scored_dictionary.csv",
    temp=0.05,
    smooth_interpolate=True,
    loess_span=0.3,
)
```

### Required auxiliary data

`aux/CADA_KERNEL.csv` and `aux/neutral_dictionary.csv` are bundled, so
the function works out of the box. Per-decade embeddings must be
provided separately (see Setup).

To use a custom neutral reference, supply any CSV with `foundation` and
`word` columns via `neutral_file`. The script infers the number of
neutral categories from the unique values in `foundation` — you are not
limited to three. Please note that when building a custom reference dictionary, 
it is advisable to run the `get_kernel_dictionary()` function on this reference 
prior to using it in this step to ensure semantic stability of the included 
reference centroids.

## 4. Document Scoring (`scoring.py`)

`score_docs(df, dic_type, aggregation="per_moral")` applies a scored
dictionary from `dictionaries/` to a corpus. The input DataFrame must
contain a `text` column (string) and a `year` column (numeric). The
function returns the same DataFrame with one column per 
foundation plus `moral_nonmoral_ratio`. When the dictionary doesn't
contain a column for the document's exact year, the closest available
year is used (with a warning).

### Aggregation rule

For each document, the foundation score is the count-weighted sum of
the dictionary's per-word probabilities, normalised by a denominator
chosen via the `aggregation` argument:

- **`per_moral`** (default) divides by the number of moral tokens
  (tokens that appear in the dictionary). This is the rule used in
  [eMFDscore](https://github.com/medianeuroscience/emfdscore), a
  similar probabilistic-lexicon scorer, and it expresses each
  foundation as its share of the document's moral signal.
- **`per_total`** divides by the total number of content tokens. This
  is closer to a classic LIWC-style rate — the proportion of the
  document devoted to the foundation — albeit with the probabilistic
  numerator that CADA uses in place of binary keyword hits.

The two rules answer different questions: `per_moral` asks "given that
this document is talking about morality, which foundation does it
emphasize?" while `per_total` asks "how morally loaded is this
document overall, and along which foundation?". Choose the rule that
matches your construct.

### Example

```python
import pandas as pd
from scoring import score_docs

docs = pd.read_csv("my_corpus.csv")        # needs 'text' and 'year' columns

# Default: eMFDscore-style normalisation.
scored = score_docs(docs, dic_type="scored_dictionary")

# LIWC-style rate.
scored_rate = score_docs(
    docs,
    dic_type="scored_dictionary",
    aggregation="per_total",
)
```

## Repository layout

```
CADAscore/
├── README.md
├── requirements.txt
├── example_pipeline.ipynb     # end-to-end demo
├── kernel.py
├── expansion.py
├── relevance.py
├── scoring.py
├── lgde_cluster.py         # vendored LGDE backend
├── aux/
│   ├── stopwords.csv            # plain stop-word list (default reference for kernel.py)
│   ├── CADA_KERNEL.csv          # default moral kernel for relevance.py
│   ├── neutral_dictionary.csv   # default neutral references for relevance.py
│   ├── example_embeddings/      # sampled GoogleBooks derived embeddings for illustration
│       └──
└── dictionaries/                # scored CSVs consumed by scoring.py
    └── ...
```

## License

GPL v3 (see `LICENSE`). Compatible with the vendored LGDE code, which
is also GPL-licensed.

## Citation

If you use these scripts, please cite the paper and link back to this
repository.
