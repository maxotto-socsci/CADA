"""
LGDE community detection (vendored).

Adapted from the LGDE library by the Barahona research group:
    https://github.com/barahona-research-group/LGDE

The severability primitives in this file are taken in turn from:
    https://github.com/barahona-research-group/severability

This file is vendored inside the CADAscore package so that
`CADAscore.expansion` is self-contained.
"""

import itertools
import functools

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from tqdm import tqdm

############################################################
#### Direct import of Severability
## From https://github.com/barahona-research-group/severability/blob/master/severability/severability.py
def transition_matrix(adj):
    """Computes a Markov transition matrix from an adjacency matrix

    adj (np.matrix): n x n matrix specifying the connection pattern of a graph

    returns a transition matrix: an n x n matrix P where P = diag(A)^-1 * A"""
    if not isinstance(adj, np.matrix):
        raise TypeError("Adjacency matrix expects np.matrix type.")
    diag = np.sum(adj, 1)
    diag[diag == 0] = 1.0  # Replace zeros with 1
    trans_mat = adj / diag  # Now safe
    return trans_mat


def retention(P_C_power):
    """Given P_C^t, compute retention, which is just the sum of all entries
    divided by the number of rows, measuring the chance of a random walker
    to stay in the component given by C"""
    return np.sum(P_C_power) / P_C_power.shape[0]


def mixing(P_C_power):
    """Given P_C^t, computes the mixing, which is how different each row of
    P_C^t is from the average of the rows, after normalization of probabilities
    to 1, an approximation of the quasistationary distribution on the nodes of
    a random walker in C
    """
    diag = np.sum(P_C_power, 1)
    norm_mat = P_C_power / diag

    quasi_dist = np.average(norm_mat, 0)

    mixing = 1 - (
        (0.5 / P_C_power.shape[0]) * np.sum(np.absolute(norm_mat - quasi_dist))
    )
    return mixing


def severability_of_matrix_power(P_C_power):
    """Computes severability, which is 1/2 * (mixing + retention)"""
    diag = np.sum(P_C_power, 1)
    if 0 in diag:
        sev = 0  # Don't let disconnected matrices
    else:
        sev = (mixing(P_C_power) + retention(P_C_power)) / 2
    return sev


def severability_of_component(P, C, t):
    """Computes severability of component C with transition matrix P at Markov
    time t"""
    P_C = P[[[i] for i in C], C]  # Get submatrix
    if t > 1:
        P_C_power = P_C**t
    else:
        P_C_power = P_C
    return severability_of_matrix_power(P_C_power)


def node_component(P, i, t, max_size=65):
    """Optimizes for the best component including a node:

    P (np.matrix):      Markov transition matrix
    i (int):            node to start from
    t (int):            Markov time
    max_size (int):     stop the search at max_size

    returns (component, severability)
    """
    if not isinstance(P, np.matrix):
        raise TypeError("Transition matrix expects np.matrix type.")
    linked_to = np.asarray(P[i, :]).nonzero()[1].tolist()
    neighbors = [item for item in linked_to if item not in [i]]
    # Order by highest severability directions to add nodes
    n_sorted = sorted(
        neighbors, key=lambda n: -1 * severability_of_component(P, [i, n], t)
    )
    for n in n_sorted:
        component, sev = component_optimise(P, [i, n], t, max_size)
        if i in component:
            break
    if i not in component:
        return ([], 0)
    else:
        return (component, sev)

def connected_component(P, C):
    """Finds the max connected component for component C
    Currently uses BFS. Possibly want to reimplement with DFS?"""
    component = C
    linked_to = np.asarray(np.sum(P[component, :], 0)).nonzero()[1].tolist()
    new_component = list(set(C + linked_to))
    if len(new_component) == len(component):
        return new_component
    else:
        return connected_component(P, new_component)

def component_optimise(P, C, t, max_size=65):
    """Optimises for the best component starting from a community:

    P (np.matrix):      Markov transition matrix
    C list(int):        component to start from
    t (int):            Markov time
    max_size (int):     stop the search at max_size

    returns (component, severability)
    """
    i = 1
    sev_max = 0
    C_max = C
    max_size = min(max_size, len(connected_component(P, C)))
    # Get up to max_size for the community using a 2-to-1 mix of greedy adds
    # and Kernighan-Lin steps
    while (len(C) < max_size) and (i < 3 * max_size):
        if (i % 3) == 0:
            (C, sev) = kernighan_lin_step(P, C, t)
        else:
            (C, sev) = greedy_add_step(P, C, t)
        if sev > sev_max:
            sev_max = sev
            C_max = C
        i = i + 1
    # Find local maximum for severability using optional Kernighan-Lin steps
    sev_improve = True
    C = C_max
    sev = sev_max
    while sev_improve:
        if len(C) == max_size:
            (C, sev) = greedy_remove_step(P, C, t)
        else:
            (C, sev) = kernighan_lin_step(P, C, t)
        if sev > sev_max:
            sev_max = sev
            C_max = C
        else:
            sev_improve = False
    return (C_max, sev_max)

def greedy_add_step(P, C, t):
    """Greedily adds a node to C such that the new severability is as high
    as possible. Note that a node *will* be added, even if any added node
    decreases the severability, unless there are no neighbors"""
    linked_to = np.asarray(np.sum(P[C, :], 0)).nonzero()[1].tolist()
    neighbors = [item for item in linked_to if item not in C]
    if len(neighbors) > 0:
        new_node = max(
            neighbors, key=lambda n: severability_of_component(P, C + [n], t)
        )
        C = C + [new_node]
    return (C, severability_of_component(P, C, t))

def greedy_remove_step(P, C, t):
    """Greedily removes a node to C such that the new severability is as high
    as possible. Note that a node *will* be removed, even if any removal
    decreases the severability"""
    removed_node = max(
        C, key=lambda n: severability_of_component(P, [x for x in C if x != n], t)
    )
    C = [x for x in C if x != removed_node]
    return (C, severability_of_component(P, C, t))

def kernighan_lin_step(P, C, t):
    """if can add a node, will either remove or add a node, optimising sev
    if can't add a node because no neighbors, will either remove a node or stay constant.
    """
    C_add, sev_add = greedy_add_step(P, C, t)
    C_rem, sev_rem = greedy_remove_step(P, C, t)
    if sev_add > sev_rem:
        return (C_add, sev_add)
    else:
        return (C_rem, sev_rem)
############################################################
############################################################


class BaseExpansion:
    """Base class for dictionary expansion.

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.
    word_list : list[str] of length n_words
        List of all words considered.
    word_vecs : array like of shape (n_words, r)
        Word vectors with dimension r for each word in word_list.
    Attributes
    ----------
    semantic_sim : sparse csr_matrix of shape (n_words, n_words)
        Normalised cosine similarity between pairs of words (sparse).
    expanded_dict_ : list[str]
        List of words in expanded dictionary.
    discovered_dict_ : list[str]
        List of discovered keywords.
    """
    def __init__(self, seed_dict, word_list, word_vecs):

        # store list of words
        self.word_list = word_list

        # store word vectors as numpy array for efficient access
        self.word_vecs = np.array(word_vecs, dtype=np.float32)

        # compute dictionary that maps words to indices
        self.word_to_ind = {self.word_list[i]: i for i in range(self.n_words)}

        # compute effective seed dictionary
        self.seed_dict = []
        for keyword in seed_dict:
            if keyword in set(self.word_list):
                self.seed_dict.append(keyword)

        # compute normalised cosine similarity (sparse)
        self.semantic_sim = None  # Will be computed lazily

        # attributes
        self.expanded_dict_ = self.seed_dict
        self.discovered_dict_ = []

    @property
    def n_words(self):
        """Computes total number of words."""
        return len(self.word_list)

    @property
    def n_seed(self):
        """Computes number of seed keywords."""
        return len(self.seed_dict)

    @property
    def n_discovered(self):
        """ "Computes number of discovered words."""
        return len(self.discovered_dict_)

    def _compute_semantic_sim_sparse(self, k=10, n_jobs=1):
        """Computes sparse normalised cosine similarity using k-NN.

        Parameters
        ----------
        k : int
            Number of nearest neighbors to keep for each word.
        n_jobs : int
            Number of parallel jobs for k-NN search. Use -1 for all cores.

        Returns
        -------
        sparse csr_matrix
            Sparse normalized cosine similarity matrix.
        """
        from sklearn.neighbors import NearestNeighbors

        print(f"Computing {k}-NN for {self.n_words} words...")

        # Normalize word vectors for cosine similarity
        norms = np.linalg.norm(self.word_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vecs = self.word_vecs / norms

        # Find k nearest neighbors using cosine similarity
        # metric='cosine' in sklearn uses 1 - cosine_similarity
        nbrs = NearestNeighbors(
            n_neighbors=k + 1,  # +1 because each word is its own neighbor
            algorithm='brute',  # More accurate for cosine
            metric='cosine',
            n_jobs=n_jobs
        ).fit(normalized_vecs)

        distances, indices = nbrs.kneighbors(normalized_vecs)

        # Build sparse similarity matrix
        # Convert cosine distance to similarity: sim = 1 - dist
        similarities = 1 - distances

        # Create sparse matrix in LIL format for efficient construction
        sim_matrix = lil_matrix((self.n_words, self.n_words), dtype=np.float32)

        print("Building sparse similarity matrix...")
        for i in tqdm(range(self.n_words), desc="Building matrix"):
            for j, sim in zip(indices[i], similarities[i]):
                if i != j:  # Skip self-loops for now
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim  # Symmetric

        # Convert to CSR for efficient operations
        sim_matrix = sim_matrix.tocsr()

        # Normalize by maximum similarity (as in original code)
        max_sim = sim_matrix.max()
        if max_sim > 0:
            sim_matrix = sim_matrix / max_sim

        return sim_matrix

    def pairwise_word_similarity(self, kw1, kw2):
        """Compute pairwise similarity between two words."""
        if self.semantic_sim is None:
            raise ValueError("Semantic similarity not computed yet. Run construct_network first.")
        return self.semantic_sim[self.word_to_ind[kw1], self.word_to_ind[kw2]]


class LGDE(BaseExpansion):
    """Class for Local Graph-based Dictionary Expansion - Modular Community Detection.

    This method constructs a weighted undirected semantic similarity graph from the
    normalised cosine similarity matrix using CkNN and computes semantic communities
    for each seed keyword using local community detection.

    Supports multiple community detection methods:
    - "severability": Original severability-based method (slow but thorough)
    - "siwo": Fast SIWO/LSWL method from CDlib (recommended for large graphs)

    Parameters
    ----------
    seed_dict : list[str]
        List of words in seed dictionary.
    word_list : list[str] of length n_words
        List of all words considered.
    word_vecs : array like of shape (n_words, r)
        Word vectors with dimension r for each word in word_list.

    Attributes
    ----------
    semantic_sim : sparse csr_matrix of shape (n_words, n_words)
        Normalised cosine similarity between pairs of words.
    expanded_dict_ : list[str]
        List of words in expanded dictionary.
    discovered_dict_ : list[str]
        List of discovered keywords.
    adjacency_ : sparse csr_matrix of shape (n_words, n_words)
        Adjacency matrix of undirected weighted semantic similarity graph.
    semantic_communities_ : dict
        Dictionary that maps seed keywords to their semantic communities.
    """

    def __init__(self, seed_dict, word_list, word_vecs):
        # initialise base class
        super().__init__(seed_dict, word_list, word_vecs)

        # attributes
        self.adjacency_ = None
        self.semantic_communities_ = {}

    def construct_network(self, k=4, n_jobs=1):
        """Construct semantic similarity network using CkNN.

        Parameters:
        -----------
        k : int
            Number of nearest neighbours considered for CkNN construction, should be
            at least 1.
        n_jobs : int
            Number of parallel jobs for k-NN search. Use -1 for all cores.
        """

        k_sim = k * 3  # Number of nearest neighbors for initial similarity computation.

        # Compute sparse semantic similarity if not already done
        if self.semantic_sim is None:
            self.semantic_sim = self._compute_semantic_sim_sparse(k=k_sim, n_jobs=n_jobs)

        print(f"Computing CkNN with k={k}...")

        # Compute CkNN graph on sparse similarity
        backbone = _compute_cknn_sparse(self.semantic_sim, k=k, delta=1.0)

        # Add self-loops to singletons
        row_sums = np.array(backbone.sum(axis=1)).flatten()
        singletons = np.where(row_sums == 0)[0]
        if len(singletons) > 0:
            print(f"Adding self-loops to {len(singletons)} singletons")
            backbone = backbone.tolil()
            for singleton in singletons:
                backbone[singleton, singleton] = 1
            backbone = backbone.tocsr()

        # Obtain similarity network (element-wise multiplication)
        self.adjacency_ = backbone.multiply(self.semantic_sim).tocsr()

        print(
            f"Network constructed: {self.adjacency_.nnz} edges ({self.adjacency_.nnz / (self.n_words ** 2) * 100:.3f}% density)")

    def detect_local_communities_severability(self, t=1, disable_tqdm=False, n_jobs=1):
        """Detect local semantic communities using Severability (threaded).

        Parameters:
        -----------
        t : int
            Markov scale t of Severability, should be at least 1.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.

        n_jobs : int
            Number of threads to use. If <= 0 uses max(1, os.cpu_count()) threads.
            If 1, runs serially (no threading).
        """
        if self.adjacency_ is None:
            raise ValueError("Network not constructed. Run construct_network first.")

        # Single dense copy (float32)
        print("Converting adjacency to dense (single copy) for severability...")
        adjacency_dense = self.adjacency_.astype(np.float32).toarray()

        # Compute transition matrix; severability expects a numpy.matrix
        P = transition_matrix(np.matrix(adjacency_dense))

        seed_list = list(self.seed_dict)  # stable order
        n_seeds = len(seed_list)
        print(f"Detecting communities for {n_seeds} seed words using Severability...")

        # If user requested serial execution or only one thread, do simple loop.
        if n_jobs is None or n_jobs <= 1:
            for keyword in tqdm(seed_list, disable=disable_tqdm):
                self.semantic_communities_[keyword] = self._compute_local_community_severability(
                    keyword, P, t, adjacency_dense
                )
        else:
            # Determine number of threads
            import os
            max_workers = n_jobs if n_jobs > 0 else max(1, os.cpu_count() or 1)
            # limit threads to number of seeds to avoid unnecessary threads
            max_workers = min(max_workers, n_seeds)

            lock = threading.Lock()

            def worker(keyword):
                """Worker function run in threads: returns (keyword, community_list)."""
                community = self._compute_local_community_severability(keyword, P, t, adjacency_dense)
                return keyword, community

            # Submit jobs
            futures = []
            with ThreadPoolExecutor(max_workers=max_workers) as exe:
                for keyword in seed_list:
                    futures.append(exe.submit(worker, keyword))

                # Use tqdm to monitor progress over as_completed
                for fut in tqdm(as_completed(futures), total=len(futures), disable=disable_tqdm):
                    try:
                        keyword, community = fut.result()
                    except Exception as e:
                        print(f"Warning: community detection failed for a seed: {e}")
                        continue

                    # thread-safe write
                    with lock:
                        self.semantic_communities_[keyword] = community

        self._compile_discovered_dict()

    def detect_local_communities_siwo(self, timeout=60, disable_tqdm=False):
        """Detect local semantic communities using SIWO/LSWL from CDlib.

        SIWO (Strength of Links-based Weighted Local) is a fast, parameter-free
        local community detection method specifically designed for weighted networks.

        Parameters:
        -----------
        timeout : float
            The maximum time in seconds for SIWO to retrieve each community.

        disable_tqdm : bool
            Disable progress bar for semantic community detection if True.

        References:
        -----------
        M. Zafarmand et al., 'Fast local community discovery relying on
        the strength of links', Social Network Analysis and Mining, vol. 13,
        no 1., p. 112, Sep. 2023, doi:10.1007/s13278-023-01115-7.
        """
        if self.adjacency_ is None:
            raise ValueError("Network not constructed. Run construct_network first.")

        try:
            import networkx as nx
            from cdlib import algorithms as cdlib_alg
        except ImportError:
            raise ImportError(
                "CDlib and NetworkX are required for SIWO community detection. "
                "Install with: pip install cdlib networkx"
            )

        print(f"Detecting communities for {len(self.seed_dict)} seed words using SIWO/LSWL...")

        # Convert sparse adjacency to NetworkX graph (preserves weights)
        nx_graph = nx.from_scipy_sparse_array(self.adjacency_)

        for keyword in tqdm(self.seed_dict, disable=disable_tqdm):
            keyword_ind = self.word_to_ind[keyword]

            try:
                # Run SIWO/LSWL with timeout
                result = cdlib_alg.lswl(nx_graph, keyword_ind, timeout=timeout)
                community_ind = result.communities[0] if result.communities else []

                # Convert indices to words
                self.semantic_communities_[keyword] = [
                    self.word_list[i] for i in community_ind
                ]

            except (IndexError, Exception) as e:
                # If SIWO fails, fall back to just the seed word
                print(f"SIWO failed for '{keyword}': {e}. Using seed only.")
                self.semantic_communities_[keyword] = [keyword]

        self._compile_discovered_dict()

    def _compile_discovered_dict(self):
        """Compile discovered dictionary from semantic communities."""
        # compile discovered dictionary
        all_communities = set(
            itertools.chain.from_iterable(list(self.semantic_communities_.values()))
        )
        self.discovered_dict_ = list(all_communities - set(self.seed_dict))

        # summarise expanded dictionary
        self.expanded_dict_ = self.seed_dict + self.discovered_dict_

        print(f"Discovered {len(self.discovered_dict_)} new words")

    def expand(self, k=4, method="siwo", t=1, timeout=60, n_jobs=1, disable_tqdm=False):
        """Expand seed dictionary with LGDE method.

        Parameters:
        -----------
        k : int
            Number of nearest neighbours considered for CkNN construction.

        method : str
            Community detection method to use:
            - "siwo": Fast SIWO/LSWL method (recommended, requires cdlib)
            - "severability": Original severability-based method (slow)

        t : int
            Markov time parameter for severability method (ignored for SIWO).

        timeout : float
            Timeout in seconds for SIWO method (ignored for severability).

        n_jobs : int
            Number of parallel jobs:
            - For network construction: passed to sklearn's k-NN search
            - For severability: number of threads for parallel community detection
            - Ignored for SIWO (uses single-threaded CDlib implementation)

        disable_tqdm : bool
            Disable progress bars if True.
        """
        # Construct network
        self.construct_network(k, n_jobs=n_jobs)

        # Detect communities using specified method
        if method.lower() == "siwo":
            self.detect_local_communities_siwo(timeout=timeout, disable_tqdm=disable_tqdm)
        elif method.lower() == "severability":
            self.detect_local_communities_severability(t=t, disable_tqdm=disable_tqdm, n_jobs=n_jobs)
        else:
            raise ValueError(
                f"Unknown method '{method}'. Use 'siwo' or 'severability'."
            )

    def _compute_local_community_severability(self, keyword, P, t, adjacency_dense):
        """Compute local community using severability method."""
        keyword_ind = self.word_to_ind[keyword]
        # check if keyword is singleton in graph
        if (
                np.sum(adjacency_dense[keyword_ind])
                == adjacency_dense[keyword_ind, keyword_ind]
        ):
            return [keyword]
        # otherwise compute local community with Severability
        else:
            community_ind = node_component(P=P, i=keyword_ind, t=t)[0]
            # if keyword was found as Severability orphan, return keyword
            if len(community_ind) == 0:
                return [keyword]
            # otherwise return local community
            else:
                return [self.word_list[i] for i in community_ind]


def _compute_cknn_sparse(sim_matrix, k=5, delta=1.0):
    """Compute CkNN graph on sparse similarity matrix.

    Parameters
    ----------
    sim_matrix : sparse csr_matrix
        Sparse similarity matrix (higher values = more similar)
    k : int
        Number of nearest neighbors for CkNN
    delta : float
        Threshold parameter for CkNN

    Returns
    -------
    sparse csr_matrix
        Sparse binary adjacency matrix of CkNN graph
    """
    if not isinstance(sim_matrix, csr_matrix):
        sim_csr = sim_matrix.tocsr()
    else:
        sim_csr = sim_matrix

    n = sim_csr.shape[0]

    # Compute k-th nearest neighbor *distance* per row.
    # For nonzero similarity entries sim_ij we define dist_ij = 1 - sim_ij.
    kth_distances = np.ones(n, dtype=np.float32)  # default 1.0 for singletons

    print("Computing k-th nearest neighbors (distances)...")
    # iterate rows using CSR index pointers (fast, memory efficient)
    for i in tqdm(range(n), desc="Computing k-NN distances"):
        row_start = sim_csr.indptr[i]
        row_end = sim_csr.indptr[i + 1]
        if row_end == row_start:
            # no neighbors: singleton -> distance 1.0
            kth_distances[i] = 1.0
            continue

        sim_vals = sim_csr.data[row_start:row_end]  # non-zero similarities in row i
        # convert to distances
        dists = 1.0 - sim_vals
        # Exclude potential self-loop if present (sim on diagonal); handle safely:
        # if there's an index equal to i, drop its corresponding dist
        row_indices = sim_csr.indices[row_start:row_end]
        mask = row_indices != i
        if mask.sum() == 0:
            # only self-loop
            kth_distances[i] = 1.0
            continue
        dists = dists[mask]

        if dists.size >= k:
            kth_distances[i] = np.partition(dists, k - 1)[k - 1]
        else:
            kth_distances[i] = dists.max()

    # Build CkNN graph (binary symmetric adjacency)
    print("Building CkNN graph...")
    backbone = lil_matrix((n, n), dtype=np.int8)

    # We'll reuse sim_csr for iterating over existing similarity edges;
    # convert sim data to distances on-the-fly rather than constructing a dense distance matrix.
    print("Applying CkNN thresholding...")
    for i in tqdm(range(n), desc="CkNN thresholding"):
        row_start = sim_csr.indptr[i]
        row_end = sim_csr.indptr[i + 1]
        for idx in range(row_start, row_end):
            j = sim_csr.indices[idx]
            if i == j:
                continue
            sim_ij = sim_csr.data[idx]
            d_ij = 1.0 - sim_ij
            # CkNN condition: d_ij / sqrt(d_ik * d_jk) < delta
            denom = np.sqrt(kth_distances[i] * kth_distances[j])
            # protect against denom == 0
            if denom == 0:
                ratio = np.inf
            else:
                ratio = d_ij / denom

            if ratio < delta:
                backbone[i, j] = 1
                backbone[j, i] = 1  # ensure symmetry

    return backbone.tocsr()