import os
import sys
import numpy as np
import scipy.sparse as sparse
import time
from typing import Union

import graphblas as gb
from graphblas import Matrix, Vector, dtypes

import gc

default_threads = max(1, os.cpu_count() // 2)

def check_random_state(seed: Union[int, None]):
    # Generate a global random number generator (global RNG) - a RandomState instance
    if seed is None or seed is np.random:
        # Return np.random.RandomState() (variable _rand = np.random.RandomState(), the seed is chosen automatically by system)
        return np.random.mtrand._rand
    if isinstance(seed, int):
        # Returns a RandomState object seeded with given integer
        return np.random.RandomState(seed)

def squared_row_norms(X: gb.Matrix, n_threads: int=default_threads):
    sq_X = Matrix(dtypes.FP64, X.shape[0], X.shape[1])
    sq_X(nthreads=n_threads) << X.ewise_mult(X, op="times")

    sq_row_norms = Vector(dtypes.FP64, X.shape[0])
    sq_row_norms(nthreads=n_threads) << sq_X.reduce_rowwise("plus")

    return sq_row_norms.to_dense(fill_value=0)


def check_centroids_density(centroids: gb.Matrix):
    gamma = 0.1

    if centroids.ss.format == "fullr":
        is_centroid_dense = True
        centroids_density = centroids.V.new().nvals / (centroids.nrows * centroids.ncols)
        if centroids_density <= gamma:
            is_centroid_dense = False
            centroids(mask=centroids.V, replace=True) << centroids
    else:
        is_centroid_dense = False
        centroids_density = centroids.nvals / (centroids.nrows * centroids.ncols)
        if centroids_density > gamma:
            is_centroid_dense = False
            centroids(mask=~centroids.S) << 0

    return centroids, is_centroid_dense


def predict_labels(X: gb.Matrix, centroids: gb.Matrix, is_centroid_dense: bool, n_threads: int=default_threads):
    n_samples = X.shape[0]
    n_clusters = centroids.shape[0]

    # x_squared_norms is not needed
    c_squared_norms = squared_row_norms(centroids, n_threads)
    XCt = Matrix(dtypes.FP64, nrows=n_samples, ncols=n_clusters)

    # For X * Ct, we use CSR x (fullc or CSC) for the best efficiency.
    # We use row format to store centroids in other places, so must convert them here to column format.
    if is_centroid_dense:
        centroids = centroids.ss.export("fullc")
        centroids = gb.Matrix.ss.import_fullc(**centroids)

        XCt << 0
        XCt(accum=gb.binary.plus, nthreads=n_threads) << X.mxm(centroids.T)
        XCt = 2 * XCt.to_dense()

    else:
        centroids = centroids.ss.export("csc")
        centroids = gb.Matrix.ss.import_csc(**centroids)

        XCt(nthreads=n_threads) << X.mxm(centroids.T)
        XCt = 2 * XCt.to_dense(fill_value=0)

    # x_squared_norms is not needed
    c_squared_norms = squared_row_norms(centroids, n_threads)
    distances_to_centroids = -XCt + c_squared_norms[np.newaxis, :]

    labels = np.argmin(distances_to_centroids, axis=1)

    return labels, distances_to_centroids


def kmeans_predict(X: Union[sparse.csr_matrix, gb.Matrix], centroids: Union[sparse.csr_matrix, gb.Matrix], n_threads: int=default_threads) -> np.array:
    """Predict cluster for each sample in X given centroids

    Parameters
    ----------
    X : `gb.Matrix (csr format), scipy.sparse.csr_matrix` 
        Input dataset for clustering
    centroids : `gb.Matrix (csr format), scipy.sparse.csr_matrix` 
        Input centroids
    n_threads : int
        Number of using threads

    Returns
    -------
    labels: `np.array`
        Array storing the assigned cluster for each sample.
    """
    if not isinstance(X, gb.Matrix):
        X = gb.io.from_scipy_sparse(X)

    if not isinstance(centroids, gb.Matrix):
        centroids = gb.io.from_scipy_sparse(centroids)

    centroids, is_centroid_dense = check_centroids_density(centroids)

    labels, _ = predict_labels(X, centroids, is_centroid_dense, n_threads)

    return labels


class SparseKmeans:
    def __init__(
        self,
        n_clusters: int = 8,
        n_threads: int = default_threads,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: Union[
            np.random.randint, np.random.RandomState, int, None
        ] = None,
        verbose: bool = False
    ):
        """Sparse K-means clustering.

        Parameters
        ----------
        n_clusters : int, default=8
            The predefined number of clusters

        n_threads : int, default=max(1, os.cpu_count() // 2)
            The predefined number of threads to use 

        max_iter : int, default=300
            Maximum number of iterations of the k-means algorithm.

        tol : float, default=1e-4
            Relative tolerance to declare convergence.

        random_state : int, RandomState instance or None, default=None
            Determines random number generation for centroid initialization.

        Attributes
        ----------
        centroids : `gb.Matrix`
            Centroids of each cluster derived from training on input matrix X.

        labels : `np.array`
            Cluster for each sample for input matrix X

        is_fitted: `bool`
            Boolean flag checking whether the model is fitted on datasets or not
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = check_random_state(random_state)
        self.n_threads = n_threads
        self.verbose = verbose

        # Flag to indicate if centroids stored in dense format
        self.is_centroid_dense = False

        self.is_fitted = False

    def _initialize_centroids(self, X: gb.Matrix):

        # The kmeans++ initialization method by Arthur and Vassilvitskii, 2007

        n_samples, n_features = X.shape
        n_clusters = self.n_clusters
        x_squared_norms = squared_row_norms(X, self.n_threads)

        centroids = Matrix(dtype=dtypes.FP64, nrows=n_clusters, ncols=n_features)

        # We can simply use self.random_state.choice(n_samples), but instead follow scikit-learn to allow
        # weights for the selection. Also, the two ways may give different results.
        first_centroid_id = self.random_state.choice(n_samples, p=[1 / n_samples] * n_samples)

        centroids[0, :] << X[first_centroid_id, :]

        Xc = Vector(dtype=dtypes.FP64, size=n_samples)
        Xc(nthreads=self.n_threads) << X.mxv(X[first_centroid_id, :])
        smallest_sq_dist = (
            x_squared_norms
            - 2 * Xc.to_dense(fill_value=0)
            + x_squared_norms[first_centroid_id]
        )
        # Maintain the potential, sum of each sample's smallest squared distance to existing centroids
        current_potential = smallest_sq_dist.sum()

        n_local_trials = 2 + int(np.log(n_clusters))

        for j in range(1, n_clusters):
            # Choose centroids candidates by sampling with probability smallest_sq_dist/current_potential, where
            # current_potential is the sum of smallest_sq_dist
            rand_vals = self.random_state.uniform(size=n_local_trials) * current_potential

            candidate_ids = np.searchsorted(np.cumsum(smallest_sq_dist), rand_vals)

            # The product between random values in [0, 1] and current_potential may numerically result in
            # rand_vals > current_potential. Then candidate_ids may be outside the desired range
            candidate_ids = np.minimum(candidate_ids, smallest_sq_dist.size - 1)

            # Compute squared distances to centroid candidates
            candidates = Matrix(dtype=dtypes.FP64, nrows=n_local_trials, ncols=n_features)
            candidates << X[candidate_ids, :]
            candidates = candidates.ss.export("csc")
            candidates = gb.Matrix.ss.import_csc(**candidates)

            Xcandidates_t = Matrix(dtype=dtypes.FP64, nrows=n_samples, ncols=n_local_trials)
            Xcandidates_t(nthreads=self.n_threads) << X.mxm(candidates.T)  # CSR * CSR
            sq_distance_to_candidates = (
                x_squared_norms[:, np.newaxis]
                - 2 * Xcandidates_t.to_dense(fill_value=0)
                + x_squared_norms[candidate_ids][np.newaxis, :]
            )
            sq_distance_to_candidates = sq_distance_to_candidates.T

            # Find the smallest squared distance from data to candidtaes
            sq_distance_to_candidates = np.minimum(
                smallest_sq_dist, sq_distance_to_candidates
            )
            candidates_potential = sq_distance_to_candidates.sum(axis=1)

            # Decide which candidate is the best
            best_candidate = np.argmin(candidates_potential)
            current_potential = candidates_potential[best_candidate]
            smallest_sq_dist = sq_distance_to_candidates[best_candidate]

            centroids[j, :] << X[candidate_ids[best_candidate], :]

        return centroids

    def _single_kmeans(self, X: gb.Matrix):
        # Allowing the kmeans procedure to allocate and initialize internal variables
        self._setup_internal_state(X)

        # Calculate the mean feature variance
        self.mean_feature_variance = self._cal_feature_variance(X)
        if self.mean_feature_variance <= sys.float_info.min:
            return

        for iter in range(self.max_iter):
            start_iter = time.time()

            # Decide to store centroids in dense/sparse format based on density
            self.centroids, self.is_centroid_dense = check_centroids_density(self.centroids)

            self._assign_cluster(X)

            if iter > 0:
                if self._converged():
                    break

            self.old_centroids = self.centroids
            self.centroids = self._update_centroids(X)

            # Allowing the kmeans procedure to update internal variables
            self._update_internal_state()

            end_iter = time.time()
            if self.verbose:
                print(
                    f"Time to conduct iteration: {iter}", end_iter - start_iter, flush=True
                )

        self.is_fitted = True

        return

    def fit(self, X: Union[sparse.csr_matrix, gb.Matrix]):
        """Conducting K-means clustering.

        Parameters
        ----------
        X : `gb.Matrix, scipy.sparse.csr_matrix` (csr format)
            Input dataset for clustering

        Returns
        -------.
        """

        n_samples = X.shape[0]
        print("Total samples: ", n_samples)

        if not isinstance(X, gb.Matrix):
            X = gb.io.from_scipy_sparse(X)

        start_init_centroids = time.time()
        self.centroids = self._initialize_centroids(X)
        end_init_centroids = time.time()

        if self.verbose:
            print("Initialze Centroids time: ", end_init_centroids - start_init_centroids)

        self._single_kmeans(X)

        self._cleanup()

        return self.labels

    def predict(self, X: gb.Matrix):
        """Predict the closest cluster for each sample in X

        Parameters
        ----------
        X : `gb.Matrix, scipy.sparse.csr_matrix` (csr format)
            Input dataset for predicting label

        Returns
        -------
        labels: `np.array`
            Array storing the assigned cluster for each sample.
        """

        if not self.is_fitted:
            raise AttributeError(
            f"This instance of Kmeans is not trained yet. "
            f"Call 'fit' before using this method."
        )

        if not isinstance(X, gb.Matrix):
            X = gb.io.from_scipy_sparse(X)

        labels, _ = predict_labels(X, self.centroids, self.is_centroid_dense, self.n_threads)

        return labels

    def _assign_cluster(self, *args, **kargs):
        pass

    def _update_centroids(self, X: gb.Matrix):
        n_samples, n_features = X.shape
        n_clusters = self.n_clusters

        cluster_sizes = np.bincount(self.labels, minlength=n_clusters)

        empty_clusters = np.where(cluster_sizes == 0)[0]
        n_empty = len(empty_clusters)

        # To handle empty clusters, we follow the setting in scikit-learn
        # We consider points with the largest distances to their assigned clusters, and reassign them to the empty clusters
        # Each empty cluster gets one point
        # We also need to update cluster_sizes
        if n_empty > 0:
            far_samples_idx = np.argpartition(self.sample_centroids_closest_distance, -n_empty)[
                : -n_empty - 1 : -1
            ]
            affected_clusters, counts = np.unique(
                self.labels[far_samples_idx], return_counts=True
            )
            cluster_sizes[affected_clusters] -= counts
            cluster_sizes[empty_clusters] = 1
            self.labels[far_samples_idx] = empty_clusters

        # Get the weight of each sample in its corresponding cluster
        weights = 1 / cluster_sizes
        weights = weights[self.labels]

        weighted_matrix = Matrix.from_coo(
            self.labels, np.arange(len(self.labels)), weights, nrows=n_clusters, ncols=n_samples
        )
        centroids = Matrix(dtypes.FP64, nrows=n_clusters, ncols=n_features)

        if self.is_centroid_dense:
            centroids << 0
            centroids(accum=gb.binary.plus, nthreads=self.n_threads) << weighted_matrix.mxm(X)
        else:
            centroids(nthreads=self.n_threads) << weighted_matrix.mxm(X)

        return centroids

    def _setup_internal_state(self, X: gb.Matrix):
        n_samples = X.shape[0]

        self.labels = np.zeros(n_samples, dtype=np.int16)

        # Maintain the distance between each sample and the assigned centroid
        # We need the distances in the function update_centroids for handling empty clusters
        self.sample_centroids_closest_distance = np.zeros(n_samples, dtype=np.float64)

        return

    def _cal_centroids_shift(self):
        n_clusters, n_features = self.centroids.shape

        centroids_shift_squared = Matrix(dtypes.FP64, nrows=n_clusters, ncols=n_features)
        centroids_shift_squared(nthreads=self.n_threads) << self.centroids.ewise_union(
            self.old_centroids, op="minus", left_default=0, right_default=0
        )
        centroids_shift_squared = squared_row_norms(centroids_shift_squared, self.n_threads)

        centroids_shift = np.sqrt(centroids_shift_squared)

        return centroids_shift

    def _cal_feature_variance(self, X: gb.Matrix):
        n_samples = X.shape[0]

        means = X.reduce_columnwise(op="add").to_dense(fill_value=0) / n_samples
        sum_squared = (X ** 2).reduce_columnwise(op="add").to_dense(fill_value=0)
        variances = (sum_squared - n_samples * (means ** 2)) / n_samples

        return np.mean(variances)

    def _converged(self):
        pass

    def _update_internal_state(self, *args, **kargs):
        pass

    def _cleanup(self, keep=['centroids', 'labels', 'is_fitted', 'is_centroid_dense', 'n_threads', 'verbose', 'random_state', 'n_clusters', 'max_iter', 'tol']):
        for attr in list(self.__dict__.keys()):
            if attr not in keep:
                delattr(self, attr)


class LloydKmeans(SparseKmeans):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _assign_cluster(self, X: gb.Matrix):
        self.labels, distances_to_centroids = predict_labels(X, self.centroids, self.is_centroid_dense, self.n_threads)
        np.min(distances_to_centroids, axis=1, out=self.sample_centroids_closest_distance)

        # Preventing memory issue with mxm operation
        gc.collect()

        return

    def _converged(self):

        centroids_shift = self._cal_centroids_shift()
        tol = (centroids_shift**2).sum()

        return tol <= self.mean_feature_variance * self.tol


class ElkanKmeans(SparseKmeans):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _assign_cluster(self, X: gb.Matrix):
        n_samples = X.shape[0]
        n_clusters = self.centroids.shape[0]

        samples_idx = np.arange(n_samples)

        c_squared_norms = squared_row_norms(self.centroids, self.n_threads)

        CCt = Matrix(dtypes.FP64, nrows=n_clusters, ncols=n_clusters)
        CCt(nthreads=self.n_threads) << self.centroids.mxm(self.centroids.T)
        CCt = CCt.to_dense(fill_value=0)

        half_centroid_centroid_distances = c_squared_norms[:, np.newaxis] - 2 * CCt + c_squared_norms[np.newaxis, :]
        np.clip(half_centroid_centroid_distances, 0, None, out=half_centroid_centroid_distances)
        half_centroid_centroid_distances = 0.5 * np.sqrt(half_centroid_centroid_distances)

        # We may do row-wise products between samples (i.e., X) and centroids[labels,:], but the expansion of
        # centroids is time/memory inefficient. Instead, we use a matrix-matrix product with mask.
        samples_centroids_product = Matrix(dtypes.FP64, nrows=n_samples, ncols=n_clusters)
        samples_centroids_mask = Matrix.from_coo(
            np.arange(n_samples), self.labels, values=1, nrows=n_samples, ncols=n_clusters
        )

        # Always use dense centroids for calculating samples_centroids_product here and Xcj in the loop
        if self.centroids.ss.format != "fullr":
            self.centroids(~self.centroids.S) << 0

        samples_centroids_product(mask=samples_centroids_mask.S, nthreads=self.n_threads) << X.mxm(self.centroids.T)
        samples_centroids_product = samples_centroids_product.reduce_rowwise(gb.binary.min).to_dense(fill_value=0)

        self.sample_centroids_closest_distance = self.x_squared_norms - 2 * samples_centroids_product + c_squared_norms[self.labels]
        np.clip(self.sample_centroids_closest_distance, 0, None, out=self.sample_centroids_closest_distance)
        self.sample_centroids_closest_distance = np.sqrt(self.sample_centroids_closest_distance)

        for j in range(n_clusters):

            # Apply two conditions first to get a subset as half_centroid_centroid_distances[j, labels] expensively expand a small array to a larger one
            candidate_idx = samples_idx[(self.labels != j) & (self.sample_centroids_closest_distance > self.lower_bounds[j, :])]
            candidate_idx = candidate_idx[self.sample_centroids_closest_distance[candidate_idx] > half_centroid_centroid_distances[j, self.labels[candidate_idx]]]

            if len(candidate_idx) == 0:
                continue

            Xcj = Vector(dtypes.FP64, size=n_samples)
            Xcj_mask = Vector.from_coo(candidate_idx, 1, size=n_samples)

            # Update lower bounds according to new centroids
            Xcj(mask=Xcj_mask.S, nthreads=self.n_threads) << X.mxv(self.centroids[j, :])
            Xcj = Xcj.to_dense(fill_value=0)
            distance_to_cj = self.x_squared_norms[candidate_idx] - 2 * Xcj[candidate_idx] + c_squared_norms[j]
            np.clip(distance_to_cj, 0, None, out=distance_to_cj)
            distance_to_cj = np.sqrt(distance_to_cj)
            self.lower_bounds[j, candidate_idx] = distance_to_cj

            reassignment_mask = self.sample_centroids_closest_distance[candidate_idx] > self.lower_bounds[j, candidate_idx]

            samples_to_reassign_idx = candidate_idx[reassignment_mask]

            if len(samples_to_reassign_idx) == 0:
                continue

            self.labels[samples_to_reassign_idx] = j
            update_distances = distance_to_cj[reassignment_mask]
            self.sample_centroids_closest_distance[samples_to_reassign_idx] = update_distances

        # Preventing memory issue with mxm operation
        gc.collect()

        return

    def _setup_internal_state(self, X: gb.Matrix):
        n_samples = X.shape[0]

        self.labels = np.zeros(n_samples, dtype=np.int16)

        # Maintain the distance between each sample and the assigned centroid
        # We need the distances in the function update_centroids for handling empty clusters
        self.sample_centroids_closest_distance = np.zeros(n_samples, dtype=np.float64)

        self.x_squared_norms = squared_row_norms(X, self.n_threads)

        self.lower_bounds = np.zeros((self.n_clusters, n_samples), dtype=np.float64)

        self.mean_feature_variance = self._cal_feature_variance(X)

        return

    def _update_internal_state(self):
        self.centroids_shift = self._cal_centroids_shift()
        self.lower_bounds -= self.centroids_shift[:, np.newaxis]
        np.maximum(self.lower_bounds, 0, self.lower_bounds)

        return

    def _converged(self):
        tol = (self.centroids_shift**2).sum()

        return tol <= self.mean_feature_variance * self.tol
