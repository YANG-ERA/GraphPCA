"""Multi-sample GraphPCA-Turbo.

This module extends GraphPCA-Turbo to jointly analyse multiple spatial
transcriptomics samples. Each sample has its own spatial embedding and graph,
whereas all samples share a common orthonormal gene-loading matrix.

The implemented objective is

    sum_s omega_s [ ||X_s - Z_s W.T||_F^2
                    + lambda_s tr(Z_s.T L_s Z_s) ],
    subject to W.T W = I.

For fixed W, each Z_s is obtained from the sparse SPD system

    (I + lambda_s L_s) Z_s = X_s W.

For fixed {Z_s}, W is updated by an orthogonal Procrustes step based on
sum_s omega_s X_s.T Z_s.

Sample-wise centring is implemented implicitly and therefore does not densify
sparse expression matrices.
"""

from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, issparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import cg, spsolve
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger(__name__)

try:
    import gpca_cpp
    HAS_CPP_MULTI = hasattr(gpca_cpp, "solve_gpca_embedding")
except ImportError:
    HAS_CPP_MULTI = False

ArrayLike = Union[np.ndarray, csr_matrix]
ScalarOrSequence = Union[float, int, Sequence[Union[float, int]]]


@dataclass
class MultiGPCAInfo:
    """Diagnostics returned by :func:`Run_Multi_GPCA`."""

    converged: bool
    n_iter: int
    relative_changes: List[float]
    sample_weights: np.ndarray
    lambdas: np.ndarray
    gene_means: List[np.ndarray]
    pcg_status: List[List[int]]


def _as_list(value: Any, length: int, name: str) -> List[Any]:
    if value is None or isinstance(value, (str, bytes)) or np.isscalar(value):
        return [value] * length
    try:
        values = list(value)
    except TypeError:
        return [value] * length
    if len(values) != length:
        raise ValueError(f"'{name}' must have length {length}; received {len(values)}.")
    return values


def _expression_matrix(adata: Any) -> csr_matrix:
    if not hasattr(adata, "X"):
        raise TypeError("Each sample must provide an '.X' expression matrix.")
    x = adata.X
    if issparse(x):
        return x.tocsr().astype(np.float64, copy=False)
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Each expression matrix must be two-dimensional.")
    return csr_matrix(x)


def _validate_samples(adatas: Sequence[Any], n_components: int) -> int:
    if len(adatas) < 2:
        raise ValueError("Multi-sample GraphPCA requires at least two samples.")

    shapes = [adata.shape for adata in adatas]
    p = int(shapes[0][1])
    if any(int(shape[1]) != p for shape in shapes):
        raise ValueError(
            "All samples must contain the same number of genes. Align genes "
            "and use the same ordering before calling Run_Multi_GPCA."
        )
    if not 1 <= n_components <= p:
        raise ValueError(f"n_components must lie in [1, {p}].")

    # If AnnData-style gene names are available, require identical ordering.
    first_names = getattr(adatas[0], "var_names", None)
    if first_names is not None:
        first_names = np.asarray(first_names).astype(str)
        for idx, adata in enumerate(adatas[1:], start=1):
            names = getattr(adata, "var_names", None)
            if names is None:
                continue
            names = np.asarray(names).astype(str)
            if names.shape != first_names.shape or not np.array_equal(names, first_names):
                raise ValueError(
                    f"Gene names/order differ between sample 0 and sample {idx}. "
                    "Subset and reorder all samples to identical var_names first."
                )
    return p


def _set_n_neighbors(platform: Optional[str], n_neighbors: Optional[int]) -> int:
    if n_neighbors is not None:
        if int(n_neighbors) < 1:
            raise ValueError("n_neighbors must be positive.")
        return int(n_neighbors)
    if platform == "Visium":
        return 6
    if platform == "ST":
        return 4
    raise ValueError(
        "Specify n_neighbors for every sample or provide platform='Visium'/'ST'."
    )


def _build_phi(
    n_obs: int,
    location: Optional[np.ndarray],
    network: Optional[ArrayLike],
    platform: Optional[str],
    n_neighbors: Optional[int],
    graph_lambda: float,
) -> csr_matrix:
    if graph_lambda < 0:
        raise ValueError("All graph regularisation parameters must be non-negative.")

    if network is None:
        if location is None:
            raise ValueError("Each sample requires either a location matrix or a network.")
        location = np.asarray(location)
        if location.ndim != 2 or location.shape[0] != n_obs:
            raise ValueError(
                f"Location matrix must have shape ({n_obs}, d); received {location.shape}."
            )
        k = _set_n_neighbors(platform, n_neighbors)
        if k >= n_obs:
            raise ValueError(
                f"n_neighbors={k} must be smaller than the sample size ({n_obs})."
            )
        graph = kneighbors_graph(
            location,
            n_neighbors=k,
            metric="euclidean",
            include_self=False,
            mode="connectivity",
        ).tocsr()
    else:
        graph = csr_matrix(network, dtype=np.float64)
        if graph.shape != (n_obs, n_obs):
            raise ValueError(
                f"Network must have shape ({n_obs}, {n_obs}); received {graph.shape}."
            )

    # Match the current GraphPCA implementation: use an undirected average graph.
    graph = (0.5 * (graph + graph.T)).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    laplacian = csgraph.laplacian(graph, normed=False).tocsr()
    return (eye(n_obs, format="csr") + graph_lambda * laplacian).tocsr()


def _gene_mean(x: csr_matrix, center: bool) -> np.ndarray:
    if not center:
        return np.zeros(x.shape[1], dtype=np.float64)
    return np.asarray(x.mean(axis=0)).ravel().astype(np.float64, copy=False)


def _centered_matmul(x: csr_matrix, mean: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute (X - 1 mean.T) @ right without forming a dense centred X."""
    result = np.asarray(x @ right, dtype=np.float64)
    correction = mean @ right
    result -= correction[None, :]
    return result


def _centered_t_matmul(x: csr_matrix, mean: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compute (X - 1 mean.T).T @ right without densifying X."""
    result = np.asarray(x.T @ right, dtype=np.float64)
    result -= np.outer(mean, np.asarray(right.sum(axis=0)).ravel())
    return result


def _resolve_weights(
    adatas: Sequence[Any], sample_weights: Optional[Union[str, Sequence[float]]]
) -> np.ndarray:
    n_obs = np.asarray([adata.shape[0] for adata in adatas], dtype=np.float64)
    if sample_weights is None or (
        isinstance(sample_weights, str) and sample_weights == "balanced"
    ):
        weights = 1.0 / n_obs
    elif isinstance(sample_weights, str) and sample_weights == "spot":
        # Equal coefficient per sample loss; larger samples then contribute more
        # because their Frobenius losses contain more observations.
        weights = np.ones_like(n_obs)
    elif isinstance(sample_weights, str):
        raise ValueError("sample_weights must be 'balanced', 'spot', or a positive sequence.")
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != n_obs.shape:
            raise ValueError(f"sample_weights must have length {len(adatas)}.")
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("All sample weights must be finite and strictly positive.")
    return weights / weights.sum()


def _streaming_initial_loading(
    expressions: Sequence[csr_matrix],
    means: Sequence[np.ndarray],
    weights: np.ndarray,
    n_components: int,
    random_seed: int,
    oversample: int = 10,
    power_iter: int = 2,
) -> np.ndarray:
    """Randomized eigensolver for the weighted pooled centred covariance.

    It evaluates sum_s w_s X_s.T X_s times a thin matrix sample-by-sample,
    avoiding concatenation of all observations or construction of a p x p
    covariance matrix.
    """
    p = expressions[0].shape[1]
    rank = min(p, n_components + max(0, int(oversample)))
    rng = np.random.default_rng(random_seed)
    q, _ = np.linalg.qr(rng.standard_normal((p, rank)), mode="reduced")

    for _ in range(max(1, int(power_iter))):
        b = np.zeros((p, rank), dtype=np.float64)
        for x, mean, weight in zip(expressions, means, weights):
            xq = _centered_matmul(x, mean, q)
            b += weight * _centered_t_matmul(x, mean, xq)
        q, _ = np.linalg.qr(b, mode="reduced")

    small_cov = np.zeros((q.shape[1], q.shape[1]), dtype=np.float64)
    for x, mean, weight in zip(expressions, means, weights):
        xq = _centered_matmul(x, mean, q)
        small_cov += weight * (xq.T @ xq)
    eigenvalues, eigenvectors = np.linalg.eigh(small_cov)
    order = np.argsort(eigenvalues)[::-1][:n_components]
    w = q @ eigenvectors[:, order]
    w, _ = np.linalg.qr(w, mode="reduced")
    return w[:, :n_components]


def _cg_compat(
    a: csr_matrix,
    b: np.ndarray,
    x0: np.ndarray,
    preconditioner: csr_matrix,
    tol: float,
    maxiter: int,
) -> Tuple[np.ndarray, int]:
    """Call SciPy CG across old and new tolerance APIs."""
    try:
        return cg(
            a,
            b,
            x0=x0,
            M=preconditioner,
            maxiter=maxiter,
            rtol=tol,
            atol=0.0,
        )
    except TypeError:  # SciPy < 1.12
        return cg(a, b, x0=x0, M=preconditioner, maxiter=maxiter, tol=tol)


def _solve_embedding(
    x: csr_matrix,
    mean: np.ndarray,
    phi: csr_matrix,
    w: np.ndarray,
    z0: np.ndarray,
    pcg_tol: float,
    pcg_max_iter: int,
    use_cpp: bool = False,
) -> Tuple[np.ndarray, List[int]]:
    if use_cpp:
        z, status = gpca_cpp.solve_gpca_embedding(
            x, phi, z0, w, mean, pcg_max_iter, pcg_tol
        )
        return np.asarray(z), [int(code) for code in np.asarray(status).ravel()]

    rhs = _centered_matmul(x, mean, w)
    z = np.empty_like(rhs)

    diagonal = phi.diagonal().astype(np.float64, copy=False)
    if np.any(diagonal <= 0):
        raise ValueError("The PCG coefficient matrix must have a positive diagonal.")
    jacobi_inverse = diags(1.0 / diagonal, format="csr")

    status: List[int] = []
    for component in range(w.shape[1]):
        solution, info = _cg_compat(
            phi,
            rhs[:, component],
            z0[:, component],
            jacobi_inverse,
            pcg_tol,
            pcg_max_iter,
        )
        z[:, component] = solution
        status.append(int(info))
    return z, status


def _procrustes_loading(
    expressions: Sequence[csr_matrix],
    means: Sequence[np.ndarray],
    embeddings: Sequence[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    p = expressions[0].shape[1]
    k = embeddings[0].shape[1]
    cross_covariance = np.zeros((p, k), dtype=np.float64)
    for x, mean, z, weight in zip(expressions, means, embeddings, weights):
        cross_covariance += weight * _centered_t_matmul(x, mean, z)

    u, _, vt = np.linalg.svd(cross_covariance, full_matrices=False)
    return u @ vt


def _align_joint_basis(
    embeddings: Sequence[np.ndarray],
    loading: np.ndarray,
    weights: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    # Equivalent to right-singular alignment of a weighted stacked Z, but only
    # requires a k x k Gram matrix.
    gram = np.zeros((loading.shape[1], loading.shape[1]), dtype=np.float64)
    for z, weight in zip(embeddings, weights):
        gram += weight * (z.T @ z)
    _, rotation = np.linalg.eigh(gram)
    rotation = rotation[:, ::-1]
    return [z @ rotation for z in embeddings], loading @ rotation


def _weighted_relative_change(
    current: Sequence[np.ndarray], previous: Sequence[np.ndarray], weights: np.ndarray
) -> float:
    numerator = 0.0
    denominator = 0.0
    for z, old, weight in zip(current, previous, weights):
        numerator += weight * float(np.sum((z - old) ** 2))
        denominator += weight * float(np.sum(z**2))
    return float(np.sqrt(numerator) / (np.sqrt(denominator) + 1e-12))


def Run_Multi_GPCA(
    adatas: Sequence[Any],
    locations: Optional[Sequence[Optional[np.ndarray]]] = None,
    n_components: int = 50,
    networks: Optional[Sequence[Optional[ArrayLike]]] = None,
    platforms: Optional[Union[str, Sequence[Optional[str]]]] = None,
    lambdas: ScalarOrSequence = 0.5,
    n_neighbors: Optional[Union[int, Sequence[Optional[int]]]] = None,
    sample_weights: Optional[Union[str, Sequence[float]]] = "balanced",
    center: bool = True,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    outer_tol: float = 1e-6,
    max_iter: int = 10,
    random_seed: int = 666,
    align: bool = True,
    n_jobs: int = 1,
    mode: str = "iterative",
    save_embedding: bool = True,
    embedding_key: str = "X_GraphPCA_MS",
    save_reconstruction: bool = False,
    reconstruction_key: str = "GraphPCA_MS_ReX",
    return_log: bool = False,
    return_info: bool = False,
):
    """Run shared-loading multi-sample GraphPCA-Turbo.

    Parameters
    ----------
    adatas
        Sequence of AnnData-like objects. All samples must contain exactly the
        same genes in the same order.
    locations
        Sequence of spatial coordinate matrices. Not required for samples with
        a user-provided network.
    n_components
        Number of shared latent dimensions.
    networks
        Sequence of optional sample-specific adjacency matrices.
    platforms
        A scalar or sequence containing ``'Visium'`` or ``'ST'``. Only used to
        infer the number of neighbours when ``n_neighbors`` is not supplied.
    lambdas
        Scalar or sample-specific graph regularisation strengths.
    n_neighbors
        Scalar or sample-specific number of spatial neighbours.
    sample_weights
        ``'balanced'`` (default) uses weights proportional to 1/n_s;
        ``'spot'`` applies the same coefficient to every sample loss, allowing
        larger samples to contribute more; a positive custom sequence is also
        accepted.
    center
        Remove sample-specific gene means implicitly. This does not densify
        sparse matrices.
    pcg_tol, pcg_max_iter
        Tolerance and maximum iterations for each PCG solve.
    outer_tol, max_iter
        Convergence tolerance and maximum alternating iterations.
    n_jobs
        Number of sample-level threads used for the Z updates. Use cautiously
        when BLAS/OpenMP already uses multiple threads.
    mode
        ``'iterative'`` is currently implemented. ``'accelerated'`` falls back
        to the Python joint optimizer because the repository's existing C++
        kernel updates a sample-specific W internally and therefore cannot be
        used for a shared-W objective without a revised extension.
    save_embedding
        Store each Z_s in ``adata.obsm[embedding_key]`` when possible.
    return_log
        Return iteration snapshots. This can be memory intensive.
    return_info
        Return a :class:`MultiGPCAInfo` diagnostic object.

    Returns
    -------
    Z_list, W
        Sample-specific embeddings and the shared gene-loading matrix.
        Optional logs and diagnostics are appended according to flags.
    """
    adatas = list(adatas)
    p = _validate_samples(adatas, n_components)
    n_samples = len(adatas)

    if mode not in {"iterative", "accelerated", "exact"}:
        raise ValueError("mode must be 'iterative', 'accelerated', or 'exact'.")
    use_cpp = mode == "accelerated" and HAS_CPP_MULTI
    if mode == "accelerated" and not HAS_CPP_MULTI:
        warnings.warn(
            "The installed gpca_cpp extension does not expose the multi-sample "
            "Z-only PCG kernel. Falling back to the Python iterative solver. "
            "Rebuild the extension using the supplied gpca_core.cpp patch to "
            "enable accelerated multi-sample updates.",
            RuntimeWarning,
            stacklevel=2,
        )
        mode = "iterative"

    if pcg_tol <= 0 or outer_tol <= 0:
        raise ValueError("pcg_tol and outer_tol must be positive.")
    if pcg_max_iter < 1 or max_iter < 1:
        raise ValueError("pcg_max_iter and max_iter must be positive integers.")
    if n_jobs < 1:
        raise ValueError("n_jobs must be at least one.")

    locations_list = (
        [None] * n_samples if locations is None else _as_list(locations, n_samples, "locations")
    )
    networks_list = (
        [None] * n_samples if networks is None else _as_list(networks, n_samples, "networks")
    )
    platforms_list = _as_list(platforms, n_samples, "platforms")
    lambdas_array = np.asarray(_as_list(lambdas, n_samples, "lambdas"), dtype=np.float64)
    neighbors_list = _as_list(n_neighbors, n_samples, "n_neighbors")
    weights = _resolve_weights(adatas, sample_weights)

    expressions = [_expression_matrix(adata) for adata in adatas]
    means = [_gene_mean(x, center) for x in expressions]
    phis = [
        _build_phi(
            adata.shape[0],
            location,
            network,
            platform,
            neighbors,
            graph_lambda,
        )
        for adata, location, network, platform, neighbors, graph_lambda in zip(
            adatas,
            locations_list,
            networks_list,
            platforms_list,
            neighbors_list,
            lambdas_array,
        )
    ]

    logger.info(
        "Initializing multi-sample GraphPCA: %d samples, %d spots, %d genes, %d components.",
        n_samples,
        sum(adata.shape[0] for adata in adatas),
        p,
        n_components,
    )

    if mode == "exact":
        # Intended for small datasets only. Solve Phi_s Y_s = Xc_s exactly,
        # aggregate H = sum w_s Xc_s.T Y_s, and recover Z_s = Y_s W.
        smoothed_expressions: List[np.ndarray] = []
        covariance = np.zeros((p, p), dtype=np.float64)
        for x, mean, phi, weight in zip(expressions, means, phis, weights):
            dense_x = x.toarray()
            dense_x -= mean[None, :]
            y = np.asarray(spsolve(phi, dense_x), dtype=np.float64)
            smoothed_expressions.append(y)
            covariance += weight * (dense_x.T @ y)
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        w = eigenvectors[:, np.argsort(eigenvalues)[::-1][:n_components]]
        z_list = [y @ w for y in smoothed_expressions]
        relative_changes: List[float] = []
        pcg_status: List[List[int]] = [[] for _ in adatas]
        converged = True
        iterations_run = 1
        logs: List[Tuple[List[np.ndarray], np.ndarray]] = []
    else:
        w = _streaming_initial_loading(
            expressions,
            means,
            weights,
            n_components,
            random_seed,
        )
        z_list = [np.zeros((adata.shape[0], n_components), dtype=np.float64) for adata in adatas]
        logs = []
        relative_changes = []
        pcg_status = [[] for _ in adatas]
        converged = False
        iterations_run = max_iter

        def solve_one(index: int, previous_z: np.ndarray):
            return _solve_embedding(
                expressions[index],
                means[index],
                phis[index],
                w,
                previous_z,
                pcg_tol,
                pcg_max_iter,
                use_cpp=use_cpp,
            )

        for iteration in range(max_iter):
            start = time.time()
            previous = [z.copy() for z in z_list]

            if n_jobs == 1:
                results = [solve_one(i, previous[i]) for i in range(n_samples)]
            else:
                with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                    futures = [executor.submit(solve_one, i, previous[i]) for i in range(n_samples)]
                    results = [future.result() for future in futures]

            z_list = [result[0] for result in results]
            statuses = [result[1] for result in results]
            pcg_status = statuses
            for sample_index, status in enumerate(statuses):
                failed = [i for i, code in enumerate(status) if code != 0]
                if failed:
                    logger.warning(
                        "Sample %d: PCG did not fully converge for %d/%d components; "
                        "status codes=%s.",
                        sample_index,
                        len(failed),
                        n_components,
                        sorted(set(status[i] for i in failed)),
                    )

            w = _procrustes_loading(expressions, means, z_list, weights)
            change = _weighted_relative_change(z_list, previous, weights)
            relative_changes.append(change)
            logger.info(
                "Multi-sample iteration %d/%d completed in %.2fs | Diff: %.4e",
                iteration + 1,
                max_iter,
                time.time() - start,
                change,
            )

            if return_log:
                logs.append(([z.copy() for z in z_list], w.copy()))
            if change < outer_tol:
                converged = True
                iterations_run = iteration + 1
                logger.info("Multi-sample GraphPCA converged in %d iterations.", iterations_run)
                break
        else:
            logger.warning("Reached maximum alternating iterations (%d).", max_iter)

    if align:
        z_list, w = _align_joint_basis(z_list, w, weights)

    if save_embedding:
        for adata, z in zip(adatas, z_list):
            if hasattr(adata, "obsm"):
                adata.obsm[embedding_key] = z

    if save_reconstruction:
        for adata, z, mean in zip(adatas, z_list, means):
            reconstruction = z @ w.T
            if center:
                reconstruction += mean[None, :]
            if hasattr(adata, "layers"):
                adata.layers[reconstruction_key] = reconstruction

    info = MultiGPCAInfo(
        converged=converged,
        n_iter=iterations_run,
        relative_changes=relative_changes,
        sample_weights=weights.copy(),
        lambdas=lambdas_array.copy(),
        gene_means=[mean.copy() for mean in means],
        pcg_status=pcg_status,
    )

    output: List[Any] = [z_list, w]
    if return_log:
        output.append(logs)
    if return_info:
        output.append(info)
    return tuple(output)


def Project_Multi_GPCA(
    adata: Any,
    loading: np.ndarray,
    location: Optional[np.ndarray] = None,
    network: Optional[ArrayLike] = None,
    platform: Optional[str] = None,
    graph_lambda: float = 0.5,
    n_neighbors: Optional[int] = None,
    center: bool = True,
    gene_mean: Optional[np.ndarray] = None,
    pcg_tol: float = 1e-6,
    pcg_max_iter: int = 500,
    save_embedding: bool = True,
    embedding_key: str = "X_GraphPCA_MS",
) -> np.ndarray:
    """Project an unseen sample into a previously learned shared loading space."""
    x = _expression_matrix(adata)
    loading = np.asarray(loading, dtype=np.float64)
    if loading.ndim != 2 or loading.shape[0] != x.shape[1]:
        raise ValueError(
            f"loading must have shape ({x.shape[1]}, k); received {loading.shape}."
        )
    if gene_mean is None:
        mean = _gene_mean(x, center)
    else:
        mean = np.asarray(gene_mean, dtype=np.float64)
        if mean.shape != (x.shape[1],):
            raise ValueError(f"gene_mean must have shape ({x.shape[1]},).")

    phi = _build_phi(
        x.shape[0],
        location,
        network,
        platform,
        n_neighbors,
        graph_lambda,
    )
    z0 = np.zeros((x.shape[0], loading.shape[1]), dtype=np.float64)
    z, status = _solve_embedding(x, mean, phi, loading, z0, pcg_tol, pcg_max_iter)
    if any(code != 0 for code in status):
        logger.warning("PCG did not fully converge during out-of-sample projection.")
    if save_embedding and hasattr(adata, "obsm"):
        adata.obsm[embedding_key] = z
    return z


# PEP-8 aliases while retaining the repository's existing Run_GPCA naming style.
run_multi_gpca = Run_Multi_GPCA
project_multi_gpca = Project_Multi_GPCA
