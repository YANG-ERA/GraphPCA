import time
import logging
import scanpy as sc
import numpy as np
import scipy
from scipy.sparse import csgraph, csr_matrix, diags, eye, issparse
from scipy.sparse.linalg import cg
from sklearn.neighbors import kneighbors_graph
from sklearn.utils.extmath import randomized_svd

# Initialize module-level logger
logger = logging.getLogger(__name__)

# ==========================================
# Attempt to dynamically load the C++ acceleration module.
# Fall back to pure Python if the compiled extension is not found to prevent crashes.
# ==========================================
try:
    import gpca_cpp 
    HAS_CPP = True
except ImportError:
    HAS_CPP = False

def set_n_neighbors(platform: str = None, n_neighbors: int = None) -> int:
    """Determine the number of neighbors based on the spatial platform."""
    if n_neighbors is not None:
        return int(n_neighbors)

    if platform == "Visium":
        return 6
    elif platform == "ST":
        return 4
    else:
        raise ValueError("Please specify 'n_neighbors' or provide a valid 'platform' (e.g., 'Visium' or 'ST').")


def Run_GPCA(
    adata, 
    location: np.ndarray, 
    n_components: int = 50, 
    network=None,               # Retained for backward compatibility to avoid parameter errors
    method: str = "knn", 
    platform: str = None, 
    _lambda: float = 0.5, 
    n_neighbors: int = None, 
    kinner: int = 500, 
    tol: float = 1e-6, 
    max_iter: int = 5, 
    random_seed: int = 666, 
    align: bool = True, 
    save_reconstruction: bool = False,
    mode: str = "standard",     # New acceleration parameter: "standard" or "accelerated"
    return_log: bool = False    # Compatibility parameter: Set to True if old tutorials unpack ZW_log
):
    """
    Run Graph Principal Component Analysis (GPCA).
    
    Parameters:
    -----------
    mode : str
        "standard" (default): Pure Python execution.
        "accelerated": Uses C++ backend for fast PCG optimization.
    return_log : bool
        If True, returns (Z, W, ZW_log) for backward compatibility. 
        If False, returns (Z, W) to save memory.
    """
    # 1. Initialization and setup
    np.random.seed(random_seed)
    
    # Convert to CSR format for optimal computational performance
    Expr = adata.X
    if not issparse(Expr):
        Expr = csr_matrix(Expr)
    else:
        Expr = Expr.tocsr()
        
    n, m = Expr.shape
    n_neighbors_final = set_n_neighbors(platform, n_neighbors)

    logger.info("Initializing GPCA...")

    # 2. Construct spatial adjacency graph and Laplacian matrix
    if method == "knn":
        graph = kneighbors_graph(
            np.asarray(location), 
            n_neighbors_final, 
            metric='euclidean', 
            include_self=False
        )
        graph = 0.5 * (graph + graph.T)
        
    graphL = csgraph.laplacian(graph, normed=False)
    
    Phi = eye(location.shape[0], format='csr') + _lambda * graphL
    Phi = Phi.tocsr()

    # 3. Initial SVD decomposition
    _, _, VT = randomized_svd(Expr, n_components=n_components, n_iter=5, random_state=random_seed)
    W = VT.T.copy() 
    Z = np.zeros((n, n_components), dtype=np.float64)
    ZW_log = []

    # ==========================================
    # 4. Alternating optimization iterations
    # ==========================================
    if mode == "accelerated" and HAS_CPP:
        logger.info("Starting PCG optimization loop in C++ (Accelerated Mode)...")
        # Call C++; the C++ version does not return ZW_log to save memory
        Z, W = gpca_cpp.run_gpca_iterations(Expr, Phi, Z, W, max_iter, kinner, tol)
        
    else:
        if mode == "accelerated" and not HAS_CPP:
            logger.warning("C++ module 'gpca_cpp' not found. Falling back to Python 'standard' mode.")
        else:
            logger.info("Starting PCG optimization loop in Python (Standard Mode)...")
            
        M = diags(Phi.diagonal())
        for iteration in range(max_iter):
            iter_start_time = time.time()
            logger.info(f"Iteration {iteration + 1}/{max_iter} started.")

            Z_old = Z.copy()
            
            # Update Z (PCG solver)
            for i in range(n_components):
                b = Expr @ W[:, i] 
                z_i, _ = cg(Phi, b, x0=Z[:, i], maxiter=kinner, M=M)
                Z[:, i] = z_i
                
            # Update W (SVD decomposition)
            ZtExpr = Z.T @ Expr
            G, _, Vt = np.linalg.svd(ZtExpr, full_matrices=False)
            W = Vt.T @ G.T
            
            diff = np.linalg.norm(Z - Z_old, ord='fro') / (np.linalg.norm(Z, ord='fro') + 1e-12)
            elapsed_time = time.time() - iter_start_time
            
            logger.info(f"Iteration {iteration + 1} completed in {elapsed_time:.2f}s | Diff: {diff:.4e}")
            
            # Record log if backward compatibility is required
            if return_log:
                ZW_log.append([Z.copy(), W.copy()])
            
            if diff < tol:
                logger.info(f"Converged successfully in {iteration + 1} iterations.")
                break
        else:
            logger.warning(f"Reached maximum iterations ({max_iter}) without convergence.")

    # 5. Matrix alignment and post-processing
    if align:
        _, _, Vt_align = np.linalg.svd(Z, full_matrices=False)
        V_align = Vt_align.T
        Z = Z @ V_align
        W = W @ V_align

    if save_reconstruction:
        adata.layers['GraphPCA_ReX'] = np.dot(Z, W.T) 

    # 6. Return results (compatible with old unpacking methods)
    if return_log:
        return Z, W, ZW_log
    
    return Z, W