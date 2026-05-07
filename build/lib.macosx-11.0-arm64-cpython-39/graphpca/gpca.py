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
    location: np.ndarray = None, 
    n_components: int = 50, 
    network=None,               # Prioritize user-provided graph if not None
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
    mode: str = "accelerated",        # "exact", "iterative", or "accelerated"
    return_log: bool = False    # Compatibility parameter for ZW_log unpacking
):
    """
    Run Graph Principal Component Analysis (GPCA).
    
    Parameters:
    -----------
    mode : str
        "exact" (default): Direct matrix inversion. Best for small sample sizes.
        "iterative": Pure Python PCG alternating optimization. Good for medium-to-large sizes.
        "accelerated": C++ backend for PCG alternating optimization. Best for ultra-large sizes.
    """
    # 1. Initialization
    np.random.seed(random_seed)
    logger.info(f"Initializing GPCA in '{mode}' mode...")

    # 2. Construct spatial adjacency graph and Laplacian matrix
    if network is not None:
        # Use user-provided network graph directly
        graph = network
    else:
        # Construct graph based on location and platform
        n_neighbors_final = set_n_neighbors(platform, n_neighbors)
        if method == "knn":
            graph = kneighbors_graph(
                np.asarray(location), 
                n_neighbors_final, 
                metric='euclidean', 
                include_self=False
            )
            graph = 0.5 * (graph + graph.T)
        
    graphL = csgraph.laplacian(graph, normed=False)
    
    # Base Graph Laplacian matrix
    Phi = eye(adata.shape[0], format='csr') + _lambda * graphL
    ZW_log = []

    # ==========================================
    # Branch 1: Exact Solution (Direct Inversion)
    # ==========================================
    if mode == "exact":
        logger.info("Starting direct matrix inversion (Exact Mode)...")
        Expr = adata.X
        
        # Dense matrix format is safer and faster for exact direct matrix operations
        if issparse(Expr):
            Expr = Expr.todense()
        else:
            Expr = np.asarray(Expr)
            
        if issparse(Phi):
            Ginv = np.array(np.linalg.inv(Phi.todense()))
        else:
            Ginv = np.array(np.linalg.inv(Phi))
            
        C = np.dot(np.dot(Expr.T, Ginv), Expr)
        lambdas, W = np.linalg.eigh(C)
        
        # Select top n_components
        W = W[:, ::-1]
        W = W[:, :n_components]
        Z = np.dot(np.dot(Ginv, Expr), W)
        
        if return_log:
            ZW_log.append([Z.copy(), W.copy()])

    # ==========================================
    # Branch 2: Iterative Optimization (Python or C++)
    # ==========================================
    elif mode in ["iterative", "accelerated"]:
        # CSR Sparse format is optimal for PCG and C++ backend
        Expr = adata.X
        if not issparse(Expr):
            Expr = csr_matrix(Expr)
        else:
            Expr = Expr.tocsr()
            
        Phi = Phi.tocsr()
        n, m = Expr.shape

        # Initial SVD decomposition
        _, _, VT = randomized_svd(Expr, n_components=n_components, n_iter=5, random_state=random_seed)
        W = VT.T.copy() 
        Z = np.zeros((n, n_components), dtype=np.float64)

        if mode == "accelerated" and HAS_CPP:
            logger.info("Starting PCG optimization loop in C++ (Accelerated Mode)...")
            Z, W = gpca_cpp.run_gpca_iterations(Expr, Phi, Z, W, max_iter, kinner, tol)
            
        else:
            if mode == "accelerated" and not HAS_CPP:
                logger.warning("C++ module 'gpca_cpp' not found. Falling back to Python 'iterative' mode.")
            else:
                logger.info("Starting PCG optimization loop in Python (Iterative Mode)...")
                
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
                
                if return_log:
                    ZW_log.append([Z.copy(), W.copy()])
                
                if diff < tol:
                    logger.info(f"Converged successfully in {iteration + 1} iterations.")
                    break
            else:
                logger.warning(f"Reached maximum iterations ({max_iter}) without convergence.")

        # Matrix alignment (Only strictly needed for iterative/SVD-based outputs)
        if align:
            _, _, Vt_align = np.linalg.svd(Z, full_matrices=False)
            V_align = Vt_align.T
            Z = Z @ V_align
            W = W @ V_align

    else:
        raise ValueError("Invalid mode. Choose from 'exact', 'iterative', or 'accelerated'.")

    # ==========================================
    # Post-processing and Returns
    # ==========================================
    if save_reconstruction:
        adata.layers['GraphPCA_ReX'] = np.dot(Z, W.T) 

    if return_log:
        return Z, W, ZW_log
    
    return Z, W