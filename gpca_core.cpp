#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>   
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <chrono>
#include <tuple>

namespace py = pybind11;

std::tuple<Eigen::MatrixXd, Eigen::MatrixXd> 
run_gpca_iterations(
    const Eigen::SparseMatrix<double>& Expr, 
    const Eigen::SparseMatrix<double>& Phi,
    Eigen::MatrixXd Z, 
    Eigen::MatrixXd W,
    int max_iter, 
    int kinner, 
    double tol) {
    
    int n_components = W.cols();
    
    // Dynamically fetch the Python logger named "gpca_cpp"
    py::object logging = py::module_::import("logging");
    py::object logger = logging.attr("getLogger")("gpca_cpp");

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::DiagonalPreconditioner<double>> cg;
    cg.setMaxIterations(kinner);
    cg.setTolerance(1e-5); 
    cg.compute(Phi);

    for (int iter = 0; iter < max_iter; ++iter) {
        // Record start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Log the start of the iteration
        logger.attr("info")(py::str("Iteration {}/{} started.").format(iter + 1, max_iter));

        Eigen::MatrixXd Z_old = Z;
        
        // 1. Update Z (PCG solver)
        for (int i = 0; i < n_components; ++i) {
            Eigen::VectorXd b = Expr * W.col(i);
            Z.col(i) = cg.solveWithGuess(b, Z.col(i)); 
        }
        
        // 2. Update W (SVD decomposition)
        Eigen::MatrixXd ZtExpr = Z.transpose() * Expr;
        Eigen::BDCSVD<Eigen::MatrixXd> svd(ZtExpr, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Eigen::MatrixXd U = svd.matrixU();
        Eigen::MatrixXd V = svd.matrixV();
        W = V * U.transpose();
        
        // 3. Calculate convergence difference
        double diff = (Z - Z_old).norm() / (Z.norm() + 1e-12);
        
        // Calculate elapsed time
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end_time - start_time;
        
        // Format and output completion log for current iteration
        char buffer[128];
        snprintf(buffer, sizeof(buffer), "Iteration %d completed in %.2fs | Diff: %.4e", iter + 1, elapsed.count(), diff);
        logger.attr("info")(buffer);
        
        // 4. Check for convergence
        if (diff < tol) {
            logger.attr("info")(py::str("Converged successfully in {} iterations.").format(iter + 1));
            break;
        }
        if (iter == max_iter - 1) {
            logger.attr("warning")(py::str("Reached maximum iterations ({}) without convergence.").format(max_iter));
        }
    }
    
    // Return only Z and W to save memory footprint
    return std::make_tuple(Z, W);
}

PYBIND11_MODULE(gpca_cpp, m) {
    m.doc() = "C++ acceleration for GPCA iterations";
    m.def("run_gpca_iterations", &run_gpca_iterations, 
          "Run GPCA core iterations",
          py::arg("Expr"), py::arg("Phi"), py::arg("Z"), py::arg("W"), 
          py::arg("max_iter"), py::arg("kinner"), py::arg("tol"));
}