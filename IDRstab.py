import numpy as np
from typing import Tuple, List
import scipy.io as sio
import requests
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import tempfile
import tarfile

class CRSMatrix:
    def __init__(self, values: np.ndarray, col_indices: np.ndarray, row_ptr: np.ndarray, shape: Tuple[int, int]):
        """
        Initialize CRS (Compressed Row Storage) format matrix
        
        Args:
            values: Non-zero values of the matrix
            col_indices: Column indices of non-zero values
            row_ptr: Row pointers indicating start of each row in values and col_indices
            shape: Matrix shape (rows, cols)
        """
        self.values = values
        self.col_indices = col_indices
        self.row_ptr = row_ptr
        self.shape = shape

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Matrix-vector multiplication"""
        result = np.zeros(self.shape[0])
        for i in range(self.shape[0]):
            start = self.row_ptr[i]
            end = self.row_ptr[i + 1]
            for j in range(start, end):
                result[i] += self.values[j] * x[self.col_indices[j]]
        return result

def diagonal_scaling(A: CRSMatrix, b: np.ndarray) -> Tuple[CRSMatrix, np.ndarray, np.ndarray]:
    """
    Apply diagonal scaling to the system Ax = b
    
    Args:
        A: CRS format matrix
        b: Right-hand side vector
        
    Returns:
        A_scaled: Scaled matrix
        b_scaled: Scaled right-hand side
        D: Diagonal scaling matrix
    """
    n = A.shape[0]
    D = np.zeros(n)
    
    # Compute diagonal elements
    for i in range(n):
        start = A.row_ptr[i]
        end = A.row_ptr[i + 1]
        for j in range(start, end):
            if A.col_indices[j] == i:
                D[i] = 1.0 / np.sqrt(abs(A.values[j]))
                break
    
    # Scale matrix and right-hand side
    values_scaled = A.values.copy()
    for i in range(n):
        start = A.row_ptr[i]
        end = A.row_ptr[i + 1]
        for j in range(start, end):
            col = A.col_indices[j]
            values_scaled[j] *= D[i] * D[col]
    
    A_scaled = CRSMatrix(
        values=values_scaled,
        col_indices=A.col_indices,
        row_ptr=A.row_ptr,
        shape=A.shape
    )
    
    b_scaled = b * D
    
    return A_scaled, b_scaled, D

def idrstab(A: CRSMatrix, b: np.ndarray, tol: float = 1e-8, max_iter: int = 1000,
            s: int = 4, ell: int = 2) -> Tuple[np.ndarray, int, float, List[float], List[float], int]:
    """
    Solve Ax = b using IDRstab method
    
    Args:
        A: CRS format matrix
        b: Right-hand side vector
        tol: Tolerance for convergence (relative residual)
        max_iter: Maximum number of iterations
        s: IDR parameter (dimension of the shadow space)
        ell: Stabilization frequency (stabilization step every ell iterations)
        
    Returns:
        x: Solution vector
        iter_count: Number of iterations performed
        residual: Final relative residual
        residual_history: List of relative residual norms during iterations
        error_history: List of error norms during iterations
        matvec_count: Number of matrix-vector multiplications performed
    """
    n = len(b)
    x = np.zeros(n)  # Initial guess
    r = b - A.matvec(x)  # Initial residual
    r0_norm = np.linalg.norm(r)  # Initial residual norm
    
    # Parameters
    omega = 1.0  # Stabilization parameter
    
    # Initialize vectors
    v = np.zeros((s, n))
    t = np.zeros((s, n))
    c = np.zeros(s)
    
    # Initialize shadow residuals with random numbers
    np.random.seed(42)  # For reproducibility
    matvec_count = 1  # Count initial matvec
    for k in range(s):
        v[k] = np.random.randn(n)
        v[k] = v[k] / np.linalg.norm(v[k])  # Normalize
        t[k] = A.matvec(v[k])
        matvec_count += 1
    
    iter_count = 0
    rel_residual = 1.0  # Initial relative residual
    residual_history = [rel_residual]
    error_history = [1.0]  # Initial error norm (normalized)
    
    # True solution for error calculation
    x_true = np.ones(n)
    x_true_norm = np.linalg.norm(x_true)
    
    while rel_residual > tol and iter_count < max_iter:
        # IDR step
        for k in range(s):
            # Compute coefficients
            for i in range(k):
                c[i] = np.dot(r, t[i]) / np.dot(t[i], t[i])
                r -= c[i] * t[i]
            
            # Update shadow residual
            v[k] = r
            t[k] = A.matvec(v[k])
            matvec_count += 1
            
            # Compute step size
            alpha = np.dot(r, t[k]) / np.dot(t[k], t[k])
            
            # Update solution and residual
            x += alpha * v[k]
            r -= alpha * t[k]
            
            # Stabilization step (every ell iterations)
            if k == s - 1 and (iter_count + 1) % ell == 0:
                t_stab = A.matvec(r)
                matvec_count += 1
                omega = np.dot(r, t_stab) / np.dot(t_stab, t_stab)
                x += omega * r
                r -= omega * t_stab
        
        rel_residual = np.linalg.norm(r) / r0_norm
        residual_history.append(rel_residual)
        
        # Calculate error norm
        error = np.linalg.norm(x - x_true) / x_true_norm
        error_history.append(error)
        
        iter_count += 1
        
    return x, iter_count, rel_residual, residual_history, error_history, matvec_count

def bicgstab(A: CRSMatrix, b: np.ndarray, tol: float = 1e-8, max_iter: int = 1000) -> Tuple[np.ndarray, int, float, List[float], List[float], int]:
    """
    Solve Ax = b using BiCGStab method
    
    Args:
        A: CRS format matrix
        b: Right-hand side vector
        tol: Tolerance for convergence (relative residual)
        max_iter: Maximum number of iterations
        
    Returns:
        x: Solution vector
        iter_count: Number of iterations performed
        residual: Final relative residual
        residual_history: List of relative residual norms during iterations
        error_history: List of error norms during iterations
        matvec_count: Number of matrix-vector multiplications performed
    """
    n = len(b)
    x = np.zeros(n)  # Initial guess
    r = b - A.matvec(x)  # Initial residual
    r0_norm = np.linalg.norm(r)  # Initial residual norm
    
    # Initialize vectors
    r0 = r.copy()  # Initial residual for BiCG
    p = r.copy()
    v = np.zeros(n)
    
    # Parameters
    rho = 1.0
    alpha = 1.0
    omega = 1.0
    
    iter_count = 0
    rel_residual = 1.0  # Initial relative residual
    residual_history = [rel_residual]
    error_history = [1.0]  # Initial error norm (normalized)
    matvec_count = 1  # Count initial matvec
    
    # True solution for error calculation
    x_true = np.ones(n)
    x_true_norm = np.linalg.norm(x_true)
    
    while rel_residual > tol and iter_count < max_iter:
        # BiCG step
        rho_old = rho
        rho = np.dot(r0, r)
        beta = (rho / rho_old) * (alpha / omega)
        p = r + beta * (p - omega * v)
        
        # Matrix-vector multiplication
        v = A.matvec(p)
        matvec_count += 1
        
        alpha = rho / np.dot(r0, v)
        s = r - alpha * v
        
        # Matrix-vector multiplication
        t = A.matvec(s)
        matvec_count += 1
        
        omega = np.dot(t, s) / np.dot(t, t)
        
        # Update solution and residual
        x += alpha * p + omega * s
        r = s - omega * t
        
        rel_residual = np.linalg.norm(r) / r0_norm
        residual_history.append(rel_residual)
        
        # Calculate error norm
        error = np.linalg.norm(x - x_true) / x_true_norm
        error_history.append(error)
        
        iter_count += 1
        
    return x, iter_count, rel_residual, residual_history, error_history, matvec_count

def create_test_matrix(n: int) -> Tuple[CRSMatrix, np.ndarray, np.ndarray]:
    """
    Create a test matrix in CRS format
    
    Args:
        n: Size of the matrix
        
    Returns:
        A: CRS format matrix
        x_true: True solution
        b: Right-hand side vector
    """
    # Create a simple tridiagonal matrix
    values = []
    col_indices = []
    row_ptr = [0]
    
    for i in range(n):
        if i > 0:
            values.append(-1.0)
            col_indices.append(i-1)
        values.append(4.0)
        col_indices.append(i)
        if i < n-1:
            values.append(-1.0)
            col_indices.append(i+1)
        row_ptr.append(len(values))
    
    A = CRSMatrix(np.array(values), np.array(col_indices), np.array(row_ptr), (n, n))
    
    # Create true solution and right-hand side
    x_true = np.ones(n)
    b = A.matvec(x_true)
    
    return A, x_true, b

def download_suitesparse_matrix(matrix_id: str) -> Tuple[CRSMatrix, np.ndarray]:
    """
    Download a matrix from SuiteSparse Matrix Collection
    
    Args:
        matrix_id: Matrix ID from SuiteSparse
        
    Returns:
        A: CRS format matrix
        b: Right-hand side vector
    """
    base_url = "https://suitesparse-collection-website.herokuapp.com/MM"
    matrix_url = f"{base_url}/{matrix_id}.tar.gz"
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Download matrix
        response = requests.get(matrix_url)
        tar_path = os.path.join(temp_dir, "matrix.tar.gz")
        with open(tar_path, 'wb') as f:
            f.write(response.content)
        
        # Extract matrix
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=temp_dir)
        
        # Find .mtx file
        mtx_file = None
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.mtx'):
                    mtx_file = os.path.join(root, file)
                    break
            if mtx_file:
                break
        
        if not mtx_file:
            raise FileNotFoundError("Matrix file not found in archive")
        
        # Load matrix
        matrix = sio.mmread(mtx_file)
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        # Convert to CRS format
        A = CRSMatrix(
            values=matrix.data,
            col_indices=matrix.indices,
            row_ptr=matrix.indptr,
            shape=matrix.shape
        )
        
        # Create right-hand side vector
        n = matrix.shape[0]
        x_true = np.ones(n)
        b = A.matvec(x_true)
        
        return A, b

def plot_convergence_history(residual_history: List[float], error_history: List[float], matrix_name: str):
    """
    Plot residual and error history
    
    Args:
        residual_history: List of relative residual norms
        error_history: List of error norms
        matrix_name: Name of the matrix for plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Plot residual history
    plt.subplot(1, 2, 1)
    iterations = range(len(residual_history))
    plt.semilogy(iterations, residual_history, 'b-', label='Relative residual')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Relative residual)')
    plt.title(f'Residual History - {matrix_name}')
    plt.legend()
    plt.axhline(y=1e-8, color='r', linestyle='--', label='Tolerance (1e-8)')
    plt.ylim(1e-10, 1.0)
    
    # Plot error history
    plt.subplot(1, 2, 2)
    plt.semilogy(iterations, error_history, 'g-', label='Error norm')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Relative error)')
    plt.title(f'Error History - {matrix_name}')
    plt.legend()
    plt.ylim(1e-10, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'convergence_history_{matrix_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def compare_methods(A: CRSMatrix, b: np.ndarray, matrix_name: str):
    """
    Compare IDRstab and BiCGStab methods
    
    Args:
        A: CRS format matrix
        b: Right-hand side vector
        matrix_name: Name of the matrix for plot title
    """
    # IDRstab parameters
    s = 8
    ell = 8
    
    # Solve with IDRstab
    x_idrstab, iter_idrstab, res_idrstab, res_hist_idrstab, err_hist_idrstab, matvec_idrstab = idrstab(
        A, b, s=s, ell=ell
    )
    
    # Solve with BiCGStab
    x_bicgstab, iter_bicgstab, res_bicgstab, res_hist_bicgstab, err_hist_bicgstab, matvec_bicgstab = bicgstab(
        A, b
    )
    
    # Print results
    print(f"\nResults for matrix: {matrix_name}")
    print(f"Matrix size: {A.shape}")
    print("\nIDRstab:")
    print(f"Parameters: s={s}, ell={ell}")
    print(f"Iterations: {iter_idrstab}")
    print(f"Matrix-vector multiplications: {matvec_idrstab}")
    print(f"Final relative residual: {res_idrstab:.2e}")
    print(f"Final relative error: {err_hist_idrstab[-1]:.2e}")
    
    print("\nBiCGStab:")
    print(f"Iterations: {iter_bicgstab}")
    print(f"Matrix-vector multiplications: {matvec_bicgstab}")
    print(f"Final relative residual: {res_bicgstab:.2e}")
    print(f"Final relative error: {err_hist_bicgstab[-1]:.2e}")
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    
    # Plot residual history
    plt.subplot(1, 2, 1)
    plt.semilogy(range(len(res_hist_idrstab)), res_hist_idrstab, 'b-', label='IDRstab')
    plt.semilogy(range(len(res_hist_bicgstab)), res_hist_bicgstab, 'r-', label='BiCGStab')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Relative residual)')
    plt.title(f'Residual History - {matrix_name}')
    plt.legend()
    plt.axhline(y=1e-8, color='k', linestyle='--', label='Tolerance (1e-8)')
    plt.ylim(1e-10, 1.0)
    
    # Plot error history
    plt.subplot(1, 2, 2)
    plt.semilogy(range(len(err_hist_idrstab)), err_hist_idrstab, 'b-', label='IDRstab')
    plt.semilogy(range(len(err_hist_bicgstab)), err_hist_bicgstab, 'r-', label='BiCGStab')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Relative error)')
    plt.title(f'Error History - {matrix_name}')
    plt.legend()
    plt.ylim(1e-10, 1.0)
    
    plt.tight_layout()
    plt.savefig(f'comparison_{matrix_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_suitesparse_matrices():
    """Test IDRstab with matrices from SuiteSparse"""
    # Test matrices (small to medium size)
    matrices = [
        "HB/sherman5"     # 3312x3312 petroleum reservoir simulation
    ]
    
    for matrix_id in matrices:
        print(f"\nTesting with matrix: {matrix_id}")
        try:
            A, b = download_suitesparse_matrix(matrix_id)
            
            # Apply diagonal scaling
            A_scaled, b_scaled, D = diagonal_scaling(A, b)
            
            # Compare methods
            matrix_name = matrix_id.replace('/', '_')
            compare_methods(A_scaled, b_scaled, matrix_name)
            
        except Exception as e:
            print(f"Error processing matrix {matrix_id}: {str(e)}")

if __name__ == "__main__":
    test_suitesparse_matrices()
