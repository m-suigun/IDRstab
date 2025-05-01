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
            s: int = 4, ell: int = 2) -> Tuple[np.ndarray, int, float, List[float]]:
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
    
    iter_count = 0
    rel_residual = 1.0  # Initial relative residual
    residual_history = [rel_residual]
    
    while rel_residual > tol and iter_count < max_iter:
        # IDR step
        for k in range(s):
            v[k] = r
            t[k] = A.matvec(v[k])
            
            # Compute coefficients
            for i in range(k):
                c[i] = np.dot(t[k], t[i]) / np.dot(t[i], t[i])
                v[k] -= c[i] * v[i]
                t[k] -= c[i] * t[i]
            
            # Update solution and residual
            alpha = np.dot(r, t[k]) / np.dot(t[k], t[k])
            x += alpha * v[k]
            r -= alpha * t[k]
            
            # Stabilization step (every ell iterations)
            if k == s - 1 and (iter_count + 1) % ell == 0:
                t_stab = A.matvec(r)
                omega = np.dot(r, t_stab) / np.dot(t_stab, t_stab)
                x += omega * r
                r -= omega * t_stab
        
        rel_residual = np.linalg.norm(r) / r0_norm
        residual_history.append(rel_residual)
        iter_count += 1
        
    return x, iter_count, rel_residual, residual_history

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

def plot_residual_history(residual_history: List[float], matrix_name: str):
    """
    Plot residual history
    
    Args:
        residual_history: List of relative residual norms
        matrix_name: Name of the matrix for plot title
    """
    plt.figure(figsize=(10, 6))
    iterations = range(len(residual_history))
    plt.semilogy(iterations, residual_history, 'b-', label='Relative residual')
    plt.grid(True)
    plt.xlabel('Iteration')
    plt.ylabel('log₁₀(Relative residual)')
    plt.title(f'Convergence History - {matrix_name}')
    plt.legend()
    
    # Add horizontal line at tolerance level
    plt.axhline(y=1e-8, color='r', linestyle='--', label='Tolerance (1e-8)')
    
    # Set y-axis limits to show meaningful range
    plt.ylim(1e-10, 1.0)
    
    plt.savefig(f'residual_history_{matrix_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def test_suitesparse_matrices():
    """Test IDRstab with matrices from SuiteSparse"""
    # Test matrices (small to medium size)
    matrices = [
        "HB/bcsstk01",  # 48x48 structural problem
        "HB/bcsstk03",  # 112x112 structural problem
        "HB/bcsstk05"   # 153x153 structural problem
    ]
    
    # IDRstab parameters
    s = 4  # Shadow space dimension
    ell = 2  # Stabilization frequency
    
    for matrix_id in matrices:
        print(f"\nTesting with matrix: {matrix_id}")
        try:
            A, b = download_suitesparse_matrix(matrix_id)
            
            # Apply diagonal scaling
            A_scaled, b_scaled, D = diagonal_scaling(A, b)
            
            # Solve scaled system
            x_scaled, iter_count, rel_residual, residual_history = idrstab(
                A_scaled, b_scaled, s=s, ell=ell
            )
            
            # Scale back the solution
            x = x_scaled * D
            
            print(f"Matrix size: {A.shape}")
            print(f"IDRstab parameters: s={s}, ell={ell}")
            print(f"Number of iterations: {iter_count}")
            print(f"Final relative residual: {rel_residual:.2e}")
            
            # Plot residual history
            matrix_name = matrix_id.replace('/', '_')
            plot_residual_history(residual_history, matrix_name)
            print(f"Residual history plot saved as residual_history_{matrix_name}.png")
            
        except Exception as e:
            print(f"Error processing matrix {matrix_id}: {str(e)}")

if __name__ == "__main__":
    test_suitesparse_matrices()
