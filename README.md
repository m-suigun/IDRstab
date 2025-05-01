# IDRstab Solver

A Python implementation of the IDRstab (Induced Dimension Reduction with Stabilization) method for solving large sparse linear systems.

## Features

- Implementation of IDRstab method with configurable parameters
- Support for sparse matrices in CRS (Compressed Row Storage) format
- Diagonal scaling preconditioning
- Residual history plotting
- Test cases using matrices from SuiteSparse Matrix Collection

## Requirements

- Python 3.6+
- NumPy
- SciPy
- Matplotlib
- Requests

## Installation

```bash
pip install numpy scipy matplotlib requests
```

## Usage

```python
from IDRstab import CRSMatrix, idrstab, diagonal_scaling

# Create or load your matrix in CRS format
A = CRSMatrix(values, col_indices, row_ptr, shape)
b = right_hand_side_vector

# Apply diagonal scaling
A_scaled, b_scaled, D = diagonal_scaling(A, b)

# Solve the system with IDRstab
x_scaled, iter_count, rel_residual, residual_history = idrstab(
    A_scaled, b_scaled, 
    tol=1e-8,           # Convergence tolerance
    max_iter=1000,      # Maximum iterations
    s=4,                # Shadow space dimension
    ell=2               # Stabilization frequency
)

# Scale back the solution
x = x_scaled * D
```

## Test Results

The implementation has been tested with matrices from the SuiteSparse Matrix Collection:

- bcsstk01 (48x48): 71 iterations, final residual 9.62e-09
- bcsstk03 (112x112): 930 iterations, final residual 9.91e-09
- bcsstk05 (153x153): 408 iterations, final residual 9.67e-09

## License

MIT License 