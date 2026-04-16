void sparse_multiply(
    int rows, int cols,
    const double* A,
    const double* x,
    int* out_nnz,
    double* values,
    int* col_indices,
    int* row_ptrs,
    double* y
)
{
    int nnz = 0;

    // first row always starts at index 0
    row_ptrs[0] = 0;

    // ============================
    // 1. Build CSR representation
    // ============================
    for (int i = 0; i < rows; i++) {

        for (int j = 0; j < cols; j++) {

            double val = A[i * cols + j];

            if (val != 0.0) {
                values[nnz] = val;
                col_indices[nnz] = j;
                nnz++;
            }
        }

        row_ptrs[i + 1] = nnz;
    }

    *out_nnz = nnz;

    // ============================
    // 2. Sparse Matrix-Vector Multiply
    // y = A * x
    // ============================
    for (int i = 0; i < rows; i++) {

        double sum = 0.0;

        for (int k = row_ptrs[i]; k < row_ptrs[i + 1]; k++) {
            sum += values[k] * x[col_indices[k]];
        }

        y[i] = sum;
    }
}
