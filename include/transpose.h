/**
 * @file transpose.h
 * @brief Matrix transpose operations
 *
 * Provides cache-friendly matrix transposition.
 */

#ifndef FC_TRANSPOSE_H
#define FC_TRANSPOSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Transpose a double-precision matrix
 *
 * Computes: dst = src^T
 * where src is rows×cols, dst is cols×rows
 *
 * Uses cache-friendly blocking for large matrices.
 *
 * Time complexity: O(rows * cols)
 * Space complexity: O(1)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]  rows   Number of rows in source matrix
 * @param[in]  cols   Number of columns in source matrix
 * @param[in]  src    Source matrix (rows×cols, row-major)
 * @param[in]  ld_src Leading dimension of source (stride between rows)
 * @param[out] dst    Destination matrix (cols×rows, row-major)
 * @param[in]  ld_dst Leading dimension of destination (stride between rows)
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_mat_transpose_f64(
    int64_t rows,
    int64_t cols,
    const double* src,
    int64_t ld_src,
    double* dst,
    int64_t ld_dst
);

#ifdef __cplusplus
}
#endif

#endif /* FC_TRANSPOSE_H */
