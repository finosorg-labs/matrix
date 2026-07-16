/**
 * @file vector_ops.h
 * @brief Vector operations for double-precision floating-point vectors
 *
 * Provides high-performance vector operations including:
 * - Dot product
 * - L1 and L2 norms
 * - L1 and L2 distances
 *
 * All operations are thread-safe and SIMD-accelerated where available.
 */

#ifndef FC_VECTOR_OPS_H
#define FC_VECTOR_OPS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Compute dot product of two double-precision vectors
 *
 * Computes: result = sum(x[i] * y[i]) for i in [0, n)
 *
 * Time complexity: O(n)
 * Space complexity: O(1)
 * Thread safety: Thread-safe (no shared state)
 *
 * @param[in]  x      First vector (length n)
 * @param[in]  y      Second vector (length n)
 * @param[in]  n      Vector length (must be > 0)
 * @param[out] result Dot product result
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_vec_dot_f64(const double* x, const double* y, int64_t n, double* result);

/**
 * @brief Compute L2 norm (Euclidean norm) of a double-precision vector
 *
 * Computes: result = sqrt(sum(x[i]^2))
 *
 * Time complexity: O(n)
 * Space complexity: O(1)
 * Thread safety: Thread-safe
 *
 * @param[in]  x      Input vector (length n)
 * @param[in]  n      Vector length (must be > 0)
 * @param[out] result L2 norm result
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_vec_norm_l2_f64(const double* x, int64_t n, double* result);

/**
 * @brief Compute L1 norm (Manhattan norm) of a double-precision vector
 *
 * Computes: result = sum(|x[i]|)
 *
 * Time complexity: O(n)
 * Space complexity: O(1)
 * Thread safety: Thread-safe
 *
 * @param[in]  x      Input vector (length n)
 * @param[in]  n      Vector length (must be > 0)
 * @param[out] result L1 norm result
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_vec_norm_l1_f64(const double* x, int64_t n, double* result);

/**
 * @brief Compute Euclidean distance between two double-precision vectors
 *
 * Computes: result = sqrt(sum((x[i] - y[i])^2))
 *
 * Time complexity: O(n)
 * Space complexity: O(1)
 * Thread safety: Thread-safe
 *
 * @param[in]  x      First vector (length n)
 * @param[in]  y      Second vector (length n)
 * @param[in]  n      Vector length (must be > 0)
 * @param[out] result Euclidean distance result
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_vec_distance_l2_f64(const double* x, const double* y, int64_t n, double* result);

/**
 * @brief Compute Manhattan distance between two double-precision vectors
 *
 * Computes: result = sum(|x[i] - y[i]|)
 *
 * Time complexity: O(n)
 * Space complexity: O(1)
 * Thread safety: Thread-safe
 *
 * @param[in]  x      First vector (length n)
 * @param[in]  y      Second vector (length n)
 * @param[in]  n      Vector length (must be > 0)
 * @param[out] result Manhattan distance result
 *
 * @return FC_OK on success, FC_ERR_INVALID_ARG if inputs invalid
 */
int fc_vec_distance_l1_f64(const double* x, const double* y, int64_t n, double* result);

#ifdef __cplusplus
}
#endif

#endif /* FC_VECTOR_OPS_H */
