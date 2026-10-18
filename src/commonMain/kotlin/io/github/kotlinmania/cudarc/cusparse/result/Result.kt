// port-lint: source cusparse/result.rs
package io.github.kotlinmania.cudarc.cusparse.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

data class CusparseHandle(val handle: Long)

enum class CusparseStatus(val code: Int) {
    CUSPARSE_STATUS_SUCCESS(0),
    CUSPARSE_STATUS_NOT_INITIALIZED(1),
    CUSPARSE_STATUS_ALLOC_FAILED(2),
    CUSPARSE_STATUS_INVALID_VALUE(3),
    CUSPARSE_STATUS_ARCH_MISMATCH(4),
    CUSPARSE_STATUS_MAPPING_ERROR(5),
    CUSPARSE_STATUS_EXECUTION_FAILED(6),
    CUSPARSE_STATUS_INTERNAL_ERROR(7),
    CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED(8),
    CUSPARSE_STATUS_ZERO_PIVOT(9),
}

data class CusparseError(
    val status: CusparseStatus,
) : Exception("CusparseError(${status.code}, ${status.name})"), StdError

private var nextCusparseId: Long = 1L

fun create(): Result<CusparseHandle> =
    Result.success(CusparseHandle(nextCusparseId++))

fun destroy(handle: CusparseHandle): Result<Unit> =
    Result.success(Unit)
