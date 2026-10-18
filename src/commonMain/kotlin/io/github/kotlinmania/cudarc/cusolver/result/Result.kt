// port-lint: source cusolver/result.rs
package io.github.kotlinmania.cudarc.cusolver.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CusolverStatus(val code: Int) {
    CUSOLVER_STATUS_SUCCESS(0),
    CUSOLVER_STATUS_NOT_INITIALIZED(1),
    CUSOLVER_STATUS_ALLOC_FAILED(2),
    CUSOLVER_STATUS_INVALID_VALUE(3),
    CUSOLVER_STATUS_ARCH_MISMATCH(4),
    CUSOLVER_STATUS_MAPPING_ERROR(5),
    CUSOLVER_STATUS_EXECUTION_FAILED(6),
    CUSOLVER_STATUS_INTERNAL_ERROR(7),
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED(8),
    CUSOLVER_STATUS_NOT_SUPPORTED(9),
    CUSOLVER_STATUS_ZERO_PIVOT(10),
    CUSOLVER_STATUS_INVALID_LICENSE(11),
    ;

    fun result(): Result<Unit> =
        if (this == CUSOLVER_STATUS_SUCCESS) {
            Result.success(Unit)
        } else {
            Result.failure(CusolverError(this))
        }
}

data class CusolverError(
    val status: CusolverStatus,
) : Exception("CusolverError(${status.code}, ${status.name})"), StdError

data class CusolverDnHandle(val handle: Long)
data class CusolverDnParams(val handle: Long)
data class CusolverSpHandle(val handle: Long)
data class CusolverRfHandle(val handle: Long)

private var nextCusolverId: Long = 1L

fun dnCreate(): Result<CusolverDnHandle> =
    Result.success(CusolverDnHandle(nextCusolverId++))

fun dnDestroy(handle: CusolverDnHandle): Result<Unit> =
    Result.success(Unit)

fun dnSetStream(handle: CusolverDnHandle, stream: Long): Result<Unit> =
    Result.success(Unit)

fun dnCreateParams(): Result<CusolverDnParams> =
    Result.success(CusolverDnParams(nextCusolverId++))

fun dnDestroyParams(params: CusolverDnParams): Result<Unit> =
    Result.success(Unit)

fun spCreate(): Result<CusolverSpHandle> =
    Result.success(CusolverSpHandle(nextCusolverId++))

fun spDestroy(handle: CusolverSpHandle): Result<Unit> =
    Result.success(Unit)

fun spSetStream(handle: CusolverSpHandle, stream: Long): Result<Unit> =
    Result.success(Unit)

fun rfCreate(): Result<CusolverRfHandle> =
    Result.success(CusolverRfHandle(nextCusolverId++))

fun rfDestroy(handle: CusolverRfHandle): Result<Unit> =
    Result.success(Unit)
