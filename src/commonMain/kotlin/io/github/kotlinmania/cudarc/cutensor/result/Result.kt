// port-lint: source cutensor/result.rs
package io.github.kotlinmania.cudarc.cutensor.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CutensorStatus(val code: Int) {
    CUTENSOR_STATUS_SUCCESS(0),
    CUTENSOR_STATUS_NOT_INITIALIZED(1),
    CUTENSOR_STATUS_ALLOC_FAILED(2),
    CUTENSOR_STATUS_INVALID_VALUE(3),
    CUTENSOR_STATUS_ARCH_MISMATCH(4),
    CUTENSOR_STATUS_MAPPING_ERROR(5),
    CUTENSOR_STATUS_EXECUTION_FAILED(6),
    CUTENSOR_STATUS_INTERNAL_ERROR(7),
    CUTENSOR_STATUS_NOT_SUPPORTED(8),
    CUTENSOR_STATUS_LICENSE_ERROR(9),
    CUTENSOR_STATUS_CUBLAS_ERROR(10),
    CUTENSOR_STATUS_CUDA_ERROR(11),
    CUTENSOR_STATUS_INSUFFICIENT_WORKSPACE(12),
    CUTENSOR_STATUS_INSUFFICIENT_DRIVER(13),
    CUTENSOR_STATUS_IO_ERROR(14),
}

data class CutensorError(
    val status: CutensorStatus,
) : Exception("CutensorError(${status.code}, ${status.name})"), StdError

data class CutensorHandle(val handle: Long)

data class CutensorVersion(val major: Int, val minor: Int, val patch: Int)

private var nextCutensorId: Long = 1L

fun createHandle(): Result<CutensorHandle> =
    Result.success(CutensorHandle(nextCutensorId++))

fun destroyHandle(handle: CutensorHandle): Result<Unit> =
    Result.success(Unit)

fun getVersion(): CutensorVersion =
    CutensorVersion(1, 5, 0)
