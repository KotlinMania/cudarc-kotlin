// port-lint: source cublas/result.rs
package io.github.kotlinmania.cudarc.cublas.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CublasStatus(val code: Int, val description: String) {
    CUBLAS_STATUS_SUCCESS(0, "Success"),
    CUBLAS_STATUS_NOT_INITIALIZED(1, "Not initialized"),
    CUBLAS_STATUS_ALLOC_FAILED(3, "Alloc failed"),
    CUBLAS_STATUS_INVALID_VALUE(7, "Invalid value"),
    CUBLAS_STATUS_ARCH_MISMATCH(8, "Architecture mismatch"),
    CUBLAS_STATUS_MAPPING_ERROR(11, "Mapping error"),
    CUBLAS_STATUS_EXECUTION_FAILED(13, "Execution failed"),
    CUBLAS_STATUS_INTERNAL_ERROR(14, "Internal error"),
    CUBLAS_STATUS_NOT_SUPPORTED(15, "Not supported"),
    CUBLAS_STATUS_LICENSE_ERROR(16, "License error"),
    ;

    fun result(): Result<Unit> =
        if (this == CUBLAS_STATUS_SUCCESS) {
            Result.success(Unit)
        } else {
            Result.failure(CublasError(this))
        }

    companion object {
        fun fromCode(code: Int): CublasStatus =
            entries.find { it.code == code } ?: CUBLAS_STATUS_INTERNAL_ERROR
    }
}

enum class CublasOperation(val code: Int) {
    CUBLAS_OP_N(0),
    CUBLAS_OP_T(1),
    CUBLAS_OP_C(2),
}

enum class CublasPointerMode(val code: Int) {
    CUBLAS_POINTER_MODE_HOST(0),
    CUBLAS_POINTER_MODE_DEVICE(1),
}

data class CublasHandle(val handle: Long)

data class CublasError(
    val status: CublasStatus,
) : Exception("CublasError(${status.code}, ${status.name})"), StdError

private var nextHandleId: Long = 1L
private val pointerModes: MutableMap<Long, CublasPointerMode> = mutableMapOf()

fun createHandle(): Result<CublasHandle> {
    val id = nextHandleId++
    pointerModes[id] = CublasPointerMode.CUBLAS_POINTER_MODE_HOST
    return Result.success(CublasHandle(id))
}

fun destroyHandle(handle: CublasHandle): Result<Unit> {
    pointerModes.remove(handle.handle)
    return Result.success(Unit)
}

fun setStream(handle: CublasHandle, streamHandle: Long): Result<Unit> = Result.success(Unit)

fun setPointerMode(handle: CublasHandle, mode: CublasPointerMode): Result<Unit> {
    pointerModes[handle.handle] = mode
    return Result.success(Unit)
}

fun getPointerMode(handle: CublasHandle): Result<CublasPointerMode> =
    Result.success(pointerModes[handle.handle] ?: CublasPointerMode.CUBLAS_POINTER_MODE_HOST)
