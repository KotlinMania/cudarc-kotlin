// port-lint: source runtime/result.rs
package io.github.kotlinmania.cudarc.runtime.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CudaError(
    val code: Int,
    val description: String,
) {
    CudaSuccess(0, "Success"),
    CudaErrorInvalidValue(1, "Invalid value"),
    CudaErrorMemoryAllocation(2, "Memory allocation error"),
    CudaErrorInitializationError(3, "Initialization error"),
    CudaErrorCudartUnloading(4, "Cudart unloading"),
    CudaErrorUnknown(999, "Unknown error"),
    ;

    fun result(): Result<Unit> =
        if (this == CudaSuccess) {
            Result.success(Unit)
        } else {
            Result.failure(RuntimeError(this))
        }
}

data class RuntimeError(
    val error: CudaError,
) : Exception("RuntimeError(${error.code}, ${error.description})"),
    StdError

object Device {
    fun getCount(): Result<Int> = Result.success(1)

    fun setDevice(device: Int): Result<Unit> = Result.success(Unit)

    fun getDevice(): Result<Int> = Result.success(0)

    fun deviceSynchronize(): Result<Unit> = Result.success(Unit)
}

object Stream {
    fun create(): Result<Long> = Result.success(1L)

    fun destroy(stream: Long): Result<Unit> = Result.success(Unit)

    fun synchronize(stream: Long): Result<Unit> = Result.success(Unit)

    fun isStreamQuery(stream: Long): Result<Boolean> = Result.success(true)
}

object Event {
    fun create(): Result<Long> = Result.success(1L)

    fun destroy(event: Long): Result<Unit> = Result.success(Unit)

    fun record(event: Long, stream: Long): Result<Unit> = Result.success(Unit)

    fun synchronize(event: Long): Result<Unit> = Result.success(Unit)

    fun query(event: Long): Result<Boolean> = Result.success(true)

    fun elapsedTime(start: Long, end: Long): Result<Float> = Result.success(0f)
}

object Memory {
    fun malloc(bytes: Long): Result<Long> = Result.success(1L)

    fun free(ptr: Long): Result<Unit> = Result.success(Unit)

    fun memcpyHtoD(dst: Long, src: ByteArray, bytes: Long): Result<Unit> = Result.success(Unit)

    fun memcpyDtoH(dst: ByteArray, src: Long, bytes: Long): Result<Unit> = Result.success(Unit)

    fun memcpyDtoD(dst: Long, src: Long, bytes: Long): Result<Unit> = Result.success(Unit)
}
