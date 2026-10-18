package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasHandle
import io.github.kotlinmania.cudarc.cublas.result.CublasPointerMode
import io.github.kotlinmania.cudarc.cublas.result.createHandle
import io.github.kotlinmania.cudarc.cublas.result.destroyHandle
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import kotlin.Result
import io.github.kotlinmania.cudarc.cublas.result.getPointerMode as resultGetPointerMode
import io.github.kotlinmania.cudarc.cublas.result.setPointerMode as resultSetPointerMode
import io.github.kotlinmania.cudarc.cublas.result.setStream as resultSetStream

class CudaBlas internal constructor(
    val handle: CublasHandle,
    var stream: CudaStream,
) {
    companion object {
        fun new(stream: CudaStream): Result<CudaBlas> {
            val handle = createHandle().getOrElse { return Result.failure(it) }
            resultSetStream(handle, stream.cuStream.handle).getOrThrow()
            return Result.success(CudaBlas(handle, stream))
        }
    }

    fun setStream(newStream: CudaStream): Result<Unit> {
        stream = newStream
        return resultSetStream(handle, newStream.cuStream.handle)
    }

    fun setPointerMode(mode: CublasPointerMode): Result<Unit> =
        resultSetPointerMode(handle, mode)

    fun getPointerMode(): Result<CublasPointerMode> =
        resultGetPointerMode(handle)

    fun close() {
        destroyHandle(handle)
    }
}
