// port-lint: source cusolver/safe.rs
package io.github.kotlinmania.cudarc.cusolver.safe

import io.github.kotlinmania.cudarc.cusolver.result.CusolverDnHandle
import io.github.kotlinmania.cudarc.cusolver.result.CusolverDnParams
import io.github.kotlinmania.cudarc.cusolver.result.dnCreate
import io.github.kotlinmania.cudarc.cusolver.result.dnCreateParams
import io.github.kotlinmania.cudarc.cusolver.result.dnDestroy
import io.github.kotlinmania.cudarc.cusolver.result.dnDestroyParams
import io.github.kotlinmania.cudarc.cusolver.result.dnSetStream
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import kotlin.Result

class DnHandle(
    val handle: CusolverDnHandle,
    private var stream: CudaStream,
) {
    companion object {
        fun new(stream: CudaStream): Result<DnHandle> {
            val h = dnCreate().getOrElse { return Result.failure(it) }
            val setRes = dnSetStream(h, stream.cuStream.handle)
            if (setRes.isFailure) {
                return Result.failure(setRes.exceptionOrNull() ?: Exception("Failed to set stream"))
            }
            return Result.success(DnHandle(h, stream))
        }
    }

    fun cu(): CusolverDnHandle = handle

    fun stream(): CudaStream = stream

    fun setStream(stream: CudaStream): Result<Unit> {
        this.stream = stream
        return dnSetStream(handle, stream.cuStream.handle)
    }

    fun close() {
        dnDestroy(handle)
    }
}

class DnParams(
    val params: CusolverDnParams,
) {
    companion object {
        fun new(): Result<DnParams> =
            dnCreateParams().map { DnParams(it) }
    }

    fun close() {
        dnDestroyParams(params)
    }
}
