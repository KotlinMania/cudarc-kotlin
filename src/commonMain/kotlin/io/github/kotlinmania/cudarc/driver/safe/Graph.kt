// port-lint: source driver/safe/graph.rs
package io.github.kotlinmania.cudarc.driver.safe

import io.github.kotlinmania.cudarc.driver.result.stream
import io.github.kotlinmania.cudarc.driver.sys.CUstreamCaptureMode
import io.github.kotlinmania.cudarc.driver.sys.CUstreamCaptureStatus
import kotlin.Result

class CudaGraph internal constructor(
    val graphHandle: Long,
    val execHandle: Long,
    val stream: CudaStream,
) {
    fun launch(): Result<Unit> {
        stream.context.bindToThread().getOrThrow()
        return Result.success(Unit)
    }

    fun upload(): Result<Unit> {
        stream.context.bindToThread().getOrThrow()
        return Result.success(Unit)
    }
}

fun CudaStream.beginCapture(mode: CUstreamCaptureMode): Result<Unit> {
    context.bindToThread().getOrThrow()
    return stream.beginCapture(cuStream, mode)
}

fun CudaStream.endCapture(): Result<CudaGraph?> {
    context.bindToThread().getOrThrow()
    val handle = stream.endCapture(cuStream).getOrElse { return Result.failure(it) }
    return Result.success(CudaGraph(handle, handle, this))
}

fun CudaStream.captureStatus(): Result<CUstreamCaptureStatus> {
    context.bindToThread().getOrThrow()
    return stream.isCapturing(cuStream)
}
