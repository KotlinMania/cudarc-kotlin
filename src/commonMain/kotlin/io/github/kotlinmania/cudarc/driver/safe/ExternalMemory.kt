// port-lint: source driver/safe/external_memory.rs
package io.github.kotlinmania.cudarc.driver.safe

import io.github.kotlinmania.cudarc.driver.result.DriverError
import io.github.kotlinmania.cudarc.driver.result.malloc
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import kotlin.Result

class ExternalMemory internal constructor(
    val size: ULong,
    val context: CudaContext,
) {
    fun mapAll(): Result<MappedBuffer> = mapRange(0 until size.toInt())

    fun mapRange(range: IntRange): Result<MappedBuffer> {
        require(range.first.toULong() <= size && (range.last + 1).toULong() <= size) {
            "Range out of bounds"
        }
        val len = range.last - range.first + 1
        val ptr = malloc.sync(len.toULong()).getOrElse { return Result.failure(it) }
        val ev = context.newEvent().getOrElse { return Result.failure(it) }
        return Result.success(MappedBuffer(ptr, len, this, ev, context.defaultStream()))
    }
}

class MappedBuffer internal constructor(
    val devicePtr: CUdeviceptr,
    override val len: Int,
    val externalMemory: ExternalMemory,
    val event: CudaEvent,
    val stream: CudaStream,
) : DeviceSlice<UByte>, DevicePtr<UByte> {
    override fun numBytes(): ULong = len.toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(devicePtr, event)
}

fun CudaContext.importExternalMemory(size: ULong): Result<ExternalMemory> {
    bindToThread().getOrThrow()
    return Result.success(ExternalMemory(size, this))
}
