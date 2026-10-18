// port-lint: source driver/safe/unified_memory.rs
package io.github.kotlinmania.cudarc.driver.safe

import io.github.kotlinmania.cudarc.driver.result.malloc
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.cudarc.driver.sys.CUmemAttachFlags
import kotlin.Result

class UnifiedSlice<T> internal constructor(
    val cuDevicePtr: CUdeviceptr,
    override val len: Int,
    val stream: CudaStream,
    val event: CudaEvent,
    var attachMode: CUmemAttachFlags = CUmemAttachFlags.CU_MEM_ATTACH_GLOBAL,
    internal val elements: MutableList<T> = mutableListOf(),
) : DeviceSlice<T>,
    DevicePtr<T>,
    DevicePtrMut<T>,
    HostSlice<T> {
    override fun numBytes(): ULong = (len * 8).toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(cuDevicePtr, event)

    override fun devicePtrMut(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(cuDevicePtr, event)

    override fun streamSyncedSlice(stream: CudaStream): Pair<List<T>, CudaEvent?> =
        Pair(elements, event)

    override fun streamSyncedMutSlice(stream: CudaStream): Pair<List<T>, CudaEvent?> =
        Pair(elements, event)

    fun asSlice(): List<T> = elements

    fun asMutSlice(): List<T> = elements

    fun slice(range: IntRange): UnifiedView<T> {
        val start = range.first
        val end = range.last + 1
        require(start >= 0 && end <= len && start <= end) { "Range out of bounds" }
        return UnifiedView(cuDevicePtr, end - start, event, stream, attachMode)
    }

    fun sliceMut(range: IntRange): UnifiedViewMut<T> {
        val start = range.first
        val end = range.last + 1
        require(start >= 0 && end <= len && start <= end) { "Range out of bounds" }
        return UnifiedViewMut(cuDevicePtr, end - start, event, stream, attachMode)
    }
}

class UnifiedView<T> internal constructor(
    val ptr: CUdeviceptr,
    override val len: Int,
    val event: CudaEvent,
    val stream: CudaStream,
    val attachMode: CUmemAttachFlags,
) : DeviceSlice<T>,
    DevicePtr<T> {
    override fun numBytes(): ULong = (len * 8).toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(ptr, event)
}

class UnifiedViewMut<T> internal constructor(
    val ptr: CUdeviceptr,
    override val len: Int,
    val event: CudaEvent,
    val stream: CudaStream,
    val attachMode: CUmemAttachFlags,
) : DeviceSlice<T>,
    DevicePtr<T>,
    DevicePtrMut<T> {
    override fun numBytes(): ULong = (len * 8).toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(ptr, event)

    override fun devicePtrMut(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(ptr, event)
}

fun <T> CudaContext.allocUnified(len: Int, attachGlobal: Boolean = true): Result<UnifiedSlice<T>> {
    bindToThread().getOrThrow()
    val bytes = (len * 8).toULong()
    val ptr = malloc.sync(bytes).getOrElse { return Result.failure(it) }
    val stream = defaultStream()
    val event = newEvent().getOrElse { return Result.failure(it) }
    val mode = if (attachGlobal) CUmemAttachFlags.CU_MEM_ATTACH_GLOBAL else CUmemAttachFlags.CU_MEM_ATTACH_HOST
    return Result.success(UnifiedSlice(ptr, len, stream, event, mode))
}
