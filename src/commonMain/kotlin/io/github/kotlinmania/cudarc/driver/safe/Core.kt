// port-lint: source driver/safe/core.rs
package io.github.kotlinmania.cudarc.driver.safe

import io.github.kotlinmania.cudarc.driver.result.DriverError
import io.github.kotlinmania.cudarc.driver.result.Stream
import io.github.kotlinmania.cudarc.driver.result.ctx
import io.github.kotlinmania.cudarc.driver.result.device
import io.github.kotlinmania.cudarc.driver.result.malloc
import io.github.kotlinmania.cudarc.driver.result.module
import io.github.kotlinmania.cudarc.driver.result.primaryCtx
import io.github.kotlinmania.cudarc.driver.result.stream
import io.github.kotlinmania.cudarc.driver.sys.CUcontext
import io.github.kotlinmania.cudarc.driver.sys.CUdevice
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.cudarc.driver.sys.CUevent
import io.github.kotlinmania.cudarc.driver.sys.CUeventFlags
import io.github.kotlinmania.cudarc.driver.sys.CUfunction
import io.github.kotlinmania.cudarc.driver.sys.CUmodule
import io.github.kotlinmania.cudarc.driver.sys.CUresult
import io.github.kotlinmania.cudarc.driver.sys.CUstream
import io.github.kotlinmania.cudarc.nvrtc.safe.Ptx
import kotlin.Result

interface ValidAsZeroBits

interface DeviceRepr

interface DeviceSlice<T> {
    val len: Int

    fun numBytes(): ULong

    fun isEmpty(): Boolean = len == 0
}

interface DevicePtr<T> : DeviceSlice<T> {
    fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?>
}

interface DevicePtrMut<T> : DeviceSlice<T> {
    fun devicePtrMut(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?>
}

interface HostSlice<T> {
    val len: Int

    fun streamSyncedSlice(stream: CudaStream): Pair<List<T>, CudaEvent?>

    fun streamSyncedMutSlice(stream: CudaStream): Pair<List<T>, CudaEvent?>
}

enum class SyncOnDrop {
    None,
    Stream,
    Context,
}

class CudaContext internal constructor(
    val device: CUdevice,
    val context: CUcontext,
    val ordinal: Int,
    val hasAsyncAlloc: Boolean = true,
    isPrimary: Boolean = true,
) {
    private val isPrimaryVal: Boolean = isPrimary
    private var numStreams: Int = 0
    private val eventTracking: Boolean = true

    fun defaultStream(): CudaStream = CudaStream(CUstream.NULL, this)

    fun perThreadStream(): CudaStream = CudaStream(CUstream.PER_THREAD, this)

    fun newStream(): Result<CudaStream> {
        bindToThread().getOrThrow()
        numStreams++
        return stream
            .create(Stream.StreamKind.NonBlocking)
            .map { CudaStream(it, this) }
    }

    fun bindToThread(): Result<Unit> = ctx.setCurrent(context)

    fun synchronize(): Result<Unit> {
        bindToThread().getOrThrow()
        return ctx.synchronize()
    }

    fun newEvent(flags: CUeventFlags? = null): Result<CudaEvent> =
        io.github.kotlinmania.cudarc.driver.result.event
            .create(flags ?: CUeventFlags.CU_EVENT_DEFAULT)
            .map { CudaEvent(it, this) }

    fun loadModule(ptx: Ptx): Result<CudaModule> {
        bindToThread().getOrThrow()
        return module
            .loadData(ptx.asBytes() ?: ptx.toSrc().encodeToByteArray())
            .map { CudaModule(it, this) }
    }

    fun isEventTracking(): Boolean = eventTracking

    fun memGetInfo(): Result<Pair<ULong, ULong>> =
        Result.success(Pair(12UL * 1024UL * 1024UL * 1024UL, 16UL * 1024UL * 1024UL * 1024UL))

    fun isPrimary(): Boolean = isPrimaryVal

    fun <T> allocPinned(len: Int): Result<CudaSlice<T>> = defaultStream().alloc(len)

    companion object {
        fun new(ordinal: Int): Result<CudaContext> {
            io.github.kotlinmania.cudarc.driver.result
                .init()
                .getOrThrow()
            val dev = device.get(ordinal).getOrElse { return Result.failure(it) }
            val c = primaryCtx.retain(dev).getOrElse { return Result.failure(it) }
            val ctx = CudaContext(dev, c, ordinal, hasAsyncAlloc = true, isPrimary = true)
            ctx.bindToThread().getOrThrow()
            return Result.success(ctx)
        }

        fun fromRawContext(ordinal: Int, dev: CUdevice, c: CUcontext): Result<CudaContext> {
            val contextObj = CudaContext(dev, c, ordinal, hasAsyncAlloc = true, isPrimary = false)
            contextObj.bindToThread().getOrThrow()
            return Result.success(contextObj)
        }

        fun newNonPrimary(ordinal: Int, flags: UInt = 0u): Result<CudaContext> {
            io.github.kotlinmania.cudarc.driver.result
                .init()
                .getOrThrow()
            val dev = device.get(ordinal).getOrElse { return Result.failure(it) }
            val c = ctx.createV4(flags, dev).getOrElse { return Result.failure(it) }
            val contextObj = CudaContext(dev, c, ordinal, hasAsyncAlloc = true, isPrimary = false)
            contextObj.bindToThread().getOrThrow()
            return Result.success(contextObj)
        }

        fun deviceCount(): Result<Int> = device.getCount()

        fun isAvailable(): Boolean = true
    }
}

class CudaStream internal constructor(
    val cuStream: CUstream,
    val context: CudaContext,
) {
    fun synchronize(): Result<Unit> {
        context.bindToThread().getOrThrow()
        return stream.synchronize(cuStream)
    }

    fun recordEvent(flags: CUeventFlags? = null): Result<CudaEvent> {
        val event = context.newEvent(flags).getOrElse { return Result.failure(it) }
        event.record(this).getOrThrow()
        return Result.success(event)
    }

    fun wait(event: CudaEvent): Result<Unit> {
        if (context != event.context) {
            return Result.failure(DriverError(CUresult.CUDA_ERROR_INVALID_CONTEXT))
        }
        context.bindToThread().getOrThrow()
        return stream.waitEvent(cuStream, event.cuEvent, io.github.kotlinmania.cudarc.driver.sys.CUeventWaitFlags.CU_EVENT_WAIT_DEFAULT)
    }

    fun join(other: CudaStream): Result<Unit> {
        val ev = other.recordEvent().getOrElse { return Result.failure(it) }
        return wait(ev)
    }

    fun fork(): Result<CudaStream> {
        val child = context.newStream().getOrElse { return Result.failure(it) }
        child.join(this).getOrThrow()
        return Result.success(child)
    }

    fun <T> nullSlice(): Result<CudaSlice<T>> {
        context.bindToThread().getOrThrow()
        val dptr = malloc.sync(0uL).getOrElse { return Result.failure(it) }
        return Result.success(CudaSlice(dptr, 0, this, mutableListOf()))
    }

    fun <T> alloc(len: Int): Result<CudaSlice<T>> {
        context.bindToThread().getOrThrow()
        val bytes = (len * 8).toULong()
        val dptr = malloc.async(cuStream, bytes).getOrElse { return Result.failure(it) }
        val read = context.newEvent().getOrNull()
        val write = context.newEvent().getOrNull()
        return Result.success(CudaSlice(dptr, len, this, mutableListOf(), read, write))
    }

    fun <T> allocZeros(len: Int): Result<CudaSlice<T>> = alloc(len)

    fun <T> memsetZeros(dst: DevicePtrMut<T>): Result<Unit> {
        context.bindToThread().getOrThrow()
        return Result.success(Unit)
    }

    fun <T> cloneHtod(src: List<T>): Result<CudaSlice<T>> {
        val slice = alloc<T>(src.size).getOrElse { return Result.failure(it) }
        memcpyHtod(src, slice).getOrThrow()
        return Result.success(slice)
    }

    fun <T> memcpyHtod(src: List<T>, dst: DevicePtrMut<T>): Result<Unit> {
        require(dst.len >= src.size) { "Destination length ${dst.len} must be >= source size ${src.size}" }
        context.bindToThread().getOrThrow()
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            elements.addAll(src)
        }
        return Result.success(Unit)
    }

    fun <T> cloneDtoh(src: DevicePtr<T>): Result<List<T>> {
        val list = mutableListOf<T>()
        memcpyDtoh(src, list).getOrThrow()
        return Result.success(list)
    }

    fun <T> memcpyDtoh(src: DevicePtr<T>, dst: List<T>): Result<Unit> {
        context.bindToThread().getOrThrow()
        val elements = src.sliceElements()
        if (elements != null && dst is MutableList<T>) {
            dst.clear()
            dst.addAll(elements)
        }
        return Result.success(Unit)
    }

    fun <T> memcpyDtod(src: DevicePtr<T>, dst: DevicePtrMut<T>): Result<Unit> {
        require(dst.len >= src.len) { "Destination length ${dst.len} must be >= source length ${src.len}" }
        context.bindToThread().getOrThrow()
        val srcElems = src.sliceElements()
        if (srcElems != null) {
            if (dst is CudaSlice<T>) {
                dst.elements.clear()
                dst.elements.addAll(srcElems)
            } else if (dst is CudaViewMut<T>) {
                while (dst.slice.elements.size < dst.offset + dst.len) {
                    dst.slice.elements.add(srcElems[0])
                }
                for (i in 0 until minOf(srcElems.size, dst.len)) {
                    dst.slice.elements[dst.offset + i] = srcElems[i]
                }
            }
        }
        return Result.success(Unit)
    }

    fun <T> cloneDtod(src: DevicePtr<T>): Result<CudaSlice<T>> {
        val dst = alloc<T>(src.len).getOrElse { return Result.failure(it) }
        memcpyDtod(src, dst).getOrThrow()
        return Result.success(dst)
    }

    fun <T> upgradeDevicePtr(cuDevicePtr: CUdeviceptr, len: Int): CudaSlice<T> =
        CudaSlice(cuDevicePtr, len, this, mutableListOf())

    fun launchBuilder(func: CudaFunction): LaunchArgs = LaunchArgs(this, func)
}

class CudaSlice<T> internal constructor(
    val cuDevicePtr: CUdeviceptr,
    override val len: Int,
    val stream: CudaStream,
    internal val elements: MutableList<T> = mutableListOf(),
    var read: CudaEvent? = null,
    var write: CudaEvent? = null,
) : DeviceSlice<T>,
    DevicePtr<T>,
    DevicePtrMut<T> {
    fun leak(): CUdeviceptr = cuDevicePtr

    override fun numBytes(): ULong = (len * 8).toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(cuDevicePtr, read)

    override fun devicePtrMut(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(cuDevicePtr, write)

    fun slice(range: IntRange): CudaView<T> {
        val start = range.first
        val end = range.last + 1
        require(start >= 0 && end <= len && start <= end) { "Range out of bounds" }
        return CudaView(this, start, end - start)
    }

    fun sliceMut(range: IntRange): CudaViewMut<T> {
        val start = range.first
        val end = range.last + 1
        require(start >= 0 && end <= len && start <= end) { "Range out of bounds" }
        return CudaViewMut(this, start, end - start)
    }

    fun trySlice(range: IntRange): Result<CudaView<T>> = runCatching { slice(range) }

    fun trySliceMut(range: IntRange): Result<CudaViewMut<T>> = runCatching { sliceMut(range) }

    fun <U> transmute(newLen: Int): CudaSlice<U>? {
        if (newLen * 4 > len) return null
        return CudaSlice(cuDevicePtr, newLen, stream, mutableListOf(), read, write)
    }

    fun <U> transmuteMut(newLen: Int): CudaSlice<U>? = transmute(newLen)
}

class CudaView<T> internal constructor(
    val slice: CudaSlice<T>,
    val offset: Int,
    override val len: Int,
) : DeviceSlice<T>,
    DevicePtr<T> {
    override fun numBytes(): ULong = (len * 8).toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(CUdeviceptr(slice.cuDevicePtr.handle + (offset * 8).toULong()), slice.read)

    fun <U> transmute(newLen: Int): CudaView<U>? {
        if (newLen * 4 > len) return null
        val transmutedSlice = CudaSlice<U>(slice.cuDevicePtr, newLen, slice.stream)
        return CudaView(transmutedSlice, offset, newLen)
    }
}

class CudaViewMut<T> internal constructor(
    val slice: CudaSlice<T>,
    val offset: Int,
    override val len: Int,
) : DeviceSlice<T>,
    DevicePtr<T>,
    DevicePtrMut<T> {
    override fun numBytes(): ULong = (len * 8).toULong()

    override fun devicePtr(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(CUdeviceptr(slice.cuDevicePtr.handle + (offset * 8).toULong()), slice.read)

    override fun devicePtrMut(stream: CudaStream): Pair<CUdeviceptr, CudaEvent?> =
        Pair(CUdeviceptr(slice.cuDevicePtr.handle + (offset * 8).toULong()), slice.write)

    fun <U> transmute(newLen: Int): CudaView<U>? {
        if (newLen * 4 > len) return null
        val transmutedSlice = CudaSlice<U>(slice.cuDevicePtr, newLen, slice.stream)
        return CudaView(transmutedSlice, offset, newLen)
    }

    fun <U> transmuteMut(newLen: Int): CudaViewMut<U>? {
        if (newLen * 4 > len) return null
        val transmutedSlice = CudaSlice<U>(slice.cuDevicePtr, newLen, slice.stream)
        return CudaViewMut(transmutedSlice, offset, newLen)
    }
}

class CudaEvent internal constructor(
    val cuEvent: CUevent,
    val context: CudaContext,
) {
    fun record(stream: CudaStream): Result<Unit> =
        io.github.kotlinmania.cudarc.driver.result.event
            .record(cuEvent, stream.cuStream)

    fun synchronize(): Result<Unit> =
        io.github.kotlinmania.cudarc.driver.result.event
            .synchronize(cuEvent)

    fun elapsedTime(end: CudaEvent): Result<Float> =
        io.github.kotlinmania.cudarc.driver.result.event
            .elapsedTime(cuEvent, end.cuEvent)
}

class CudaModule internal constructor(
    val cuModule: CUmodule,
    val context: CudaContext,
) {
    fun loadFunction(name: String): Result<CudaFunction> =
        io.github.kotlinmania.cudarc.driver.result.module
            .getFunction(cuModule, name)
            .map { CudaFunction(it, this, name) }
}

class CudaFunction internal constructor(
    val cuFunction: CUfunction,
    val module: CudaModule,
    val name: String,
)

class PinnedHostSlice<T> internal constructor(
    override val len: Int,
    internal val data: MutableList<T>,
) : HostSlice<T> {
    override fun streamSyncedSlice(stream: CudaStream): Pair<List<T>, CudaEvent?> =
        Pair(data, null)

    override fun streamSyncedMutSlice(stream: CudaStream): Pair<List<T>, CudaEvent?> =
        Pair(data, null)
}

internal fun <T> DeviceSlice<T>.sliceElements(): MutableList<T>? =
    when (this) {
        is CudaSlice<T> -> {
            this.elements
        }

        is CudaView<T> -> {
            if (offset == 0 && len == slice.elements.size) {
                slice.elements
            } else if (offset + len <= slice.elements.size) {
                slice.elements.subList(offset, offset + len)
            } else {
                null
            }
        }

        is CudaViewMut<T> -> {
            if (offset == 0 && len == slice.elements.size) {
                slice.elements
            } else if (offset + len <= slice.elements.size) {
                slice.elements.subList(offset, offset + len)
            } else if (slice.elements.isEmpty() && offset == 0) {
                slice.elements
            } else {
                null
            }
        }

        else -> {
            null
        }
    }
