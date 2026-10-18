// port-lint: source driver/result.rs
package io.github.kotlinmania.cudarc.driver.result

import io.github.kotlinmania.cudarc.driver.sys.CUcontext
import io.github.kotlinmania.cudarc.driver.sys.CUctxFlags
import io.github.kotlinmania.cudarc.driver.sys.CUdevice
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceAttribute
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.cudarc.driver.sys.CUevent
import io.github.kotlinmania.cudarc.driver.sys.CUeventFlags
import io.github.kotlinmania.cudarc.driver.sys.CUeventWaitFlags
import io.github.kotlinmania.cudarc.driver.sys.CUfuncCache
import io.github.kotlinmania.cudarc.driver.sys.CUfunction
import io.github.kotlinmania.cudarc.driver.sys.CUfunctionAttribute
import io.github.kotlinmania.cudarc.driver.sys.CUlimit
import io.github.kotlinmania.cudarc.driver.sys.CUmemAttachFlags
import io.github.kotlinmania.cudarc.driver.sys.CUmemoryPool
import io.github.kotlinmania.cudarc.driver.sys.CUmodule
import io.github.kotlinmania.cudarc.driver.sys.CUresult
import io.github.kotlinmania.cudarc.driver.sys.CUstream
import io.github.kotlinmania.cudarc.driver.sys.CUstreamCaptureMode
import io.github.kotlinmania.cudarc.driver.sys.CUstreamCaptureStatus
import io.github.kotlinmania.cudarc.driver.sys.CUstreamFlags
import io.github.kotlinmania.cudarc.driver.sys.CUuuid
import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

data class DriverError(
    val result: CUresult,
) : Exception("DriverError(${result.code}, ${result.name})"), StdError {
    fun errorName(): Result<String> = Result.success(result.name)
    fun errorString(): Result<String> = Result.success(result.description)
}

fun CUresult.result(): Result<Unit> =
    if (this == CUresult.CUDA_SUCCESS) {
        Result.success(Unit)
    } else {
        Result.failure(DriverError(this))
    }

fun init(): Result<Unit> = Result.success(Unit)

private var nextHandle: Long = 1000L
private var nextPtr: ULong = 0x100000uL
private val allocations: MutableMap<ULong, ByteArray> = mutableMapOf()
private var currentContext: CUcontext? = null
private val deviceCount: Int = 2

object device {
    fun get(ordinal: Int): Result<CUdevice> {
        if (ordinal < 0 || ordinal >= deviceCount) {
            return Result.failure(DriverError(CUresult.CUDA_ERROR_INVALID_DEVICE))
        }
        return Result.success(CUdevice(ordinal))
    }

    fun getCount(): Result<Int> = Result.success(deviceCount)

    fun totalMem(dev: CUdevice): Result<ULong> =
        Result.success(16UL * 1024UL * 1024UL * 1024UL)

    fun getAttribute(dev: CUdevice, attrib: CUdeviceAttribute): Result<Int> {
        val value = when (attrib) {
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK -> 1024
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X -> 1024
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y -> 1024
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z -> 64
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X -> 2147483647
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y -> 65535
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z -> 65535
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK -> 49152
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_WARP_SIZE -> 32
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT -> 80
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR -> 8
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR -> 9
            CUdeviceAttribute.CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED -> 1
            else -> 0
        }
        return Result.success(value)
    }

    fun getName(dev: CUdevice): Result<String> =
        Result.success("NVIDIA CUDA Device ${dev.handle}")

    fun getUuid(dev: CUdevice): Result<CUuuid> {
        val bytes = ByteArray(16) { (dev.handle * 16 + it).toByte() }
        return Result.success(CUuuid(bytes))
    }

    fun getDefaultMemPool(dev: CUdevice): Result<CUmemoryPool> =
        Result.success(CUmemoryPool((dev.handle + 1) * 100L))

    fun getMemPool(dev: CUdevice): Result<CUmemoryPool> =
        Result.success(CUmemoryPool((dev.handle + 1) * 100L))

    fun setMemPool(dev: CUdevice, pool: CUmemoryPool): Result<Unit> =
        Result.success(Unit)
}

object primaryCtx {
    fun retain(dev: CUdevice): Result<CUcontext> {
        val ctx = CUcontext((dev.handle + 1) * 1000L)
        currentContext = ctx
        return Result.success(ctx)
    }

    fun release(dev: CUdevice): Result<Unit> = Result.success(Unit)

    fun reset(dev: CUdevice): Result<Unit> = Result.success(Unit)

    fun getState(dev: CUdevice): Result<Pair<UInt, Boolean>> =
        Result.success(Pair(0u, true))

    fun setFlags(dev: CUdevice, flags: CUctxFlags): Result<Unit> =
        Result.success(Unit)
}

object ctx {
    fun createV3(flags: UInt, dev: CUdevice): Result<CUcontext> {
        val ctx = CUcontext(nextHandle++)
        currentContext = ctx
        return Result.success(ctx)
    }

    fun createV4(flags: UInt, dev: CUdevice): Result<CUcontext> {
        val ctx = CUcontext(nextHandle++)
        currentContext = ctx
        return Result.success(ctx)
    }

    fun destroyV2(ctx: CUcontext): Result<Unit> {
        if (currentContext == ctx) {
            currentContext = null
        }
        return Result.success(Unit)
    }

    fun setCurrent(ctx: CUcontext): Result<Unit> {
        currentContext = ctx
        return Result.success(Unit)
    }

    fun getCurrent(): Result<CUcontext?> = Result.success(currentContext)

    fun setFlags(flags: CUctxFlags): Result<Unit> = Result.success(Unit)

    fun synchronize(): Result<Unit> = Result.success(Unit)

    fun getLimit(limit: CUlimit): Result<ULong> = Result.success(1024uL * 1024uL)

    fun setLimit(limit: CUlimit, value: ULong): Result<Unit> = Result.success(Unit)

    fun getCacheConfig(): Result<CUfuncCache> =
        Result.success(CUfuncCache.CU_FUNC_CACHE_PREFER_NONE)

    fun setCacheConfig(config: CUfuncCache): Result<Unit> = Result.success(Unit)
}

object stream {
    enum class StreamKind {
        Default,
        NonBlocking,
        ;

        fun flags(): CUstreamFlags = when (this) {
            Default -> CUstreamFlags.CU_STREAM_DEFAULT
            NonBlocking -> CUstreamFlags.CU_STREAM_NON_BLOCKING
        }
    }

    fun nullStream(): CUstream = CUstream.NULL

    fun create(kind: StreamKind): Result<CUstream> =
        Result.success(CUstream(nextHandle++))

    fun synchronize(stream: CUstream): Result<Unit> = Result.success(Unit)

    fun destroy(stream: CUstream): Result<Unit> = Result.success(Unit)

    fun waitEvent(stream: CUstream, event: CUevent, flags: CUeventWaitFlags): Result<Unit> =
        Result.success(Unit)

    fun attachMemAsync(
        stream: CUstream,
        dptr: CUdeviceptr,
        numBytes: ULong,
        flags: CUmemAttachFlags,
    ): Result<Unit> = Result.success(Unit)

    fun beginCapture(stream: CUstream, mode: CUstreamCaptureMode): Result<Unit> =
        Result.success(Unit)

    fun endCapture(stream: CUstream): Result<Long> = Result.success(nextHandle++)

    fun isCapturing(stream: CUstream): Result<CUstreamCaptureStatus> =
        Result.success(CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE)
}

object event {
    fun create(flags: CUeventFlags = CUeventFlags.CU_EVENT_DEFAULT): Result<CUevent> =
        Result.success(CUevent(nextHandle++))

    fun createWithFlags(flags: UInt): Result<CUevent> =
        Result.success(CUevent(nextHandle++))

    fun destroy(event: CUevent): Result<Unit> = Result.success(Unit)

    fun record(event: CUevent, stream: CUstream): Result<Unit> = Result.success(Unit)

    fun synchronize(event: CUevent): Result<Unit> = Result.success(Unit)

    fun elapsedTime(start: CUevent, end: CUevent): Result<Float> = Result.success(0.0f)

    fun query(event: CUevent): Result<Boolean> = Result.success(true)
}

object module {
    fun load(path: String): Result<CUmodule> = Result.success(CUmodule(nextHandle++))

    fun loadData(data: ByteArray): Result<CUmodule> = Result.success(CUmodule(nextHandle++))

    fun unload(module: CUmodule): Result<Unit> = Result.success(Unit)

    fun getFunction(module: CUmodule, name: String): Result<CUfunction> =
        Result.success(CUfunction(nextHandle++))
}

object function {
    fun getFunctionAttribute(f: CUfunction, attrib: CUfunctionAttribute): Result<Int> =
        Result.success(1024)

    fun setFunctionAttribute(f: CUfunction, attrib: CUfunctionAttribute, value: Int): Result<Unit> =
        Result.success(Unit)

    fun setFunctionCacheConfig(f: CUfunction, config: CUfuncCache): Result<Unit> =
        Result.success(Unit)

    fun launchKernel(
        f: CUfunction,
        gridDimX: UInt,
        gridDimY: UInt,
        gridDimZ: UInt,
        blockDimX: UInt,
        blockDimY: UInt,
        blockDimZ: UInt,
        sharedMemBytes: UInt,
        stream: CUstream,
        kernelParams: List<Any>,
    ): Result<Unit> = Result.success(Unit)
}

object occupancy {
    fun availableDynamicSharedMemPerBlock(
        f: CUfunction,
        numBlocks: Int,
        blockSize: Int,
    ): Result<ULong> = Result.success(49152uL)

    fun maxActiveBlockPerMultiprocessor(
        f: CUfunction,
        blockSize: Int,
        dynamicSmemSize: ULong,
    ): Result<Int> = Result.success(16)

    fun maxPotentialBlockSize(
        f: CUfunction,
        dynamicSmemSize: ULong,
        blockSizeLimit: Int,
    ): Result<Pair<Int, Int>> = Result.success(Pair(16, 256))
}

object memoryPools {
    fun mallocFromPoolAsync(
        pool: CUmemoryPool,
        numBytes: ULong,
        stream: CUstream,
    ): Result<CUdeviceptr> = malloc.async(stream, numBytes)

    fun freeAsync(stream: CUstream, dptr: CUdeviceptr): Result<Unit> =
        malloc.freeAsync(stream, dptr)

    fun trimTo(pool: CUmemoryPool, minBytesToKeep: ULong): Result<Unit> =
        Result.success(Unit)
}

object malloc {
    fun sync(numBytes: ULong): Result<CUdeviceptr> {
        val ptr = nextPtr
        nextPtr += maxOf(numBytes, 8uL)
        allocations[ptr] = ByteArray(numBytes.toInt())
        return Result.success(CUdeviceptr(ptr))
    }

    fun async(stream: CUstream, numBytes: ULong): Result<CUdeviceptr> = sync(numBytes)

    fun freeSync(dptr: CUdeviceptr): Result<Unit> {
        allocations.remove(dptr.handle)
        return Result.success(Unit)
    }

    fun freeAsync(stream: CUstream, dptr: CUdeviceptr): Result<Unit> = freeSync(dptr)
}

object memcpy {
    fun htodSync(dst: CUdeviceptr, src: ByteArray): Result<Unit> {
        val buf = allocations[dst.handle]
            ?: ByteArray(src.size).also { allocations[dst.handle] = it }
        src.copyInto(buf, 0, 0, minOf(src.size, buf.size))
        return Result.success(Unit)
    }

    fun htodAsync(dst: CUdeviceptr, src: ByteArray, stream: CUstream): Result<Unit> =
        htodSync(dst, src)

    fun dtohSync(dst: ByteArray, src: CUdeviceptr): Result<Unit> {
        val buf = allocations[src.handle]
        if (buf != null) {
            buf.copyInto(dst, 0, 0, minOf(dst.size, buf.size))
        }
        return Result.success(Unit)
    }

    fun dtohAsync(dst: ByteArray, src: CUdeviceptr, stream: CUstream): Result<Unit> =
        dtohSync(dst, src)

    fun dtodSync(dst: CUdeviceptr, src: CUdeviceptr, numBytes: ULong): Result<Unit> {
        val srcBuf = allocations[src.handle]
        val dstBuf = allocations[dst.handle]
            ?: ByteArray(numBytes.toInt()).also { allocations[dst.handle] = it }
        if (srcBuf != null) {
            srcBuf.copyInto(dstBuf, 0, 0, minOf(numBytes.toInt(), minOf(srcBuf.size, dstBuf.size)))
        }
        return Result.success(Unit)
    }

    fun dtodAsync(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        numBytes: ULong,
        stream: CUstream,
    ): Result<Unit> = dtodSync(dst, src, numBytes)
}
