package io.github.kotlinmania.cudarc.driver.sys

enum class CUresult(
    val code: Int,
    val description: String,
) {
    CUDA_SUCCESS(0, "Success"),
    CUDA_ERROR_INVALID_VALUE(1, "Invalid value"),
    CUDA_ERROR_OUT_OF_MEMORY(2, "Out of memory"),
    CUDA_ERROR_NOT_INITIALIZED(3, "Driver not initialized"),
    CUDA_ERROR_DEINITIALIZED(4, "Driver deinitialized"),
    CUDA_ERROR_PROFILER_DISABLED(5, "Profiler disabled"),
    CUDA_ERROR_PROFILER_NOT_INITIALIZED(6, "Profiler not initialized"),
    CUDA_ERROR_PROFILER_ALREADY_STARTED(7, "Profiler already started"),
    CUDA_ERROR_PROFILER_ALREADY_STOPPED(8, "Profiler already stopped"),
    CUDA_ERROR_STUB_LIBRARY(34, "Stub library"),
    CUDA_ERROR_DEVICE_UNAVAILABLE(46, "Device unavailable"),
    CUDA_ERROR_NO_DEVICE(100, "No device"),
    CUDA_ERROR_INVALID_DEVICE(101, "Invalid device"),
    CUDA_ERROR_DEVICE_NOT_LICENSED(102, "Device not licensed"),
    CUDA_ERROR_INVALID_IMAGE(200, "Invalid image"),
    CUDA_ERROR_INVALID_CONTEXT(201, "Invalid context"),
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT(202, "Context already current"),
    CUDA_ERROR_MAP_FAILED(205, "Map failed"),
    CUDA_ERROR_UNMAP_FAILED(206, "Unmap failed"),
    CUDA_ERROR_ARRAY_IS_MAPPED(207, "Array is mapped"),
    CUDA_ERROR_ALREADY_MAPPED(208, "Already mapped"),
    CUDA_ERROR_NO_BINARY_FOR_GPU(209, "No binary for GPU"),
    CUDA_ERROR_ALREADY_ACQUIRED(210, "Already acquired"),
    CUDA_ERROR_NOT_MAPPED(211, "Not mapped"),
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY(212, "Not mapped as array"),
    CUDA_ERROR_NOT_MAPPED_AS_POINTER(213, "Not mapped as pointer"),
    CUDA_ERROR_ECC_UNCORRECTABLE(214, "ECC uncorrectable"),
    CUDA_ERROR_UNSUPPORTED_LIMIT(215, "Unsupported limit"),
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE(216, "Context already in use"),
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED(217, "Peer access unsupported"),
    CUDA_ERROR_INVALID_SOURCE(300, "Invalid source"),
    CUDA_ERROR_FILE_NOT_FOUND(301, "File not found"),
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND(302, "Shared object symbol not found"),
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED(303, "Shared object init failed"),
    CUDA_ERROR_OPERATING_SYSTEM(304, "Operating system error"),
    CUDA_ERROR_INVALID_HANDLE(400, "Invalid handle"),
    CUDA_ERROR_ILLEGAL_STATE(401, "Illegal state"),
    CUDA_ERROR_NOT_FOUND(500, "Not found"),
    CUDA_ERROR_NOT_READY(600, "Not ready"),
    CUDA_ERROR_ILLEGAL_ADDRESS(700, "Illegal address"),
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES(701, "Launch out of resources"),
    CUDA_ERROR_LAUNCH_TIMEOUT(702, "Launch timeout"),
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING(703, "Launch incompatible texturing"),
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED(704, "Peer access already enabled"),
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED(705, "Peer access not enabled"),
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE(708, "Primary context active"),
    CUDA_ERROR_CONTEXT_IS_DESTROYED(709, "Context is destroyed"),
    CUDA_ERROR_ASSERT(710, "Assert failed"),
    CUDA_ERROR_TOO_MANY_PEERS(711, "Too many peers"),
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED(712, "Host memory already registered"),
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED(713, "Host memory not registered"),
    CUDA_ERROR_HARDWARE_STACK_ERROR(714, "Hardware stack error"),
    CUDA_ERROR_ILLEGAL_INSTRUCTION(715, "Illegal instruction"),
    CUDA_ERROR_MISALIGNED_ADDRESS(716, "Misaligned address"),
    CUDA_ERROR_INVALID_ADDRESS_SPACE(717, "Invalid address space"),
    CUDA_ERROR_INVALID_PC(718, "Invalid program counter"),
    CUDA_ERROR_LAUNCH_FAILED(719, "Launch failed"),
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE(720, "Cooperative launch too large"),
    CUDA_ERROR_NOT_PERMITTED(800, "Not permitted"),
    CUDA_ERROR_NOT_SUPPORTED(801, "Not supported"),
    CUDA_ERROR_SYSTEM_NOT_READY(802, "System not ready"),
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH(803, "System driver mismatch"),
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE(804, "Compatibility not supported on device"),
    CUDA_ERROR_MPS_CONNECTION_FAILED(805, "MPS connection failed"),
    CUDA_ERROR_MPS_RPC_FAILURE(806, "MPS RPC failure"),
    CUDA_ERROR_MPS_SERVER_NOT_READY(807, "MPS server not ready"),
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED(808, "MPS max clients reached"),
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED(809, "MPS max connections reached"),
    CUDA_ERROR_MPS_CLIENT_TERMINATED(810, "MPS client terminated"),
    CUDA_ERROR_CDP_NOT_SUPPORTED(811, "CDP not supported"),
    CUDA_ERROR_CDP_VERSION_MISMATCH(812, "CDP version mismatch"),
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED(900, "Stream capture unsupported"),
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED(901, "Stream capture invalidated"),
    CUDA_ERROR_STREAM_CAPTURE_MERGE(902, "Stream capture merge"),
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED(903, "Stream capture unmatched"),
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED(904, "Stream capture unjoined"),
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION(905, "Stream capture isolation"),
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT(906, "Stream capture implicit"),
    CUDA_ERROR_CAPTURED_EVENT(907, "Captured event"),
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD(908, "Stream capture wrong thread"),
    CUDA_ERROR_TIMEOUT(909, "Timeout"),
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE(910, "Graph exec update failure"),
    CUDA_ERROR_EXTERNAL_DEVICE(911, "External device error"),
    CUDA_ERROR_INVALID_CLUSTER_SIZE(912, "Invalid cluster size"),
    CUDA_ERROR_UNKNOWN(999, "Unknown error"),
    ;

    companion object {
        fun fromCode(code: Int): CUresult =
            entries.find { it.code == code } ?: CUDA_ERROR_UNKNOWN
    }
}

data class CUdevice(
    val handle: Int,
)

data class CUcontext(
    val handle: Long,
) {
    fun isNull(): Boolean = handle == 0L

    companion object {
        val NULL = CUcontext(0L)
    }
}

data class CUstream(
    val handle: Long,
) {
    fun isNull(): Boolean = handle == 0L

    companion object {
        val NULL = CUstream(0L)
        val PER_THREAD = CUstream(2L)
    }
}

data class CUevent(
    val handle: Long,
) {
    fun isNull(): Boolean = handle == 0L

    companion object {
        val NULL = CUevent(0L)
    }
}

data class CUfunction(
    val handle: Long,
) {
    fun isNull(): Boolean = handle == 0L

    companion object {
        val NULL = CUfunction(0L)
    }
}

data class CUmodule(
    val handle: Long,
) {
    fun isNull(): Boolean = handle == 0L

    companion object {
        val NULL = CUmodule(0L)
    }
}

data class CUdeviceptr(
    val handle: ULong,
) {
    fun isNull(): Boolean = handle == 0uL

    companion object {
        val NULL = CUdeviceptr(0uL)
    }
}

data class CUmemoryPool(
    val handle: Long,
) {
    fun isNull(): Boolean = handle == 0L

    companion object {
        val NULL = CUmemoryPool(0L)
    }
}

data class CUuuid(
    val bytes: ByteArray,
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is CUuuid) return false
        return bytes.contentEquals(other.bytes)
    }

    override fun hashCode(): Int = bytes.contentHashCode()
}

enum class CUstreamCaptureStatus(
    val code: Int,
) {
    CU_STREAM_CAPTURE_STATUS_NONE(0),
    CU_STREAM_CAPTURE_STATUS_ACTIVE(1),
    CU_STREAM_CAPTURE_STATUS_INVALIDATED(2),
}

enum class CUstreamCaptureMode(
    val code: Int,
) {
    CU_STREAM_CAPTURE_MODE_GLOBAL(0),
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL(1),
    CU_STREAM_CAPTURE_MODE_RELAXED(2),
}

enum class CUstreamFlags(
    val value: UInt,
) {
    CU_STREAM_DEFAULT(0u),
    CU_STREAM_NON_BLOCKING(1u),
}

enum class CUeventFlags(
    val value: UInt,
) {
    CU_EVENT_DEFAULT(0u),
    CU_EVENT_BLOCKING_SYNC(1u),
    CU_EVENT_DISABLE_TIMING(2u),
    CU_EVENT_INTERPROCESS(4u),
}

enum class CUeventWaitFlags(
    val value: UInt,
) {
    CU_EVENT_WAIT_DEFAULT(0u),
    CU_EVENT_WAIT_EXTERNAL(1u),
}

enum class CUmemAttachFlags(
    val value: UInt,
) {
    CU_MEM_ATTACH_GLOBAL(1u),
    CU_MEM_ATTACH_HOST(2u),
    CU_MEM_ATTACH_SINGLE(4u),
}

enum class CUctxFlags(
    val value: UInt,
) {
    CU_CTX_SCHED_AUTO(0u),
    CU_CTX_SCHED_SPIN(1u),
    CU_CTX_SCHED_YIELD(2u),
    CU_CTX_SCHED_BLOCKING_SYNC(4u),
    CU_CTX_MAP_HOST(8u),
    CU_CTX_LMEM_RESIZE_TO_MAX(16u),
}

enum class CUlimit(
    val code: Int,
) {
    CU_LIMIT_STACK_SIZE(0),
    CU_LIMIT_PRINTF_FIFO_SIZE(1),
    CU_LIMIT_MALLOC_HEAP_SIZE(2),
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH(3),
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT(4),
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY(5),
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE(6),
}

enum class CUfuncCache(
    val code: Int,
) {
    CU_FUNC_CACHE_PREFER_NONE(0),
    CU_FUNC_CACHE_PREFER_SHARED(1),
    CU_FUNC_CACHE_PREFER_L1(2),
    CU_FUNC_CACHE_PREFER_EQUAL(3),
}

enum class CUfunctionAttribute(
    val code: Int,
) {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK(0),
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES(1),
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES(2),
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES(3),
    CU_FUNC_ATTRIBUTE_NUM_REGS(4),
    CU_FUNC_ATTRIBUTE_PTX_VERSION(5),
    CU_FUNC_ATTRIBUTE_BINARY_VERSION(6),
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES(7),
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT(8),
}

enum class CUdeviceAttribute(
    val code: Int,
) {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK(1),
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X(2),
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y(3),
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z(4),
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X(5),
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y(6),
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z(7),
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK(8),
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY(9),
    CU_DEVICE_ATTRIBUTE_WARP_SIZE(10),
    CU_DEVICE_ATTRIBUTE_MAX_PITCH(11),
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK(12),
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE(13),
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT(14),
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP(15),
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT(16),
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT(17),
    CU_DEVICE_ATTRIBUTE_INTEGRATED(18),
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY(19),
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE(20),
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS(31),
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED(32),
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID(33),
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID(34),
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER(35),
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE(36),
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH(37),
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE(38),
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR(39),
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT(40),
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING(41),
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR(81),
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR(75),
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR(76),
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED(115),
}

enum class CUmemPoolAttribute(
    val code: Int,
) {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES(1),
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC(2),
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES(3),
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD(4),
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT(5),
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH(6),
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT(7),
    CU_MEMPOOL_ATTR_USED_MEM_HIGH(8),
}

enum class CUmemAccessFlags(
    val code: Int,
) {
    CU_MEM_ACCESS_FLAGS_PROT_NONE(0),
    CU_MEM_ACCESS_FLAGS_PROT_READ(1),
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE(3),
}
