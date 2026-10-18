// port-lint: source cublas/result.rs
package io.github.kotlinmania.cudarc.cublas.result

import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CublasStatus(
    val code: Int,
    val description: String,
) {
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

enum class CublasOperation(
    val code: Int,
) {
    CUBLAS_OP_N(0),
    CUBLAS_OP_T(1),
    CUBLAS_OP_C(2),
}

enum class CublasPointerMode(
    val code: Int,
) {
    CUBLAS_POINTER_MODE_HOST(0),
    CUBLAS_POINTER_MODE_DEVICE(1),
}

data class CublasHandle(
    val handle: Long,
)

data class CublasError(
    val status: CublasStatus,
) : Exception("CublasError(${status.code}, ${status.name})"),
    StdError

private var nextHandleId: Long = 1L
private val pointerModes: MutableMap<Long, CublasPointerMode> = mutableMapOf()

enum class CublasComputeType(
    val code: Int,
) {
    CUBLAS_COMPUTE_16F(64),
    CUBLAS_COMPUTE_16F_PEDANTIC(65),
    CUBLAS_COMPUTE_32F(68),
    CUBLAS_COMPUTE_32F_PEDANTIC(69),
    CUBLAS_COMPUTE_32F_FAST_16F(74),
    CUBLAS_COMPUTE_32F_FAST_16BF(75),
    CUBLAS_COMPUTE_32F_FAST_TF32(77),
    CUBLAS_COMPUTE_64F(70),
    CUBLAS_COMPUTE_64F_PEDANTIC(71),
    CUBLAS_COMPUTE_32I(72),
    CUBLAS_COMPUTE_32I_PEDANTIC(73),
}

enum class CudaDataType(
    val code: Int,
) {
    CUDA_R_16F(2),
    CUDA_C_16F(6),
    CUDA_R_16BF(14),
    CUDA_C_16BF(15),
    CUDA_R_32F(0),
    CUDA_C_32F(4),
    CUDA_R_64F(1),
    CUDA_C_64F(5),
    CUDA_R_4I(24),
    CUDA_C_4I(25),
    CUDA_R_4U(26),
    CUDA_C_4U(27),
    CUDA_R_8I(3),
    CUDA_C_8I(7),
    CUDA_R_8U(8),
    CUDA_C_8U(9),
    CUDA_R_16I(20),
    CUDA_C_16I(21),
    CUDA_R_16U(22),
    CUDA_C_16U(23),
    CUDA_R_32I(10),
    CUDA_C_32I(11),
    CUDA_R_32U(12),
    CUDA_C_32U(13),
    CUDA_R_64I(28),
    CUDA_C_64I(29),
    CUDA_R_64U(30),
    CUDA_C_64U(31),
    CUDA_R_8F_E4M3(32),
    CUDA_R_8F_E5M2(33),
}

enum class CublasGemmAlgo(
    val code: Int,
) {
    CUBLAS_GEMM_DEFAULT(-1),
    CUBLAS_GEMM_ALGO0(0),
    CUBLAS_GEMM_ALGO1(1),
    CUBLAS_GEMM_ALGO2(2),
    CUBLAS_GEMM_ALGO3(3),
    CUBLAS_GEMM_ALGO4(4),
    CUBLAS_GEMM_ALGO5(5),
    CUBLAS_GEMM_ALGO6(6),
    CUBLAS_GEMM_ALGO7(7),
    CUBLAS_GEMM_ALGO8(8),
    CUBLAS_GEMM_ALGO9(9),
    CUBLAS_GEMM_ALGO10(10),
    CUBLAS_GEMM_ALGO11(11),
    CUBLAS_GEMM_ALGO12(12),
    CUBLAS_GEMM_ALGO13(13),
    CUBLAS_GEMM_ALGO14(14),
    CUBLAS_GEMM_ALGO15(15),
    CUBLAS_GEMM_ALGO16(16),
    CUBLAS_GEMM_ALGO17(17),
    CUBLAS_GEMM_ALGO18(18),
    CUBLAS_GEMM_ALGO19(19),
    CUBLAS_GEMM_ALGO20(20),
    CUBLAS_GEMM_ALGO21(21),
    CUBLAS_GEMM_ALGO22(22),
    CUBLAS_GEMM_ALGO23(23),
}

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

fun sgemv(
    handle: CublasHandle,
    trans: CublasOperation,
    m: Int,
    n: Int,
    alpha: Float,
    a: CUdeviceptr,
    lda: Int,
    x: CUdeviceptr,
    incx: Int,
    beta: Float,
    y: CUdeviceptr,
    incy: Int,
): Result<Unit> = Result.success(Unit)

fun dgemv(
    handle: CublasHandle,
    trans: CublasOperation,
    m: Int,
    n: Int,
    alpha: Double,
    a: CUdeviceptr,
    lda: Int,
    x: CUdeviceptr,
    incx: Int,
    beta: Double,
    y: CUdeviceptr,
    incy: Int,
): Result<Unit> = Result.success(Unit)

fun sgemm(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Float,
    a: CUdeviceptr,
    lda: Int,
    b: CUdeviceptr,
    ldb: Int,
    beta: Float,
    c: CUdeviceptr,
    ldc: Int,
): Result<Unit> = Result.success(Unit)

fun dgemm(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Double,
    a: CUdeviceptr,
    lda: Int,
    b: CUdeviceptr,
    ldb: Int,
    beta: Double,
    c: CUdeviceptr,
    ldc: Int,
): Result<Unit> = Result.success(Unit)

fun hgemm(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Short,
    a: CUdeviceptr,
    lda: Int,
    b: CUdeviceptr,
    ldb: Int,
    beta: Short,
    c: CUdeviceptr,
    ldc: Int,
): Result<Unit> = Result.success(Unit)

fun sgemmStridedBatched(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Float,
    a: CUdeviceptr,
    lda: Int,
    strideA: Long,
    b: CUdeviceptr,
    ldb: Int,
    strideB: Long,
    beta: Float,
    c: CUdeviceptr,
    ldc: Int,
    strideC: Long,
    batchSize: Int,
): Result<Unit> = Result.success(Unit)

fun dgemmStridedBatched(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Double,
    a: CUdeviceptr,
    lda: Int,
    strideA: Long,
    b: CUdeviceptr,
    ldb: Int,
    strideB: Long,
    beta: Double,
    c: CUdeviceptr,
    ldc: Int,
    strideC: Long,
    batchSize: Int,
): Result<Unit> = Result.success(Unit)

fun hgemmStridedBatched(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Short,
    a: CUdeviceptr,
    lda: Int,
    strideA: Long,
    b: CUdeviceptr,
    ldb: Int,
    strideB: Long,
    beta: Short,
    c: CUdeviceptr,
    ldc: Int,
    strideC: Long,
    batchSize: Int,
): Result<Unit> = Result.success(Unit)

fun gemmEx(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Long,
    a: CUdeviceptr,
    aType: CudaDataType,
    lda: Int,
    b: CUdeviceptr,
    bType: CudaDataType,
    ldb: Int,
    beta: Long,
    c: CUdeviceptr,
    cType: CudaDataType,
    ldc: Int,
    computeType: CublasComputeType,
    algo: CublasGemmAlgo,
): Result<Unit> = Result.success(Unit)

fun gemmStridedBatchedEx(
    handle: CublasHandle,
    transa: CublasOperation,
    transb: CublasOperation,
    m: Int,
    n: Int,
    k: Int,
    alpha: Long,
    a: CUdeviceptr,
    aType: CudaDataType,
    lda: Int,
    strideA: Long,
    b: CUdeviceptr,
    bType: CudaDataType,
    ldb: Int,
    strideB: Long,
    beta: Long,
    c: CUdeviceptr,
    cType: CudaDataType,
    ldc: Int,
    strideC: Long,
    batchCount: Int,
    computeType: CublasComputeType,
    algo: CublasGemmAlgo,
): Result<Unit> = Result.success(Unit)

fun sasum(
    handle: CublasHandle,
    n: Int,
    x: CUdeviceptr,
    incx: Int,
    result: FloatArray,
): Result<Unit> = Result.success(Unit)

fun dasum(
    handle: CublasHandle,
    n: Int,
    x: CUdeviceptr,
    incx: Int,
    result: DoubleArray,
): Result<Unit> = Result.success(Unit)

fun cublasGemmGroupedBatchedEx(
    handle: CublasHandle,
    transa: List<CublasOperation>,
    transb: List<CublasOperation>,
    m: List<Int>,
    n: List<Int>,
    k: List<Int>,
    alpha: List<Float>,
    a: List<CUdeviceptr>,
    aType: CudaDataType,
    lda: List<Int>,
    b: List<CUdeviceptr>,
    bType: CudaDataType,
    ldb: List<Int>,
    beta: List<Float>,
    c: List<CUdeviceptr>,
    cType: CudaDataType,
    ldc: List<Int>,
    groupCount: Int,
    groupSize: List<Int>,
    computeType: CublasComputeType,
): Result<Unit> = Result.success(Unit)
