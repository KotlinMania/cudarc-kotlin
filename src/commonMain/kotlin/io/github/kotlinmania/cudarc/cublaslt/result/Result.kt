// port-lint: source cublaslt/result.rs
package io.github.kotlinmania.cudarc.cublaslt.result

import io.github.kotlinmania.cudarc.cublas.result.CublasComputeType
import io.github.kotlinmania.cudarc.cublas.result.CublasStatus
import io.github.kotlinmania.cudarc.cublas.result.CudaDataType
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import kotlin.Result

data class CublasLtHandle(
    val handle: Long,
)

data class CublasLtMatrixLayout(
    val handle: Long,
)

data class CublasLtMatmulDesc(
    val handle: Long,
)

data class CublasLtMatmulPreference(
    val handle: Long,
)

data class CublasLtMatmulAlgo(
    val data: LongArray = LongArray(8),
)

data class CublasLtMatmulHeuristicResult(
    val algo: CublasLtMatmulAlgo = CublasLtMatmulAlgo(),
    val workspaceSize: Long = 0L,
    val state: CublasStatus = CublasStatus.CUBLAS_STATUS_SUCCESS,
    val wavesCount: Float = 0f,
    val reserved: IntArray = IntArray(4),
)

enum class CublasLtMatrixLayoutAttribute(
    val code: Int,
) {
    CUBLASLT_MATRIX_LAYOUT_TYPE(0),
    CUBLASLT_MATRIX_LAYOUT_ORDER(1),
    CUBLASLT_MATRIX_LAYOUT_ROWS(2),
    CUBLASLT_MATRIX_LAYOUT_COLS(3),
    CUBLASLT_MATRIX_LAYOUT_LD(4),
    CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT(5),
    CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET(6),
    CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET(7),
}

enum class CublasLtMatmulDescAttributes(
    val code: Int,
) {
    CUBLASLT_MATMUL_DESC_COMPUTE_TYPE(0),
    CUBLASLT_MATMUL_DESC_SCALE_TYPE(1),
    CUBLASLT_MATMUL_DESC_POINTER_MODE(2),
    CUBLASLT_MATMUL_DESC_TRANSA(3),
    CUBLASLT_MATMUL_DESC_TRANSB(4),
    CUBLASLT_MATMUL_DESC_TRANSC(5),
    CUBLASLT_MATMUL_DESC_FILL_MODE(6),
    CUBLASLT_MATMUL_DESC_EPILOGUE(7),
    CUBLASLT_MATMUL_DESC_BIAS_POINTER(8),
    CUBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE(9),
}

enum class CublasLtEpilogue(
    val code: Int,
) {
    CUBLASLT_EPILOGUE_DEFAULT(1),
    CUBLASLT_EPILOGUE_RELU(2),
    CUBLASLT_EPILOGUE_RELU_AUX(3),
    CUBLASLT_EPILOGUE_BIAS(4),
    CUBLASLT_EPILOGUE_RELU_BIAS(5),
    CUBLASLT_EPILOGUE_RELU_AUX_BIAS(6),
    CUBLASLT_EPILOGUE_GELU(32),
    CUBLASLT_EPILOGUE_GELU_AUX(33),
    CUBLASLT_EPILOGUE_GELU_BIAS(34),
    CUBLASLT_EPILOGUE_GELU_AUX_BIAS(35),
}

enum class CublasLtMatmulPreferenceAttributes(
    val code: Int,
) {
    CUBLASLT_MATMUL_PREF_SEARCH_MODE(0),
    CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES(1),
    CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK(2),
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES(3),
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES(4),
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES(5),
    CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES(6),
    CUBLASLT_MATMUL_PREF_MAX_WAVES_COUNT(7),
}

private var nextId: Long = 1L

fun createHandle(): Result<CublasLtHandle> =
    Result.success(CublasLtHandle(nextId++))

fun destroyHandle(handle: CublasLtHandle): Result<Unit> =
    Result.success(Unit)

fun createMatrixLayout(
    matrixType: CudaDataType,
    rows: Long,
    cols: Long,
    ld: Long,
): Result<CublasLtMatrixLayout> =
    Result.success(CublasLtMatrixLayout(nextId++))

fun setMatrixLayoutAttribute(
    matrixLayout: CublasLtMatrixLayout,
    attr: CublasLtMatrixLayoutAttribute,
    value: Long,
): Result<Unit> = Result.success(Unit)

fun destroyMatrixLayout(matrixLayout: CublasLtMatrixLayout): Result<Unit> =
    Result.success(Unit)

fun createMatmulDesc(
    computeType: CublasComputeType,
    scaleType: CudaDataType,
): Result<CublasLtMatmulDesc> =
    Result.success(CublasLtMatmulDesc(nextId++))

fun setMatmulDescAttribute(
    matmulDesc: CublasLtMatmulDesc,
    attr: CublasLtMatmulDescAttributes,
    value: Long,
): Result<Unit> = Result.success(Unit)

fun destroyMatmulDesc(matmulDesc: CublasLtMatmulDesc): Result<Unit> =
    Result.success(Unit)

fun createMatmulPref(): Result<CublasLtMatmulPreference> =
    Result.success(CublasLtMatmulPreference(nextId++))

fun setMatmulPrefAttribute(
    matmulPref: CublasLtMatmulPreference,
    attr: CublasLtMatmulPreferenceAttributes,
    value: Long,
): Result<Unit> = Result.success(Unit)

fun destroyMatmulPref(matmulPref: CublasLtMatmulPreference): Result<Unit> =
    Result.success(Unit)

fun getMatmulAlgoHeuristic(
    handle: CublasLtHandle,
    matmulDesc: CublasLtMatmulDesc,
    aLayout: CublasLtMatrixLayout,
    bLayout: CublasLtMatrixLayout,
    cLayout: CublasLtMatrixLayout,
    dLayout: CublasLtMatrixLayout,
    matmulPref: CublasLtMatmulPreference,
): Result<CublasLtMatmulHeuristicResult> =
    Result.success(CublasLtMatmulHeuristicResult())

fun matmul(
    handle: CublasLtHandle,
    matmulDesc: CublasLtMatmulDesc,
    alpha: Float,
    beta: Float,
    a: CUdeviceptr,
    aLayout: CublasLtMatrixLayout,
    b: CUdeviceptr,
    bLayout: CublasLtMatrixLayout,
    c: CUdeviceptr,
    cLayout: CublasLtMatrixLayout,
    d: CUdeviceptr,
    dLayout: CublasLtMatrixLayout,
    algo: CublasLtMatmulAlgo,
    workspace: CUdeviceptr,
    workspaceSize: Long,
    stream: Long,
): Result<Unit> = Result.success(Unit)
