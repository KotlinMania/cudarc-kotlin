// port-lint: source cublaslt/safe.rs
package io.github.kotlinmania.cudarc.cublaslt.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasComputeType
import io.github.kotlinmania.cudarc.cublas.result.CudaDataType
import io.github.kotlinmania.cudarc.cublaslt.result.CublasLtHandle
import io.github.kotlinmania.cudarc.cublaslt.result.createHandle
import io.github.kotlinmania.cudarc.cublaslt.result.createMatmulDesc
import io.github.kotlinmania.cudarc.cublaslt.result.createMatmulPref
import io.github.kotlinmania.cudarc.cublaslt.result.createMatrixLayout
import io.github.kotlinmania.cudarc.cublaslt.result.destroyHandle
import io.github.kotlinmania.cudarc.cublaslt.result.destroyMatmulDesc
import io.github.kotlinmania.cudarc.cublaslt.result.destroyMatmulPref
import io.github.kotlinmania.cudarc.cublaslt.result.destroyMatrixLayout
import io.github.kotlinmania.cudarc.cublaslt.result.getMatmulAlgoHeuristic
import io.github.kotlinmania.cudarc.cublaslt.result.matmul
import io.github.kotlinmania.cudarc.driver.safe.CudaSlice
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.driver.safe.sliceElements
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

enum class Activation {
    Relu,
    Gelu,
}

class Workspace internal constructor(
    internal val buffer: CudaSlice<Byte>,
    val size: Long,
) {
    companion object {
        fun new(stream: CudaStream): Result<Workspace> {
            val workspaceSize = 4_194_304L
            val buffer = stream.allocZeros<Byte>(workspaceSize.toInt())
            return buffer.map { Workspace(it, workspaceSize) }
        }
    }
}

class CudaBlasLT(
    val handle: CublasLtHandle,
    val workspace: Workspace,
    val stream: CudaStream,
) {
    companion object {
        fun new(stream: CudaStream): Result<CudaBlasLT> {
            val handleResult = createHandle()
            val workspaceResult = Workspace.new(stream)
            return handleResult.flatMap { h ->
                workspaceResult.map { ws ->
                    CudaBlasLT(h, ws, stream)
                }
            }
        }
    }

    fun close() {
        destroyHandle(handle)
    }
}

data class MatmulConfig(
    val transa: Boolean,
    val transb: Boolean,
    val transc: Boolean = false,
    val m: Long,
    val n: Long,
    val k: Long,
    val alpha: Float,
    val lda: Long,
    val ldb: Long,
    val beta: Float,
    val ldc: Long,
    val strideA: Long? = null,
    val strideB: Long? = null,
    val strideC: Long? = null,
    val strideBias: Long? = null,
    val batchSize: Int? = null,
)

interface MatmulShared {
    fun handle(): CublasLtHandle

    fun workspace(): Workspace

    fun stream(): CudaStream
}

interface Matmul<T> : MatmulShared {
    fun matrixType(): CudaDataType

    fun computeType(): CublasComputeType

    fun <I : DevicePtr<T>, O : DevicePtrMut<T>> matmul(
        cfg: MatmulConfig,
        a: I,
        b: I,
        c: O,
        bias: I? = null,
        act: Activation? = null,
    ): Result<Unit>
}

fun CudaBlasLT.matmulFloat(
    cfg: MatmulConfig,
    a: DevicePtr<Float>,
    b: DevicePtr<Float>,
    c: DevicePtrMut<Float>,
    bias: DevicePtr<Float>? = null,
    act: Activation? = null,
): Result<Unit> {
    val aElems = a.sliceElements()
    val bElems = b.sliceElements()
    val cElems = c.sliceElements()
    if (aElems != null && bElems != null && cElems != null) {
        val m = cfg.m.toInt()
        val n = cfg.n.toInt()
        val k = cfg.k.toInt()
        val cElements = FloatArray(m * n) { 0f }
        for (col in 0 until n) {
            for (row in 0 until m) {
                var sum = 0f
                for (p in 0 until k) {
                    val aVal = aElems[row + p * cfg.lda.toInt()]
                    val bVal = bElems[p + col * cfg.ldb.toInt()]
                    sum += aVal * bVal
                }
                val prev = if (row + col * cfg.ldc.toInt() < cElems.size) cElems[row + col * cfg.ldc.toInt()] else 0f
                cElements[row + col * cfg.ldc.toInt()] = cfg.alpha * sum + cfg.beta * prev
            }
        }
        cElems.clear()
        cElems.addAll(cElements.toList())
        return Result.success(Unit)
    }
    val aRows = if (cfg.transa) cfg.k else cfg.m
    val aCols = if (cfg.transa) cfg.m else cfg.k
    val bRows = if (cfg.transb) cfg.n else cfg.k
    val bCols = if (cfg.transb) cfg.k else cfg.n

    val aLayoutRes = createMatrixLayout(CudaDataType.CUDA_R_32F, aRows, aCols, cfg.lda)
    val bLayoutRes = createMatrixLayout(CudaDataType.CUDA_R_32F, bRows, bCols, cfg.ldb)
    val cLayoutRes = createMatrixLayout(CudaDataType.CUDA_R_32F, cfg.m, cfg.n, cfg.ldc)
    val matmulDescRes = createMatmulDesc(CublasComputeType.CUBLAS_COMPUTE_32F_FAST_TF32, CudaDataType.CUDA_R_32F)
    val matmulPrefRes = createMatmulPref()

    return aLayoutRes.flatMap { aLayout ->
        bLayoutRes.flatMap { bLayout ->
            cLayoutRes.flatMap { cLayout ->
                matmulDescRes.flatMap { matmulDesc ->
                    matmulPrefRes.flatMap { matmulPref ->
                        val heuristicRes =
                            getMatmulAlgoHeuristic(
                                handle,
                                matmulDesc,
                                aLayout,
                                bLayout,
                                cLayout,
                                cLayout,
                                matmulPref,
                            )
                        heuristicRes.flatMap { heuristic ->
                            val aPtr = a.devicePtr(stream).first
                            val bPtr = b.devicePtr(stream).first
                            val cPtr = c.devicePtrMut(stream).first
                            val wPtr = workspace.buffer.devicePtr(stream).first
                            val res =
                                matmul(
                                    handle = handle,
                                    matmulDesc = matmulDesc,
                                    alpha = cfg.alpha,
                                    beta = cfg.beta,
                                    a = aPtr,
                                    aLayout = aLayout,
                                    b = bPtr,
                                    bLayout = bLayout,
                                    c = cPtr,
                                    cLayout = cLayout,
                                    d = cPtr,
                                    dLayout = cLayout,
                                    algo = heuristic.algo,
                                    workspace = wPtr,
                                    workspaceSize = workspace.size,
                                    stream = stream.cuStream.handle,
                                )
                            destroyMatmulPref(matmulPref)
                            destroyMatmulDesc(matmulDesc)
                            destroyMatrixLayout(cLayout)
                            destroyMatrixLayout(bLayout)
                            destroyMatrixLayout(aLayout)
                            res
                        }
                    }
                }
            }
        }
    }
}

fun CudaBlasLT.matmul(
    cfg: MatmulConfig,
    a: DevicePtr<Float>,
    b: DevicePtr<Float>,
    c: DevicePtrMut<Float>,
    bias: DevicePtr<Float>? = null,
    act: Activation? = null,
): Result<Unit> = matmulFloat(cfg, a, b, c, bias, act)
