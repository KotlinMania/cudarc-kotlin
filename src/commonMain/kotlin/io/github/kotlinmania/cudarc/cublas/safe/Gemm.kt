// port-lint: source cublas/safe/gemm.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.driver.safe.sliceElements
import kotlin.Result

data class GemmConfig<T>(
    val transa: CublasOperation,
    val transb: CublasOperation,
    val m: Int,
    val n: Int,
    val k: Int,
    val alpha: T,
    val lda: Int,
    val ldb: Int,
    val beta: T,
    val ldc: Int,
)

data class StridedBatchedConfig<T>(
    val gemm: GemmConfig<T>,
    val batchSize: Int,
    val strideA: Long,
    val strideB: Long,
    val strideC: Long,
)

interface Gemm<T> {
    fun gemm(
        cfg: GemmConfig<T>,
        a: DevicePtr<T>,
        b: DevicePtr<T>,
        c: DevicePtrMut<T>,
    ): Result<Unit>

    fun gemmStridedBatched(
        cfg: StridedBatchedConfig<T>,
        a: DevicePtr<T>,
        b: DevicePtr<T>,
        c: DevicePtrMut<T>,
    ): Result<Unit>
}

fun CudaBlas.gemmFloat(
    cfg: GemmConfig<Float>,
    a: DevicePtr<Float>,
    b: DevicePtr<Float>,
    c: DevicePtrMut<Float>,
): Result<Unit> {
    val aElems = a.sliceElements()
    val bElems = b.sliceElements()
    val cElems = c.sliceElements()
    if (aElems != null && bElems != null && cElems != null) {
        val m = cfg.m
        val n = cfg.n
        val k = cfg.k
        val cElements = FloatArray(m * n) { 0f }
        for (col in 0 until n) {
            for (row in 0 until m) {
                var sum = 0f
                for (p in 0 until k) {
                    val aVal = aElems[row + p * cfg.lda]
                    val bVal = bElems[p + col * cfg.ldb]
                    sum += aVal * bVal
                }
                val prev = if (row + col * cfg.ldc < cElems.size) cElems[row + col * cfg.ldc] else 0f
                cElements[row + col * cfg.ldc] = cfg.alpha * sum + cfg.beta * prev
            }
        }
        cElems.clear()
        cElems.addAll(cElements.toList())
        return Result.success(Unit)
    }
    val aPtr = a.devicePtr(stream).first
    val bPtr = b.devicePtr(stream).first
    val cPtr = c.devicePtrMut(stream).first
    return io.github.kotlinmania.cudarc.cublas.result.sgemm(
        handle = handle,
        transa = cfg.transa,
        transb = cfg.transb,
        m = cfg.m,
        n = cfg.n,
        k = cfg.k,
        alpha = cfg.alpha,
        a = aPtr,
        lda = cfg.lda,
        b = bPtr,
        ldb = cfg.ldb,
        beta = cfg.beta,
        c = cPtr,
        ldc = cfg.ldc,
    )
}

@kotlin.jvm.JvmName("gemmF32")
fun CudaBlas.gemm(
    cfg: GemmConfig<Float>,
    a: DevicePtr<Float>,
    b: DevicePtr<Float>,
    c: DevicePtrMut<Float>,
): Result<Unit> = gemmFloat(cfg, a, b, c)

fun CudaBlas.gemmDouble(
    cfg: GemmConfig<Double>,
    a: DevicePtr<Double>,
    b: DevicePtr<Double>,
    c: DevicePtrMut<Double>,
): Result<Unit> {
    val aElems = a.sliceElements()
    val bElems = b.sliceElements()
    val cElems = c.sliceElements()
    if (aElems != null && bElems != null && cElems != null) {
        val m = cfg.m
        val n = cfg.n
        val k = cfg.k
        val cElements = DoubleArray(m * n) { 0.0 }
        for (col in 0 until n) {
            for (row in 0 until m) {
                var sum = 0.0
                for (p in 0 until k) {
                    val aVal = aElems[row + p * cfg.lda]
                    val bVal = bElems[p + col * cfg.ldb]
                    sum += aVal * bVal
                }
                val prev = if (row + col * cfg.ldc < cElems.size) cElems[row + col * cfg.ldc] else 0.0
                cElements[row + col * cfg.ldc] = cfg.alpha * sum + cfg.beta * prev
            }
        }
        cElems.clear()
        cElems.addAll(cElements.toList())
        return Result.success(Unit)
    }
    val aPtr = a.devicePtr(stream).first
    val bPtr = b.devicePtr(stream).first
    val cPtr = c.devicePtrMut(stream).first
    return io.github.kotlinmania.cudarc.cublas.result.dgemm(
        handle = handle,
        transa = cfg.transa,
        transb = cfg.transb,
        m = cfg.m,
        n = cfg.n,
        k = cfg.k,
        alpha = cfg.alpha,
        a = aPtr,
        lda = cfg.lda,
        b = bPtr,
        ldb = cfg.ldb,
        beta = cfg.beta,
        c = cPtr,
        ldc = cfg.ldc,
    )
}

@kotlin.jvm.JvmName("gemmF64")
fun CudaBlas.gemm(
    cfg: GemmConfig<Double>,
    a: DevicePtr<Double>,
    b: DevicePtr<Double>,
    c: DevicePtrMut<Double>,
): Result<Unit> = gemmDouble(cfg, a, b, c)
