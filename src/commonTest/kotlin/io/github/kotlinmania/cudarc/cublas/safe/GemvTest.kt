// port-lint: tests cublas/safe/gemv.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertTrue

class GemvTest {
    private fun gemv_truth(
        alpha: Float,
        a: List<List<Float>>,
        x: List<Float>,
        beta: Float,
        y: FloatArray,
    ) {
        val m = a.size
        val n = x.size
        for (i in 0 until m) {
            y[i] *= beta
        }
        for (i in 0 until m) {
            for (j in 0 until n) {
                y[i] += alpha * a[i][j] * x[j]
            }
        }
    }

    private fun gemv_truth_double(
        alpha: Double,
        a: List<List<Double>>,
        x: List<Double>,
        beta: Double,
        y: DoubleArray,
    ) {
        val m = a.size
        val n = x.size
        for (i in 0 until m) {
            y[i] *= beta
        }
        for (i in 0 until m) {
            for (j in 0 until n) {
                y[i] += alpha * a[i][j] * x[j]
            }
        }
    }

    @Test
    fun test_sgemv() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()
        val m = 2
        val n = 5
        val a =
            listOf(
                listOf(0.9314776f, 0.10300648f, -0.620774f, 1.5270752f, 0.0259804f),
                listOf(0.16820757f, -0.94463515f, -1.3850101f, 1.0600523f, 1.5124008f),
            )
        val b = listOf(-1.3441996f, 1.3965541f, -0.89106345f, 0.21196432f, -0.95535654f)
        val c = floatArrayOf(1.0f, 1.0f)
        gemv_truth(1.0f, a, b, 0.0f, c)

        val aFlat =
            listOf(
                0.9314776f,
                0.10300648f,
                -0.620774f,
                1.5270752f,
                0.0259804f,
                0.16820757f,
                -0.94463515f,
                -1.3850101f,
                1.0600523f,
                1.5124008f,
            )
        val aDev = stream.cloneHtod(aFlat).getOrThrow()
        val bDev = stream.cloneHtod(b).getOrThrow()
        val cDev = stream.allocZeros<Float>(m).getOrThrow()

        blas
            .gemvFloat(
                GemvConfig(
                    trans = CublasOperation.CUBLAS_OP_T,
                    m = n,
                    n = m,
                    alpha = 1.0f,
                    lda = n,
                    incx = 1,
                    beta = 0.0f,
                    incy = 1,
                ),
                aDev,
                bDev,
                cDev,
            ).getOrThrow()

        val cHost = stream.cloneDtoh(cDev).getOrThrow()
        for (i in 0 until m) {
            assertTrue(kotlin.math.abs(cHost[i] - c[i]) <= 1e-6f)
        }
    }

    @Test
    fun test_dgemv() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()
        val m = 2
        val n = 5
        val a =
            listOf(
                listOf(0.9314776, 0.10300648, -0.620774, 1.5270752, 0.0259804),
                listOf(0.16820757, -0.94463515, -1.3850101, 1.0600523, 1.5124008),
            )
        val b = listOf(-1.3441996, 1.3965541, -0.89106345, 0.21196432, -0.95535654)
        val c = doubleArrayOf(1.0, 1.0)
        gemv_truth_double(1.0, a, b, 0.0, c)

        val aFlat =
            listOf(
                0.9314776,
                0.10300648,
                -0.620774,
                1.5270752,
                0.0259804,
                0.16820757,
                -0.94463515,
                -1.3850101,
                1.0600523,
                1.5124008,
            )
        val aDev = stream.cloneHtod(aFlat).getOrThrow()
        val bDev = stream.cloneHtod(b).getOrThrow()
        val cDev = stream.allocZeros<Double>(m).getOrThrow()

        blas
            .gemvDouble(
                GemvConfig(
                    trans = CublasOperation.CUBLAS_OP_T,
                    m = n,
                    n = m,
                    alpha = 1.0,
                    lda = n,
                    incx = 1,
                    beta = 0.0,
                    incy = 1,
                ),
                aDev,
                bDev,
                cDev,
            ).getOrThrow()

        val cHost = stream.cloneDtoh(cDev).getOrThrow()
        for (i in 0 until m) {
            assertTrue(kotlin.math.abs(cHost[i] - c[i]) <= 1e-6)
        }
    }
}
