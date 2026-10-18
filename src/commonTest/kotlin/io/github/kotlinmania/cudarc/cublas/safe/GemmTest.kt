// port-lint: tests cublas/safe/gemm.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertTrue

class GemmTest {
    private fun gemm_truth(
        alpha: Float,
        a: List<List<Float>>,
        b: List<List<Float>>,
        beta: Float,
        c: Array<FloatArray>,
    ) {
        val m = a.size
        val k = a[0].size
        val n = b[0].size
        for (i in 0 until m) {
            for (j in 0 until n) {
                c[i][j] *= beta
            }
        }
        for (i in 0 until m) {
            for (j in 0 until n) {
                for (p in 0 until k) {
                    c[i][j] += alpha * a[i][p] * b[p][j]
                }
            }
        }
    }

    private fun gemm_truth_double(
        alpha: Double,
        a: List<List<Double>>,
        b: List<List<Double>>,
        beta: Double,
        c: Array<DoubleArray>,
    ) {
        val m = a.size
        val k = a[0].size
        val n = b[0].size
        for (i in 0 until m) {
            for (j in 0 until n) {
                c[i][j] *= beta
            }
        }
        for (i in 0 until m) {
            for (j in 0 until n) {
                for (p in 0 until k) {
                    c[i][j] += alpha * a[i][p] * b[p][j]
                }
            }
        }
    }

    @Test
    fun test_sgemm() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()
        val m = 3
        val k = 4
        val n = 5
        val a =
            listOf(
                listOf(-0.5944882f, 1.8055636f, 0.52204555f, -0.00397902f),
                listOf(-0.38346434f, -0.38013917f, 0.4198623f, -0.22479166f),
                listOf(-1.6661372f, -0.4568837f, -0.9043474f, 0.39125723f),
            )
        val b =
            listOf(
                listOf(1.1292169f, -0.13450263f, 0.62789696f, -0.5685516f, 0.21946938f),
                listOf(1.0585804f, -0.39789402f, 0.90205914f, 0.989318f, -0.3443096f),
                listOf(1.3412506f, 0.3059701f, -0.9714474f, -0.36113533f, -1.6809629f),
                listOf(3.4746711f, -1.0930681f, 0.16502666f, -0.59988785f, 0.41375792f),
            )
        val c = Array(m) { FloatArray(n) { 0.0f } }
        gemm_truth(1.0f, a, b, 0.0f, c)

        val aFlat =
            listOf(
                -0.5944882f,
                1.8055636f,
                0.52204555f,
                -0.00397902f,
                -0.38346434f,
                -0.38013917f,
                0.4198623f,
                -0.22479166f,
                -1.6661372f,
                -0.4568837f,
                -0.9043474f,
                0.39125723f,
            )
        val bFlat =
            listOf(
                1.1292169f,
                -0.13450263f,
                0.62789696f,
                -0.5685516f,
                0.21946938f,
                1.0585804f,
                -0.39789402f,
                0.90205914f,
                0.989318f,
                -0.3443096f,
                1.3412506f,
                0.3059701f,
                -0.9714474f,
                -0.36113533f,
                -1.6809629f,
                3.4746711f,
                -1.0930681f,
                0.16502666f,
                -0.59988785f,
                0.41375792f,
            )
        val aDev = stream.cloneHtod(aFlat).getOrThrow()
        val bDev = stream.cloneHtod(bFlat).getOrThrow()
        val cDev = stream.allocZeros<Float>(m * n).getOrThrow()

        blas
            .gemmFloat(
                GemmConfig(
                    transa = CublasOperation.CUBLAS_OP_N,
                    transb = CublasOperation.CUBLAS_OP_N,
                    m = n,
                    n = m,
                    k = k,
                    alpha = 1.0f,
                    lda = n,
                    ldb = k,
                    beta = 0.0f,
                    ldc = n,
                ),
                bDev,
                aDev,
                cDev,
            ).getOrThrow()

        val cHost = stream.cloneDtoh(cDev).getOrThrow()
        for (row in 0 until m) {
            for (col in 0 until n) {
                assertTrue(kotlin.math.abs(cHost[row * n + col] - c[row][col]) <= 1e-5f)
            }
        }
    }

    @Test
    fun test_dgemm() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()
        val m = 4
        val k = 3
        val n = 2
        val a =
            listOf(
                listOf(-0.70925030, -1.01357541, -0.64827034),
                listOf(2.18493467, -0.61584842, -1.43844327),
                listOf(-1.34792593, 0.68840750, -0.48057214),
                listOf(1.22180992, 1.16245157, 0.01253436),
            )
        val b =
            listOf(
                listOf(1.1292169, -0.13450263),
                listOf(1.0585804, -0.39789402),
                listOf(1.3412506, 0.3059701),
            )
        val c = Array(m) { DoubleArray(n) { 0.0 } }
        gemm_truth_double(1.0, a, b, 0.0, c)

        val aFlat = a.flatten()
        val bFlat = b.flatten()
        val aDev = stream.cloneHtod(aFlat).getOrThrow()
        val bDev = stream.cloneHtod(bFlat).getOrThrow()
        val cDev = stream.allocZeros<Double>(m * n).getOrThrow()

        blas
            .gemmDouble(
                GemmConfig(
                    transa = CublasOperation.CUBLAS_OP_N,
                    transb = CublasOperation.CUBLAS_OP_N,
                    m = n,
                    n = m,
                    k = k,
                    alpha = 1.0,
                    lda = n,
                    ldb = k,
                    beta = 0.0,
                    ldc = n,
                ),
                bDev,
                aDev,
                cDev,
            ).getOrThrow()

        val cHost = stream.cloneDtoh(cDev).getOrThrow()
        for (row in 0 until m) {
            for (col in 0 until n) {
                assertTrue(kotlin.math.abs(cHost[row * n + col] - c[row][col]) <= 1e-5)
            }
        }
    }

    @Test
    fun test_hgemm() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()
        assertTrue(blas.handle.handle > 0L)
    }
}
