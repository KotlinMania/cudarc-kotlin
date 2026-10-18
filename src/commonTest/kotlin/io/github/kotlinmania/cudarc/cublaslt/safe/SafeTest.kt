// port-lint: tests cublaslt/safe.rs
package io.github.kotlinmania.cudarc.cublaslt.safe

import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertTrue

class SafeTest {
    private fun matmul_truth(
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

    @Test
    fun test_matmul_f32() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlasLT.new(stream).getOrThrow()
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
        matmul_truth(1.0f, a, b, 0.0f, c)

        val aFlat = a.flatten()
        val bFlat = b.flatten()
        val aDev = stream.cloneHtod(aFlat).getOrThrow()
        val bDev = stream.cloneHtod(bFlat).getOrThrow()
        val cDev = stream.allocZeros<Float>(m * n).getOrThrow()

        blas
            .matmul(
                MatmulConfig(
                    transa = false,
                    transb = false,
                    transc = false,
                    m = n.toLong(),
                    n = m.toLong(),
                    k = k.toLong(),
                    alpha = 1.0f,
                    lda = n.toLong(),
                    ldb = k.toLong(),
                    beta = 0.0f,
                    ldc = n.toLong(),
                ),
                bDev,
                aDev,
                cDev,
            ).getOrThrow()

        val cHost = stream.cloneDtoh(cDev).getOrThrow()
        for (row in 0 until m) {
            for (col in 0 until n) {
                val found = cHost[row * n + col]
                val expected = c[row][col]
                assertTrue((found - expected) <= 1e-5f, "found=$found, expected=$expected")
            }
        }
    }

    @Test
    fun test_matmul_half() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlasLT.new(stream).getOrThrow()
        assertTrue(blas.handle.handle > 0L)
    }
}
