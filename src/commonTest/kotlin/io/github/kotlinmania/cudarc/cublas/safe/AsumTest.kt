// port-lint: tests cublas/safe/asum.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertTrue

class AsumTest {
    private fun dasum_truth(a: List<Double>, c: DoubleArray, n: Int, incx: Int) {
        c[0] = 0.0
        var i = 0
        var count = 0
        while (count < n && i < a.size) {
            c[0] += kotlin.math.abs(a[i])
            i += incx
            count++
        }
    }

    private fun sasum_truth(a: List<Float>, c: FloatArray, n: Int, incx: Int) {
        c[0] = 0.0f
        var i = 0
        var count = 0
        while (count < n && i < a.size) {
            c[0] += kotlin.math.abs(a[i])
            i += incx
            count++
        }
    }

    @Test
    fun test_sasum() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()

        data class Sample(
            val x: List<Float>,
            val initialResult: Float,
            val n: Int,
            val incx: Int,
        )
        val samples =
            listOf(
                Sample(listOf(-0.5944882f, 1.8055636f, 0.52204555f, -0.00397902f), 0.0f, 4, 1),
                Sample(listOf(-0.5944882f, 1.8055636f, 0.52204555f, -0.00397902f), -1.0f, 4, 1),
                Sample(listOf(-0.5944882f, 1.8055636f, 0.52204555f, -0.00397902f), 3.0f, 4, 1),
                Sample(listOf(-0.38346434f, -0.38013917f, 0.4198623f), 0.0f, 3, 1),
                Sample(listOf(-1.6661372f, -0.4568837f, -0.9043474f, 0.39125723f, 0.39125723f), 0.0f, 5, 1),
            )

        for (sample in samples) {
            val actual = floatArrayOf(sample.initialResult)
            val expected = floatArrayOf(sample.initialResult)
            sasum_truth(sample.x, actual, sample.n, sample.incx)

            val xDev = stream.cloneHtod(sample.x).getOrThrow()
            blas
                .asumFloat(
                    AsumConfig(n = sample.n, incx = sample.incx),
                    xDev,
                    expected,
                ).getOrThrow()

            val delta = kotlin.math.abs(actual[0] - expected[0])
            val epsilon = 2.0f * 1.1920929e-7f
            assertTrue(
                delta <= epsilon,
                "(${actual[0]} - ${expected[0]}).abs() -> $delta <= $epsilon",
            )
        }
    }

    @Test
    fun test_dasum() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val blas = CudaBlas.new(stream).getOrThrow()

        data class Sample(
            val x: List<Double>,
            val initialResult: Double,
            val n: Int,
            val incx: Int,
        )
        val samples =
            listOf(
                Sample(listOf(-0.5944882, 1.8055636, 0.52204555, -0.00397902), 0.0, 4, 1),
                Sample(listOf(-0.5944882, 1.8055636, 0.52204555, -0.00397902), -1.0, 4, 1),
                Sample(listOf(-0.5944882, 1.8055636, 0.52204555, -0.00397902), 3.0, 4, 1),
                Sample(listOf(-0.38346434, -0.38013917, 0.4198623), 0.0, 3, 1),
                Sample(listOf(-1.6661372, -0.4568837, -0.9043474, 0.39125723, 0.39125723), 0.0, 5, 1),
            )

        for (sample in samples) {
            val actual = doubleArrayOf(sample.initialResult)
            val expected = doubleArrayOf(sample.initialResult)
            dasum_truth(sample.x, actual, sample.n, sample.incx)

            val xDev = stream.cloneHtod(sample.x).getOrThrow()
            blas
                .asumDouble(
                    AsumConfig(n = sample.n, incx = sample.incx),
                    xDev,
                    expected,
                ).getOrThrow()

            val delta = kotlin.math.abs(actual[0] - expected[0])
            val epsilon = 2.0 * 2.220446049250313e-16
            assertTrue(
                delta <= epsilon,
                "(${actual[0]} - ${expected[0]}).abs() -> $delta <= $epsilon",
            )
        }
    }
}
