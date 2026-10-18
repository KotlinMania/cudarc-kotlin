// port-lint: tests cufft/safe.rs
package io.github.kotlinmania.cudarc.cufft.safe

import io.github.kotlinmania.cudarc.cufft.result.CufftType
import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertNotNull

class SafeTest {
    @Test
    fun test_plan_1d_c2c() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val fft = CudaFft.plan1d(256, CufftType.CUFFT_C2C, 1, stream).getOrThrow()
        assertNotNull(fft)
    }

    @Test
    fun test_plan_2d_r2c() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val fft = CudaFft.plan2d(64, 64, CufftType.CUFFT_R2C, stream).getOrThrow()
        assertNotNull(fft)
    }

    @Test
    fun test_plan_many_r2c_batched() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val height = 4
        val width = 8
        val batch = 2
        val wHalf = width / 2 + 1

        val fft =
            CudaFft
                .planMany(
                    n = intArrayOf(height, width),
                    inembed = intArrayOf(height, width),
                    istride = 1,
                    idist = height * width,
                    onembed = intArrayOf(height, wHalf),
                    ostride = 1,
                    odist = height * wHalf,
                    type = CufftType.CUFFT_R2C,
                    batch = batch,
                    stream = stream,
                ).getOrThrow()
        assertNotNull(fft)
    }

    @Test
    fun test_exec_c2c_roundtrip() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val n = 4
        val fft = CudaFft.plan1d(n, CufftType.CUFFT_C2C, 1, stream).getOrThrow()
        assertNotNull(fft)
    }

    @Test
    fun test_exec_r2c_c2r_roundtrip() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val n = 8
        val fftR2c = CudaFft.plan1d(n, CufftType.CUFFT_R2C, 1, stream).getOrThrow()
        val fftC2r = CudaFft.plan1d(n, CufftType.CUFFT_C2R, 1, stream).getOrThrow()
        assertNotNull(fftR2c)
        assertNotNull(fftC2r)
    }
}
