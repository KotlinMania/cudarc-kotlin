// port-lint: tests curand/safe.rs
package io.github.kotlinmania.cudarc.curand.safe

import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SafeTest {
    private fun gen_uniform(seed: ULong, n: Int): List<Float> {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<Float>(n).getOrThrow()
        val rng = CudaRng.new(seed, stream).getOrThrow()
        rng.fillWithUniform(aDev).getOrThrow()
        return stream.cloneDtoh(aDev).getOrThrow()
    }

    private fun gen_normal(seed: ULong, n: Int, mean: Float, std: Float): List<Float> {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<Float>(n).getOrThrow()
        val rng = CudaRng.new(seed, stream).getOrThrow()
        rng.fillWithNormal(aDev, mean, std).getOrThrow()
        return stream.cloneDtoh(aDev).getOrThrow()
    }

    private fun gen_log_normal(seed: ULong, n: Int, mean: Float, std: Float): List<Float> {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<Float>(n).getOrThrow()
        val rng = CudaRng.new(seed, stream).getOrThrow()
        rng.fillWithLogNormal(aDev, mean, std).getOrThrow()
        return stream.cloneDtoh(aDev).getOrThrow()
    }

    @Test
    fun test_seed_reproducible() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val aDev = stream.allocZeros<Float>(10).getOrThrow()
        val bDev = stream.allocZeros<Float>(10).getOrThrow()

        val aRng = CudaRng.new(0uL, stream).getOrThrow()
        val bRng = CudaRng.new(0uL, stream).getOrThrow()

        aRng.fillWithUniform(aDev).getOrThrow()
        bRng.fillWithUniform(bDev).getOrThrow()

        val aHost = stream.cloneDtoh(aDev).getOrThrow()
        val bHost = stream.cloneDtoh(bDev).getOrThrow()
        assertEquals(aHost, bHost)
    }

    @Test
    fun test_different_seeds_neq() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val aDev = stream.allocZeros<Float>(10).getOrThrow()
        val bDev = stream.allocZeros<Float>(10).getOrThrow()

        val aRng = CudaRng.new(0uL, stream).getOrThrow()
        val bRng = CudaRng.new(1uL, stream).getOrThrow()

        aRng.fillWithUniform(aDev).getOrThrow()
        bRng.fillWithUniform(bDev).getOrThrow()

        val aHost = stream.cloneDtoh(aDev).getOrThrow()
        val bHost = stream.cloneDtoh(bDev).getOrThrow()
        assertNotEquals(aHost, bHost)
    }

    @Test
    fun test_set_offset() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val aDev = stream.allocZeros<Float>(10).getOrThrow()
        val bDev = stream.allocZeros<Float>(10).getOrThrow()

        val aRng = CudaRng.new(0uL, stream).getOrThrow()
        val bRng = CudaRng.new(0uL, stream).getOrThrow()

        aRng.setOffset(10uL).getOrThrow()
        aRng.fillWithUniform(aDev).getOrThrow()
        bRng.fillWithUniform(bDev).getOrThrow()

        val aHost = stream.cloneDtoh(aDev).getOrThrow()
        val bHost = stream.cloneDtoh(bDev).getOrThrow()
        assertNotNull(aHost)
        assertNotNull(bHost)
    }

    private val N = 1000

    @Test
    fun test_uniform_f32() {
        val a = gen_uniform(0uL, N)
        for (i in 0 until N) {
            assertTrue(0.0f < a[i] && a[i] <= 1.0f)
        }
    }

    @Test
    fun test_uniform_f64() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<Double>(N).getOrThrow()
        val rng = CudaRng.new(0uL, stream).getOrThrow()
        rng.fillWithUniformDouble(aDev).getOrThrow()
        val a = stream.cloneDtoh(aDev).getOrThrow()
        for (i in 0 until N) {
            assertTrue(0.0 < a[i] && a[i] <= 1.0)
        }
    }

    @Test
    fun test_uniform_u32() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<UInt>(N).getOrThrow()
        val rng = CudaRng.new(0uL, stream).getOrThrow()
        rng.fillWithUniformUInt(aDev).getOrThrow()
        val a = stream.cloneDtoh(aDev).getOrThrow()
        for (i in 0 until N) {
            assertTrue(a[i] > 0u)
        }
    }

    @Test
    fun test_normal_f32() {
        val a = gen_normal(0uL, N, 0.0f, 1.0f)
        for (i in 0 until N) {
            assertTrue(a[i] != 0.0f)
        }

        val b = gen_normal(0uL, N, -1.0f, 2.0f)
        for (i in 0 until N) {
            assertNotEquals(a[i], b[i])
        }
    }

    @Test
    fun test_normal_f64() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<Double>(N).getOrThrow()
        val rng = CudaRng.new(0uL, stream).getOrThrow()
        rng.fillWithNormalDouble(aDev, 0.0, 1.0).getOrThrow()
        val a = stream.cloneDtoh(aDev).getOrThrow()
        for (i in 0 until N) {
            assertTrue(a[i] != 0.0)
        }

        val bDev = stream.allocZeros<Double>(N).getOrThrow()
        rng.fillWithNormalDouble(bDev, -1.0, 2.0).getOrThrow()
        val b = stream.cloneDtoh(bDev).getOrThrow()
        for (i in 0 until N) {
            assertNotEquals(a[i], b[i])
        }
    }

    @Test
    fun test_log_normal_f32() {
        val a = gen_log_normal(0uL, N, 0.0f, 1.0f)
        for (i in 0 until N) {
            assertTrue(a[i] != 0.0f)
        }

        val b = gen_log_normal(0uL, N, -1.0f, 2.0f)
        for (i in 0 until N) {
            assertNotEquals(a[i], b[i])
        }
    }

    @Test
    fun test_log_normal_f64() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val aDev = stream.allocZeros<Double>(N).getOrThrow()
        val rng = CudaRng.new(0uL, stream).getOrThrow()
        rng.fillWithLogNormalDouble(aDev, 0.0, 1.0).getOrThrow()
        val a = stream.cloneDtoh(aDev).getOrThrow()
        for (i in 0 until N) {
            assertTrue(a[i] != 0.0)
        }

        val bDev = stream.allocZeros<Double>(N).getOrThrow()
        rng.fillWithLogNormalDouble(bDev, -1.0, 2.0).getOrThrow()
        val b = stream.cloneDtoh(bDev).getOrThrow()
        for (i in 0 until N) {
            assertNotEquals(a[i], b[i])
        }
    }
}
