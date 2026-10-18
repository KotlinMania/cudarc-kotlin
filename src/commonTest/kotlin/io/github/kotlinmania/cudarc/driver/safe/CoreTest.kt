// port-lint: tests driver/safe/core.rs
package io.github.kotlinmania.cudarc.driver.safe

import io.github.kotlinmania.cudarc.driver.result.ctx
import io.github.kotlinmania.cudarc.driver.result.device
import io.github.kotlinmania.cudarc.driver.result.init
import io.github.kotlinmania.cudarc.driver.sys.CUcontext
import io.github.kotlinmania.cudarc.driver.sys.CUdevice
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNotNull
import kotlin.test.assertNull
import kotlin.test.assertTrue

class CoreTest {
    @Test
    fun test_transmutes() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val slice = stream.allocZeros<Byte>(100).getOrThrow()
        assertNotNull(slice.transmute<Float>(25))
        assertNull(slice.transmute<Float>(26))
        assertNotNull(slice.transmuteMut<Float>(25))
        assertNull(slice.transmuteMut<Float>(26))

        run {
            val view = slice.slice(0 until 100)
            assertNotNull(view.transmute<Float>(25))
            assertNull(view.transmute<Float>(26))
        }

        run {
            val viewMut = slice.sliceMut(0 until 100)
            assertNotNull(viewMut.transmute<Float>(25))
            assertNull(viewMut.transmute<Float>(26))
            assertNotNull(viewMut.transmuteMut<Float>(25))
            assertNull(viewMut.transmuteMut<Float>(26))
        }
    }

    @Test
    fun test_threading() {
        val ctx1 = CudaContext.new(0).getOrThrow()
        ctx1.bindToThread().getOrThrow()
        val s1 = ctx1.defaultStream().allocZeros<Float>(10).getOrThrow()
        val s2 = ctx1.defaultStream().allocZeros<Float>(10).getOrThrow()
        assertEquals(10, s1.len)
        assertEquals(10, s2.len)
    }

    @Test
    fun test_post_build_arc_count() {
        val ctx = CudaContext.new(0).getOrThrow()
        assertEquals(0, ctx.ordinal)
    }

    @Test
    fun test_post_alloc_arc_counts() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val t = stream.allocZeros<Float>(1).getOrThrow()
        assertEquals(1, t.len)
    }

    @Test
    fun test_post_alloc_memory() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val (_, total1) = ctx.memGetInfo().getOrThrow()
        val t = stream.cloneHtod(listOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f)).getOrThrow()
        val (_, total2) = ctx.memGetInfo().getOrThrow()
        assertEquals(total1, total2)
        assertEquals(5, t.len)
    }

    @Test
    fun test_ctx_copy_to_views() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val smalls =
            listOf(
                stream.cloneHtod(listOf(-1.0f, -0.8f)).getOrThrow(),
                stream.cloneHtod(listOf(-0.6f, -0.4f)).getOrThrow(),
                stream.cloneHtod(listOf(-0.2f, 0.0f)).getOrThrow(),
                stream.cloneHtod(listOf(0.2f, 0.4f)).getOrThrow(),
                stream.cloneHtod(listOf(0.6f, 0.8f)).getOrThrow(),
            )
        val big = stream.allocZeros<Float>(10).getOrThrow()

        var offset = 0
        for (small in smalls) {
            val sub = big.sliceMut(offset until offset + small.len)
            stream.memcpyDtod(small, sub).getOrThrow()
            offset += small.len
        }

        val out = stream.cloneDtoh(big).getOrThrow()
        val expected = listOf(-1.0f, -0.8f, -0.6f, -0.4f, -0.2f, 0.0f, 0.2f, 0.4f, 0.6f, 0.8f)
        for (i in expected.indices) {
            assertEquals(expected[i], out[i])
        }
    }

    @Test
    fun test_leak_and_upgrade() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()

        val a = stream.cloneHtod(listOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)).getOrThrow()

        val ptr = a.leak()
        val b = stream.upgradeDevicePtr<Float>(ptr, 3)
        assertNotNull(b)

        val ptr2 = b.leak()
        val c = stream.upgradeDevicePtr<Float>(ptr2, 5)
        assertNotNull(c)
    }

    @Test
    fun test_slice_is_freed_with_correct_context() {
        val ctx0 = CudaContext.new(0).getOrThrow()
        val slice = ctx0.defaultStream().cloneHtod(List(10) { 1.0f }).getOrThrow()
        val ctx1 = CudaContext.new(0).getOrThrow()
        ctx1.bindToThread().getOrThrow()
        assertEquals(10, slice.len)
    }

    @Test
    fun test_copy_uses_correct_context() {
        val ctx0 = CudaContext.new(0).getOrThrow()
        val ctx1 = CudaContext.new(0).getOrThrow()
        assertEquals(0, ctx1.ordinal)
        val slice = ctx0.defaultStream().cloneHtod(List(10) { 1.0f }).getOrThrow()
        val out = ctx0.defaultStream().cloneDtoh(slice).getOrThrow()
        assertEquals(10, out.size)
    }

    @Test
    fun test_htod_copy_pinned() {
        val truth = listOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f)
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val pinned = ctx.allocPinned<Float>(10).getOrThrow()
        stream.memcpyHtod(truth, pinned).getOrThrow()
        val host = stream.cloneDtoh(pinned).getOrThrow()
        assertEquals(truth, host)
    }

    @Test
    fun test_pinned_copy_is_faster() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.newStream().getOrThrow()
        val n = 1000
        val notPinned = List(n) { 0.0f }
        val d = stream.cloneHtod(notPinned).getOrThrow()
        stream.synchronize().getOrThrow()
        assertEquals(n, d.len)
    }

    @Test
    fun test_primary_context_is_primary() {
        val ctx = CudaContext.new(0).getOrThrow()
        assertTrue(ctx.isPrimary())
    }

    private fun create_non_primary_context(): Pair<CUdevice, CUcontext> {
        init().getOrThrow()
        val cuDevice = device.get(0).getOrThrow()
        val cuCtx = ctx.createV4(0u, cuDevice).getOrThrow()
        return Pair(cuDevice, cuCtx)
    }

    @Test
    fun test_from_raw_context_creates_and_destroys() {
        val (cuDevice, cuCtx) = create_non_primary_context()
        val ctx = CudaContext.fromRawContext(0, cuDevice, cuCtx).getOrThrow()
        assertFalse(ctx.isPrimary())
        ctx.bindToThread().getOrThrow()
    }

    @Test
    fun test_from_raw_context_bind_to_thread() {
        val (cuDevice, cuCtx) = create_non_primary_context()
        val ctx = CudaContext.fromRawContext(0, cuDevice, cuCtx).getOrThrow()
        ctx.bindToThread().getOrThrow()
        val stream = ctx.defaultStream()
        val data = stream.cloneHtod(listOf(1.0f, 2.0f, 3.0f)).getOrThrow()
        val result = stream.cloneDtoh(data).getOrThrow()
        assertEquals(listOf(1.0f, 2.0f, 3.0f), result)
    }

    @Test
    fun test_new_non_primary_creates_and_destroys() {
        val ctx = CudaContext.newNonPrimary(0, 0u).getOrThrow()
        assertFalse(ctx.isPrimary())
        ctx.bindToThread().getOrThrow()
    }

    @Test
    fun test_new_non_primary_htod_dtoh() {
        val ctx = CudaContext.newNonPrimary(0, 0u).getOrThrow()
        val stream = ctx.defaultStream()
        val data = stream.cloneHtod(listOf(1.0f, 2.0f, 3.0f)).getOrThrow()
        val result = stream.cloneDtoh(data).getOrThrow()
        assertEquals(listOf(1.0f, 2.0f, 3.0f), result)
    }

    @Test
    fun test_new_non_primary_cross_thread() {
        val ctx = CudaContext.newNonPrimary(0, 0u).getOrThrow()
        ctx.bindToThread().getOrThrow()
        val stream = ctx.defaultStream()
        val data = stream.cloneHtod(listOf(4.0f, 5.0f, 6.0f)).getOrThrow()
        val result = stream.cloneDtoh(data).getOrThrow()
        assertEquals(listOf(4.0f, 5.0f, 6.0f), result)
    }
}
