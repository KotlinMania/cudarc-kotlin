// port-lint: tests driver/result.rs
package io.github.kotlinmania.cudarc.driver.result

import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import io.github.kotlinmania.cudarc.driver.safe.CudaSlice
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull

class ResultTest {
    @Test
    fun peer_transfer_contexts() {
        val ctx1 = CudaContext.new(0).getOrThrow()
        if (device.getCount().getOrThrow() < 2) {
            return
        }
        val stream1 = ctx1.defaultStream()
        val a: CudaSlice<Double> = stream1.allocZeros<Double>(10).getOrThrow()

        val ctx2 = CudaContext.new(1).getOrThrow()
        val stream2 = ctx2.defaultStream()
        val b = stream2.cloneDtod(a).getOrThrow()
        val aHost = stream1.cloneDtoh(a).getOrThrow()
        val bHost = stream2.cloneDtoh(b).getOrThrow()
        assertEquals(aHost.size, bHost.size)
    }

    @Test
    fun peer_transfer_self() {
        val ctx1 = CudaContext.new(0).getOrThrow()
        val stream1 = ctx1.defaultStream()
        val a: CudaSlice<Double> = stream1.allocZeros<Double>(10).getOrThrow()

        val ctx2 = CudaContext.new(0).getOrThrow()
        val stream2 = ctx2.defaultStream()
        val b = stream2.cloneDtod(a).getOrThrow()
        val aHost = stream1.cloneDtoh(a).getOrThrow()
        val bHost = stream2.cloneDtoh(b).getOrThrow()
        assertEquals(aHost.size, bHost.size)
    }

    @Test
    fun re_associate_context_for_memory_op() {
        val ctx1 = CudaContext.new(0).getOrThrow()
        if (device.getCount().getOrThrow() < 2) {
            return
        }
        val stream1 = ctx1.defaultStream()
        val a: CudaSlice<Double> = stream1.allocZeros<Double>(10).getOrThrow()

        val ctx2 = CudaContext.new(1).getOrThrow()
        assertNotNull(ctx2)

        stream1.cloneDtoh(a).getOrThrow()
    }
}
