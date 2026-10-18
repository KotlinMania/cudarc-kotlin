// port-lint: tests nccl/safe.rs
package io.github.kotlinmania.cudarc.nccl.safe

import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertEquals

class SafeTest {
    @Test
    fun test_all_reduce() {
        val n = 2
        val nDevices = CudaContext.deviceCount().getOrThrow()
        val id = Id.new().getOrThrow()
        for (i in 0 until nDevices) {
            val ctx = CudaContext.new(i).getOrThrow()
            val stream = ctx.defaultStream()
            val comm = Comm.fromRank(stream, i, nDevices, id).getOrThrow()
            val slice = stream.cloneHtod(List(n) { (i + 1).toFloat() }).getOrThrow()
            val sliceReceive = stream.allocZeros<Float>(n).getOrThrow()
            comm.allReduce(slice, sliceReceive, ReduceOp.Sum).getOrThrow()
            val out = stream.cloneDtoh(sliceReceive).getOrThrow()
            assertEquals(n, out.size)
        }
    }

    @Test
    fun test_all_reduce_views() {
        val n = 2
        val nDevices = CudaContext.deviceCount().getOrThrow()
        val id = Id.new().getOrThrow()
        for (i in 0 until nDevices) {
            val ctx = CudaContext.new(i).getOrThrow()
            val stream = ctx.defaultStream()
            val comm = Comm.fromRank(stream, i, nDevices, id).getOrThrow()
            val slice = stream.cloneHtod(List(n) { (i + 1).toFloat() }).getOrThrow()
            val sliceReceive = stream.allocZeros<Float>(n).getOrThrow()
            val sliceView = slice.slice(0 until n)
            val sliceReceiveView = sliceReceive.sliceMut(0 until n)
            comm.allReduce(sliceView, sliceReceiveView, ReduceOp.Sum).getOrThrow()
            val out = stream.cloneDtoh(sliceReceive).getOrThrow()
            assertEquals(n, out.size)
        }
    }
}
