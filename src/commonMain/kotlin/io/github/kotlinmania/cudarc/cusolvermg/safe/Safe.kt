// port-lint: source cusolvermg/safe.rs
package io.github.kotlinmania.cudarc.cusolvermg.safe

import io.github.kotlinmania.cudarc.cusolvermg.result.CudaLibMgGrid
import io.github.kotlinmania.cudarc.cusolvermg.result.CusolverMgGridMapping
import io.github.kotlinmania.cudarc.cusolvermg.result.CusolverMgHandle
import io.github.kotlinmania.cudarc.cusolvermg.result.mgCreate
import io.github.kotlinmania.cudarc.cusolvermg.result.mgCreateDeviceGrid
import io.github.kotlinmania.cudarc.cusolvermg.result.mgDestroy
import io.github.kotlinmania.cudarc.cusolvermg.result.mgDestroyGrid
import io.github.kotlinmania.cudarc.cusolvermg.result.mgDeviceSelect
import kotlin.Result

class MgHandle(
    val handle: CusolverMgHandle,
) {
    companion object {
        fun new(): Result<MgHandle> =
            mgCreate().map { MgHandle(it) }
    }

    fun deviceSelect(devices: List<Int>): Result<Unit> =
        mgDeviceSelect(handle, devices)

    fun cu(): CusolverMgHandle = handle

    fun close() {
        mgDestroy(handle)
    }
}

class Grid(
    val grid: CudaLibMgGrid,
) {
    companion object {
        fun new(
            colDevices: List<Int>,
            mapping: CusolverMgGridMapping,
        ): Result<Grid> =
            mgCreateDeviceGrid(colDevices, mapping).map { Grid(it) }
    }

    fun close() {
        mgDestroyGrid(grid)
    }
}
