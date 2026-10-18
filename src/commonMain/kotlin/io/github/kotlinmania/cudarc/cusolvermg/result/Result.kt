// port-lint: source cusolvermg/result.rs
package io.github.kotlinmania.cudarc.cusolvermg.result

import io.github.kotlinmania.cudarc.cusolver.result.CusolverStatus
import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

data class CusolverMgError(
    val status: CusolverStatus,
) : Exception("CusolverMgError(${status.code}, ${status.name})"),
    StdError

data class CusolverMgHandle(
    val handle: Long,
)

data class CudaLibMgGrid(
    val handle: Long,
)

enum class CusolverMgGridMapping(
    val code: Int,
) {
    CUDALIBMG_GRID_MAPPING_COL_MAJOR(0),
    CUDALIBMG_GRID_MAPPING_ROW_MAJOR(1),
}

private var nextMgId: Long = 1L

fun mgCreate(): Result<CusolverMgHandle> =
    Result.success(CusolverMgHandle(nextMgId++))

fun mgDestroy(handle: CusolverMgHandle): Result<Unit> =
    Result.success(Unit)

fun mgDeviceSelect(handle: CusolverMgHandle, devices: List<Int>): Result<Unit> =
    Result.success(Unit)

fun mgCreateDeviceGrid(
    colDevices: List<Int>,
    mapping: CusolverMgGridMapping,
): Result<CudaLibMgGrid> =
    Result.success(CudaLibMgGrid(nextMgId++))

fun mgDestroyGrid(grid: CudaLibMgGrid): Result<Unit> =
    Result.success(Unit)
