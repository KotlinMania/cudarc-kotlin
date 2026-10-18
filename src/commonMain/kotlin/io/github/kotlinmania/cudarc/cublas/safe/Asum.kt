// port-lint: source cublas/safe/asum.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.dasum
import io.github.kotlinmania.cudarc.cublas.result.sasum
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.sliceElements
import kotlin.Result

data class AsumConfig(
    val n: Int,
    val incx: Int,
)

interface Asum<T> {
    fun <X : DevicePtr<T>> asum(
        cfg: AsumConfig,
        x: X,
        result: Array<T>,
    ): Result<Unit>
}

fun CudaBlas.asumFloat(
    cfg: AsumConfig,
    x: DevicePtr<Float>,
    result: FloatArray,
): Result<Unit> {
    val xElems = x.sliceElements()
    if (xElems != null) {
        var sum = 0.0f
        var count = 0
        var i = 0
        while (count < cfg.n && i < xElems.size) {
            sum += kotlin.math.abs(xElems[i])
            i += cfg.incx
            count++
        }
        result[0] = sum
        return Result.success(Unit)
    }
    val xPtr = x.devicePtr(stream).first
    return sasum(
        handle = handle,
        n = cfg.n,
        x = xPtr,
        incx = cfg.incx,
        result = result,
    )
}

fun CudaBlas.asumDouble(
    cfg: AsumConfig,
    x: DevicePtr<Double>,
    result: DoubleArray,
): Result<Unit> {
    val xElems = x.sliceElements()
    if (xElems != null) {
        var sum = 0.0
        var count = 0
        var i = 0
        while (count < cfg.n && i < xElems.size) {
            sum += kotlin.math.abs(xElems[i])
            i += cfg.incx
            count++
        }
        result[0] = sum
        return Result.success(Unit)
    }
    val xPtr = x.devicePtr(stream).first
    return dasum(
        handle = handle,
        n = cfg.n,
        x = xPtr,
        incx = cfg.incx,
        result = result,
    )
}
