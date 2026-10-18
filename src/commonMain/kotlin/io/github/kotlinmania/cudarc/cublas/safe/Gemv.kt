// port-lint: source cublas/safe/gemv.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasError
import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.cublas.result.dgemv
import io.github.kotlinmania.cudarc.cublas.result.sgemv
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import kotlin.Result

data class GemvConfig<T>(
    val trans: CublasOperation,
    val m: Int,
    val n: Int,
    val alpha: T,
    val lda: Int,
    val incx: Int,
    val beta: T,
    val incy: Int,
)

interface Gemv<T> {
    fun <A : DevicePtr<T>, X : DevicePtr<T>, Y : DevicePtrMut<T>> gemv(
        cfg: GemvConfig<T>,
        a: A,
        x: X,
        y: Y,
    ): Result<Unit>
}

fun CudaBlas.gemvFloat(
    cfg: GemvConfig<Float>,
    a: DevicePtr<Float>,
    x: DevicePtr<Float>,
    y: DevicePtrMut<Float>,
): Result<Unit> {
    val aPtr = a.devicePtr(stream).first
    val xPtr = x.devicePtr(stream).first
    val yPtr = y.devicePtrMut(stream).first
    return sgemv(
        handle = handle,
        trans = cfg.trans,
        m = cfg.m,
        n = cfg.n,
        alpha = cfg.alpha,
        a = aPtr,
        lda = cfg.lda,
        x = xPtr,
        incx = cfg.incx,
        beta = cfg.beta,
        y = yPtr,
        incy = cfg.incy,
    )
}

fun CudaBlas.gemvDouble(
    cfg: GemvConfig<Double>,
    a: DevicePtr<Double>,
    x: DevicePtr<Double>,
    y: DevicePtrMut<Double>,
): Result<Unit> {
    val aPtr = a.devicePtr(stream).first
    val xPtr = x.devicePtr(stream).first
    val yPtr = y.devicePtrMut(stream).first
    return dgemv(
        handle = handle,
        trans = cfg.trans,
        m = cfg.m,
        n = cfg.n,
        alpha = cfg.alpha,
        a = aPtr,
        lda = cfg.lda,
        x = xPtr,
        incx = cfg.incx,
        beta = cfg.beta,
        y = yPtr,
        incy = cfg.incy,
    )
}
