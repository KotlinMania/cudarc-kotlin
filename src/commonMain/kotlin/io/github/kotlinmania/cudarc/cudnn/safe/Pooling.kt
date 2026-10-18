// port-lint: source cudnn/safe/pooling.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnError
import io.github.kotlinmania.cudarc.cudnn.result.CudnnNanPropagation
import io.github.kotlinmania.cudarc.cudnn.result.CudnnPoolingDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnPoolingMode
import io.github.kotlinmania.cudarc.cudnn.result.createPoolingDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyPoolingDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.poolingForward
import io.github.kotlinmania.cudarc.cudnn.result.setPoolingDescriptor
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

data class PoolingDescriptor<T>(
    val desc: CudnnPoolingDescriptor,
    val handle: Cudnn,
    val typeInfo: T,
) {
    fun close() {
        destroyPoolingDescriptor(desc)
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createPoolingNd(
    typeInfo: T,
    window: IntArray,
    pads: IntArray,
    strides: IntArray,
    mode: CudnnPoolingMode,
    nanPropagation: CudnnNanPropagation,
): Result<PoolingDescriptor<T>> {
    val descRes = createPoolingDescriptor()
    return descRes.flatMap { desc ->
        setPoolingDescriptor(desc, mode, nanPropagation, window.size, window, pads, strides).map {
            PoolingDescriptor(desc, this, typeInfo)
        }
    }
}

data class PoolingForward<P : CudnnDataTypeInfo, X : CudnnDataTypeInfo, Y : CudnnDataTypeInfo>(
    val pooling: PoolingDescriptor<P>,
    val x: TensorDescriptor<X>,
    val y: TensorDescriptor<Y>,
) {
    fun <Src : DevicePtr<Float>, Dst : DevicePtrMut<Float>> launch(
        alpha: Float,
        beta: Float,
        src: Src,
        dst: Dst,
    ): Result<Unit> {
        val stream = x.handle.stream
        val xPtr = src.devicePtr(stream).first
        val yPtr = dst.devicePtrMut(stream).first
        return poolingForward(
            handle = pooling.handle.handle,
            poolingDesc = pooling.desc,
            alpha = alpha,
            xDesc = x.desc,
            x = xPtr,
            beta = beta,
            yDesc = y.desc,
            y = yPtr,
        )
    }
}
