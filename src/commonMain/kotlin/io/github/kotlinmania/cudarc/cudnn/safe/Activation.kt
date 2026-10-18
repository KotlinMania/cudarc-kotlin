// port-lint: source cudnn/safe/activation.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnActivationDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnActivationMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnError
import io.github.kotlinmania.cudarc.cudnn.result.CudnnNanPropagation
import io.github.kotlinmania.cudarc.cudnn.result.activationForward
import io.github.kotlinmania.cudarc.cudnn.result.createActivationDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyActivationDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setActivationDescriptor
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

data class ActivationDescriptor<T>(
    val desc: CudnnActivationDescriptor,
    val handle: Cudnn,
    val typeInfo: T,
) {
    fun close() {
        destroyActivationDescriptor(desc)
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createActivation(
    typeInfo: T,
    mode: CudnnActivationMode,
    nanPropagation: CudnnNanPropagation,
    coef: Double,
): Result<ActivationDescriptor<T>> {
    val descRes = createActivationDescriptor()
    return descRes.flatMap { desc ->
        setActivationDescriptor(desc, mode, nanPropagation, coef).map {
            ActivationDescriptor(desc, this, typeInfo)
        }
    }
}

data class ActivationForward<A : CudnnDataTypeInfo, X : CudnnDataTypeInfo, Y : CudnnDataTypeInfo>(
    val act: ActivationDescriptor<A>,
    val x: TensorDescriptor<X>,
    val y: TensorDescriptor<Y>,
) {
    fun <Src : DevicePtr<Float>, Dst : DevicePtrMut<Float>> launch(
        alpha: Float,
        beta: Float,
        src: Src,
        dst: Dst,
    ): Result<Unit> {
        val stream = act.handle.stream
        val xPtr = src.devicePtr(stream).first
        val yPtr = dst.devicePtrMut(stream).first
        return activationForward(
            handle = act.handle.handle,
            activationDesc = act.desc,
            alpha = alpha,
            xDesc = x.desc,
            x = xPtr,
            beta = beta,
            yDesc = y.desc,
            y = yPtr,
        )
    }
}
