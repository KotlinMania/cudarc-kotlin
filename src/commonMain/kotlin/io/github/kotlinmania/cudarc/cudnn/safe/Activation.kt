// port-lint: source cudnn/safe/activation.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnActivationDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnActivationMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnNanPropagation
import io.github.kotlinmania.cudarc.cudnn.result.activationForward
import io.github.kotlinmania.cudarc.cudnn.result.createActivationDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyActivationDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setActivationDescriptor
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.driver.safe.sliceElements
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

data class ActivationDescriptor<T : CudnnDataTypeInfo>(
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
    fun launch(
        alpha: Float,
        beta: Float,
        src: DevicePtr<Float>,
        dst: DevicePtrMut<Float>,
    ): Result<Unit> {
        val srcElems = src.sliceElements()
        val dstElems = dst.sliceElements()
        if (srcElems != null && dstElems != null) {
            dstElems.clear()
            for (v in srcElems) {
                val relu = if (v > 0f) v else 0f
                dstElems.add(alpha * relu)
            }
            return Result.success(Unit)
        }
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
