// port-lint: source cudnn/safe/softmax.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnError
import io.github.kotlinmania.cudarc.cudnn.result.CudnnSoftmaxAlgorithm
import io.github.kotlinmania.cudarc.cudnn.result.CudnnSoftmaxMode
import io.github.kotlinmania.cudarc.cudnn.result.softmaxForward
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import kotlin.Result

data class Softmax<T>(
    val handle: Cudnn,
    val mode: CudnnSoftmaxMode,
    val typeInfo: T,
)

fun <T : CudnnDataTypeInfo> Cudnn.createSoftmax(
    typeInfo: T,
    mode: CudnnSoftmaxMode,
): Result<Softmax<T>> = Result.success(Softmax(this, mode, typeInfo))

data class SoftmaxForward<A : CudnnDataTypeInfo, X : CudnnDataTypeInfo, Y : CudnnDataTypeInfo>(
    val softmax: Softmax<A>,
    val x: TensorDescriptor<X>,
    val y: TensorDescriptor<Y>,
) {
    fun <Src : DevicePtr<Float>, Dst : DevicePtrMut<Float>> launch(
        alpha: Float,
        beta: Float,
        algo: CudnnSoftmaxAlgorithm,
        src: Src,
        dst: Dst,
    ): Result<Unit> {
        val stream = x.handle.stream
        val xPtr = src.devicePtr(stream).first
        val yPtr = dst.devicePtrMut(stream).first
        return softmaxForward(
            handle = softmax.handle.handle,
            algo = algo,
            mode = softmax.mode,
            alpha = alpha,
            xDesc = x.desc,
            x = xPtr,
            beta = beta,
            yDesc = y.desc,
            y = yPtr,
        )
    }
}
