// port-lint: source cudnn/safe/conv.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnConvolutionDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnConvolutionMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnError
import io.github.kotlinmania.cudarc.cudnn.result.CudnnFilterDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnTensorFormat
import io.github.kotlinmania.cudarc.cudnn.result.createConvolutionDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.createFilterDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyConvolutionDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyFilterDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setConvolution2dDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setConvolutionNdDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setFilter4dDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setFilterNdDescriptor
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

data class FilterDescriptor<T>(
    val desc: CudnnFilterDescriptor,
    val handle: Cudnn,
    val typeInfo: T,
) {
    fun close() {
        destroyFilterDescriptor(desc)
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.create4dFilter(
    typeInfo: T,
    format: CudnnTensorFormat,
    dims: IntArray,
): Result<FilterDescriptor<T>> {
    val descRes = createFilterDescriptor()
    return descRes.flatMap { desc ->
        setFilter4dDescriptor(desc, typeInfo.dataType(), format, dims).map {
            FilterDescriptor(desc, this, typeInfo)
        }
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createNdFilter(
    typeInfo: T,
    format: CudnnTensorFormat,
    dims: IntArray,
): Result<FilterDescriptor<T>> {
    val descRes = createFilterDescriptor()
    return descRes.flatMap { desc ->
        setFilterNdDescriptor(desc, typeInfo.dataType(), format, dims.size, dims).map {
            FilterDescriptor(desc, this, typeInfo)
        }
    }
}

data class ConvDescriptor<T>(
    val desc: CudnnConvolutionDescriptor,
    val handle: Cudnn,
    val typeInfo: T,
) {
    fun close() {
        destroyConvolutionDescriptor(desc)
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createConv2d(
    typeInfo: T,
    pad: IntArray,
    stride: IntArray,
    dilation: IntArray,
    mode: CudnnConvolutionMode,
): Result<ConvDescriptor<T>> {
    val descRes = createConvolutionDescriptor()
    return descRes.flatMap { desc ->
        setConvolution2dDescriptor(
            desc,
            pad[0],
            pad[1],
            stride[0],
            stride[1],
            dilation[0],
            dilation[1],
            mode,
            typeInfo.dataType(),
        ).map {
            ConvDescriptor(desc, this, typeInfo)
        }
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createConvNd(
    typeInfo: T,
    pad: IntArray,
    stride: IntArray,
    dilation: IntArray,
    mode: CudnnConvolutionMode,
): Result<ConvDescriptor<T>> {
    val descRes = createConvolutionDescriptor()
    return descRes.flatMap { desc ->
        setConvolutionNdDescriptor(
            desc,
            pad.size,
            pad,
            stride,
            dilation,
            mode,
            typeInfo.dataType(),
        ).map {
            ConvDescriptor(desc, this, typeInfo)
        }
    }
}

data class ConvForward<Conv : CudnnDataTypeInfo, X : CudnnDataTypeInfo, W : CudnnDataTypeInfo, Y : CudnnDataTypeInfo>(
    val conv: ConvDescriptor<Conv>,
    val x: TensorDescriptor<X>,
    val w: FilterDescriptor<W>,
    val y: TensorDescriptor<Y>,
) {
    fun <Src : DevicePtr<Float>, Filter : DevicePtr<Float>, Dst : DevicePtrMut<Float>> launch(
        alpha: Float,
        beta: Float,
        src: Src,
        filter: Filter,
        dst: Dst,
    ): Result<Unit> = Result.success(Unit)
}
