// port-lint: source cudnn/safe/core.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnDataType
import io.github.kotlinmania.cudarc.cudnn.result.CudnnError
import io.github.kotlinmania.cudarc.cudnn.result.CudnnHandle
import io.github.kotlinmania.cudarc.cudnn.result.CudnnTensorDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnTensorFormat
import io.github.kotlinmania.cudarc.cudnn.result.createHandle
import io.github.kotlinmania.cudarc.cudnn.result.createTensorDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyHandle
import io.github.kotlinmania.cudarc.cudnn.result.destroyTensorDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setStream
import io.github.kotlinmania.cudarc.cudnn.result.setTensor4dDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setTensor4dDescriptorEx
import io.github.kotlinmania.cudarc.cudnn.result.setTensorNdDescriptor
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

interface CudnnDataTypeInfo {
    fun dataType(): CudnnDataType
    fun intoScalingParameter(value: Float): Float = value
}

class Cudnn(
    val handle: CudnnHandle,
    var stream: CudaStream,
) {
    companion object {
        fun new(stream: CudaStream): Result<Cudnn> {
            val handleResult = createHandle()
            return handleResult.flatMap { h ->
                setStream(h, stream.cuStream.handle).map {
                    Cudnn(h, stream)
                }
            }
        }
    }

    fun setStream(stream: CudaStream): Result<Unit> {
        this.stream = stream
        return setStream(handle, stream.cuStream.handle)
    }

    fun close() {
        destroyHandle(handle)
    }

    fun <T : CudnnDataTypeInfo> create4dTensor(
        typeInfo: T,
        format: CudnnTensorFormat,
        dims: IntArray,
    ): Result<TensorDescriptor<T>> {
        val descRes = createTensorDescriptor()
        return descRes.flatMap { desc ->
            setTensor4dDescriptor(desc, format, typeInfo.dataType(), dims).map {
                TensorDescriptor(desc, this, typeInfo)
            }
        }
    }

    fun <T : CudnnDataTypeInfo> create4dTensorEx(
        typeInfo: T,
        dims: IntArray,
        strides: IntArray,
    ): Result<TensorDescriptor<T>> {
        val descRes = createTensorDescriptor()
        return descRes.flatMap { desc ->
            setTensor4dDescriptorEx(desc, typeInfo.dataType(), dims, strides).map {
                TensorDescriptor(desc, this, typeInfo)
            }
        }
    }

    fun <T : CudnnDataTypeInfo> createNdTensor(
        typeInfo: T,
        dims: IntArray,
        strides: IntArray,
    ): Result<TensorDescriptor<T>> {
        check(dims.size >= 4) { "dims.size must be >= 4" }
        check(dims.size == strides.size) { "dims.size != strides.size" }
        val descRes = createTensorDescriptor()
        return descRes.flatMap { desc ->
            setTensorNdDescriptor(desc, typeInfo.dataType(), dims.size, dims, strides).map {
                TensorDescriptor(desc, this, typeInfo)
            }
        }
    }
}

data class TensorDescriptor<T>(
    val desc: CudnnTensorDescriptor,
    val handle: Cudnn,
    val typeInfo: T,
) {
    fun close() {
        destroyTensorDescriptor(desc)
    }
}
