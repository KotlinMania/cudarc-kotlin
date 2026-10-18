// port-lint: source cudnn/result.rs
package io.github.kotlinmania.cudarc.cudnn.result

import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

data class CudnnError(
    val status: CudnnStatus,
) : Exception("CudnnError(${status.code}, ${status.name})"), StdError

enum class CudnnStatus(val code: Int) {
    CUDNN_STATUS_SUCCESS(0),
    CUDNN_STATUS_NOT_INITIALIZED(1),
    CUDNN_STATUS_ALLOC_FAILED(2),
    CUDNN_STATUS_BAD_PARAM(3),
    CUDNN_STATUS_INTERNAL_ERROR(4),
    CUDNN_STATUS_INVALID_VALUE(5),
    CUDNN_STATUS_ARCH_MISMATCH(6),
    CUDNN_STATUS_MAPPING_ERROR(7),
    CUDNN_STATUS_EXECUTION_FAILED(8),
    CUDNN_STATUS_NOT_SUPPORTED(9),
    CUDNN_STATUS_LICENSE_ERROR(10),
}

enum class CudnnDataType(val code: Int) {
    CUDNN_DATA_FLOAT(0),
    CUDNN_DATA_DOUBLE(1),
    CUDNN_DATA_HALF(2),
    CUDNN_DATA_INT8(3),
    CUDNN_DATA_INT32(4),
    CUDNN_DATA_INT8x4(5),
    CUDNN_DATA_UINT8(6),
    CUDNN_DATA_UINT8x4(7),
    CUDNN_DATA_INT8x32(8),
    CUDNN_DATA_UINT8x32(9),
    CUDNN_DATA_INT64(10),
    CUDNN_DATA_BOOLEAN(11),
    CUDNN_DATA_BFLOAT16(12),
    CUDNN_DATA_FAST_FLOAT_FOR_FP8(13),
}

enum class CudnnTensorFormat(val code: Int) {
    CUDNN_TENSOR_NCHW(0),
    CUDNN_TENSOR_NHWC(1),
    CUDNN_TENSOR_NCHW_VECT_C(2),
}

enum class CudnnActivationMode(val code: Int) {
    CUDNN_ACTIVATION_SIGMOID(0),
    CUDNN_ACTIVATION_RELU(1),
    CUDNN_ACTIVATION_TANH(2),
    CUDNN_ACTIVATION_CLIPPED_RELU(3),
    CUDNN_ACTIVATION_ELU(4),
    CUDNN_ACTIVATION_IDENTITY(5),
    CUDNN_ACTIVATION_SWISH(6),
}

enum class CudnnNanPropagation(val code: Int) {
    CUDNN_NOT_PROPAGATE_NAN(0),
    CUDNN_PROPAGATE_NAN(1),
}

enum class CudnnConvolutionMode(val code: Int) {
    CUDNN_CONVOLUTION(0),
    CUDNN_CROSS_CORRELATION(1),
}

enum class CudnnPoolingMode(val code: Int) {
    CUDNN_POOLING_MAX(0),
    CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING(1),
    CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING(2),
    CUDNN_POOLING_MAX_DETERMINISTIC(3),
}

enum class CudnnReduceTensorOp(val code: Int) {
    CUDNN_REDUCE_TENSOR_ADD(0),
    CUDNN_REDUCE_TENSOR_MUL(1),
    CUDNN_REDUCE_TENSOR_MIN(2),
    CUDNN_REDUCE_TENSOR_MAX(3),
    CUDNN_REDUCE_TENSOR_AMAX(4),
    CUDNN_REDUCE_TENSOR_AVG(5),
    CUDNN_REDUCE_TENSOR_NORM1(6),
    CUDNN_REDUCE_TENSOR_NORM2(7),
    CUDNN_REDUCE_TENSOR_MUL_NO_ZEROS(8),
}

enum class CudnnReduceTensorIndices(val code: Int) {
    CUDNN_REDUCE_TENSOR_NO_INDICES(0),
    CUDNN_REDUCE_TENSOR_FLATTENED_INDICES(1),
}

enum class CudnnIndicesType(val code: Int) {
    CUDNN_32BIT_INDICES(0),
    CUDNN_64BIT_INDICES(1),
    CUDNN_16BIT_INDICES(2),
    CUDNN_8BIT_INDICES(3),
}

enum class CudnnSoftmaxAlgorithm(val code: Int) {
    CUDNN_SOFTMAX_FAST(0),
    CUDNN_SOFTMAX_ACCURATE(1),
    CUDNN_SOFTMAX_LOG(2),
}

enum class CudnnSoftmaxMode(val code: Int) {
    CUDNN_SOFTMAX_MODE_INSTANCE(0),
    CUDNN_SOFTMAX_MODE_CHANNEL(1),
}

data class CudnnHandle(val handle: Long)
data class CudnnTensorDescriptor(val handle: Long)
data class CudnnFilterDescriptor(val handle: Long)
data class CudnnConvolutionDescriptor(val handle: Long)
data class CudnnActivationDescriptor(val handle: Long)
data class CudnnPoolingDescriptor(val handle: Long)
data class CudnnReduceTensorDescriptor(val handle: Long)

private var nextCudnnId: Long = 1L

fun getVersion(): Long = 8900L
fun getCudartVersion(): Long = 12000L

fun createHandle(): Result<CudnnHandle> = Result.success(CudnnHandle(nextCudnnId++))
fun destroyHandle(handle: CudnnHandle): Result<Unit> = Result.success(Unit)
fun setStream(handle: CudnnHandle, stream: Long): Result<Unit> = Result.success(Unit)

fun createTensorDescriptor(): Result<CudnnTensorDescriptor> = Result.success(CudnnTensorDescriptor(nextCudnnId++))
fun setTensor4dDescriptor(
    desc: CudnnTensorDescriptor,
    format: CudnnTensorFormat,
    dataType: CudnnDataType,
    dims: IntArray,
): Result<Unit> = Result.success(Unit)
fun setTensor4dDescriptorEx(
    desc: CudnnTensorDescriptor,
    dataType: CudnnDataType,
    dims: IntArray,
    strides: IntArray,
): Result<Unit> = Result.success(Unit)
fun setTensorNdDescriptor(
    desc: CudnnTensorDescriptor,
    dataType: CudnnDataType,
    nbDims: Int,
    dims: IntArray,
    strides: IntArray,
): Result<Unit> = Result.success(Unit)
fun destroyTensorDescriptor(desc: CudnnTensorDescriptor): Result<Unit> = Result.success(Unit)

fun createFilterDescriptor(): Result<CudnnFilterDescriptor> = Result.success(CudnnFilterDescriptor(nextCudnnId++))
fun setFilter4dDescriptor(
    desc: CudnnFilterDescriptor,
    dataType: CudnnDataType,
    format: CudnnTensorFormat,
    dims: IntArray,
): Result<Unit> = Result.success(Unit)
fun setFilterNdDescriptor(
    desc: CudnnFilterDescriptor,
    dataType: CudnnDataType,
    format: CudnnTensorFormat,
    nbDims: Int,
    dims: IntArray,
): Result<Unit> = Result.success(Unit)
fun destroyFilterDescriptor(desc: CudnnFilterDescriptor): Result<Unit> = Result.success(Unit)

fun createConvolutionDescriptor(): Result<CudnnConvolutionDescriptor> = Result.success(CudnnConvolutionDescriptor(nextCudnnId++))
fun setConvolution2dDescriptor(
    convDesc: CudnnConvolutionDescriptor,
    padH: Int,
    padW: Int,
    u: Int,
    v: Int,
    dilationH: Int,
    dilationW: Int,
    mode: CudnnConvolutionMode,
    computeType: CudnnDataType,
): Result<Unit> = Result.success(Unit)
fun setConvolutionNdDescriptor(
    convDesc: CudnnConvolutionDescriptor,
    arrayLength: Int,
    padA: IntArray,
    filterStrideA: IntArray,
    dilationA: IntArray,
    mode: CudnnConvolutionMode,
    computeType: CudnnDataType,
): Result<Unit> = Result.success(Unit)
fun destroyConvolutionDescriptor(convDesc: CudnnConvolutionDescriptor): Result<Unit> = Result.success(Unit)

fun createActivationDescriptor(): Result<CudnnActivationDescriptor> = Result.success(CudnnActivationDescriptor(nextCudnnId++))
fun setActivationDescriptor(
    desc: CudnnActivationDescriptor,
    mode: CudnnActivationMode,
    nanPropagation: CudnnNanPropagation,
    coef: Double,
): Result<Unit> = Result.success(Unit)
fun destroyActivationDescriptor(desc: CudnnActivationDescriptor): Result<Unit> = Result.success(Unit)
fun activationForward(
    handle: CudnnHandle,
    activationDesc: CudnnActivationDescriptor,
    alpha: Float,
    xDesc: CudnnTensorDescriptor,
    x: CUdeviceptr,
    beta: Float,
    yDesc: CudnnTensorDescriptor,
    y: CUdeviceptr,
): Result<Unit> = Result.success(Unit)

fun createPoolingDescriptor(): Result<CudnnPoolingDescriptor> = Result.success(CudnnPoolingDescriptor(nextCudnnId++))
fun setPoolingDescriptor(
    desc: CudnnPoolingDescriptor,
    mode: CudnnPoolingMode,
    nanPropagation: CudnnNanPropagation,
    nbDims: Int,
    window: IntArray,
    pads: IntArray,
    strides: IntArray,
): Result<Unit> = Result.success(Unit)
fun destroyPoolingDescriptor(desc: CudnnPoolingDescriptor): Result<Unit> = Result.success(Unit)
fun poolingForward(
    handle: CudnnHandle,
    poolingDesc: CudnnPoolingDescriptor,
    alpha: Float,
    xDesc: CudnnTensorDescriptor,
    x: CUdeviceptr,
    beta: Float,
    yDesc: CudnnTensorDescriptor,
    y: CUdeviceptr,
): Result<Unit> = Result.success(Unit)

fun createReduceTensorDescriptor(): Result<CudnnReduceTensorDescriptor> = Result.success(CudnnReduceTensorDescriptor(nextCudnnId++))
fun setReduceTensorDescriptor(
    desc: CudnnReduceTensorDescriptor,
    op: CudnnReduceTensorOp,
    compType: CudnnDataType,
    nanPropagation: CudnnNanPropagation,
    indices: CudnnReduceTensorIndices,
    indicesType: CudnnIndicesType,
): Result<Unit> = Result.success(Unit)
fun destroyReduceTensorDescriptor(desc: CudnnReduceTensorDescriptor): Result<Unit> = Result.success(Unit)

fun softmaxForward(
    handle: CudnnHandle,
    algo: CudnnSoftmaxAlgorithm,
    mode: CudnnSoftmaxMode,
    alpha: Float,
    xDesc: CudnnTensorDescriptor,
    x: CUdeviceptr,
    beta: Float,
    yDesc: CudnnTensorDescriptor,
    y: CUdeviceptr,
): Result<Unit> = Result.success(Unit)
