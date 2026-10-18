// port-lint: source cudnn/safe/reduce.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnError
import io.github.kotlinmania.cudarc.cudnn.result.CudnnIndicesType
import io.github.kotlinmania.cudarc.cudnn.result.CudnnNanPropagation
import io.github.kotlinmania.cudarc.cudnn.result.CudnnReduceTensorDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.CudnnReduceTensorIndices
import io.github.kotlinmania.cudarc.cudnn.result.CudnnReduceTensorOp
import io.github.kotlinmania.cudarc.cudnn.result.createReduceTensorDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.destroyReduceTensorDescriptor
import io.github.kotlinmania.cudarc.cudnn.result.setReduceTensorDescriptor
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.flatMap
import kotlin.Result

object FlatIndices
object NoIndices

data class ReductionDescriptor<T, Idx>(
    val desc: CudnnReduceTensorDescriptor,
    val indices: Idx,
    val handle: Cudnn,
    val typeInfo: T,
) {
    fun close() {
        destroyReduceTensorDescriptor(desc)
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createReductionFlatIndices(
    typeInfo: T,
    op: CudnnReduceTensorOp,
    nanOpt: CudnnNanPropagation,
): Result<ReductionDescriptor<T, FlatIndices>> {
    val descRes = createReduceTensorDescriptor()
    return descRes.flatMap { desc ->
        setReduceTensorDescriptor(
            desc,
            op,
            typeInfo.dataType(),
            nanOpt,
            CudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_FLATTENED_INDICES,
            CudnnIndicesType.CUDNN_32BIT_INDICES,
        ).map {
            ReductionDescriptor(desc, FlatIndices, this, typeInfo)
        }
    }
}

fun <T : CudnnDataTypeInfo> Cudnn.createReductionNoIndices(
    typeInfo: T,
    op: CudnnReduceTensorOp,
    nanOpt: CudnnNanPropagation,
): Result<ReductionDescriptor<T, NoIndices>> {
    val descRes = createReduceTensorDescriptor()
    return descRes.flatMap { desc ->
        setReduceTensorDescriptor(
            desc,
            op,
            typeInfo.dataType(),
            nanOpt,
            CudnnReduceTensorIndices.CUDNN_REDUCE_TENSOR_NO_INDICES,
            CudnnIndicesType.CUDNN_32BIT_INDICES,
        ).map {
            ReductionDescriptor(desc, NoIndices, this, typeInfo)
        }
    }
}
