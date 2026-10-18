// port-lint: source cublas/safe/grouped_gemm.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasComputeType
import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.cublas.result.CudaDataType
import io.github.kotlinmania.cudarc.cublas.result.cublasGemmGroupedBatchedEx
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import kotlin.Result

interface GroupedGemmDtype {
    fun dataType(): CudaDataType

    fun computeType(): CublasComputeType
}

data class GroupedGemmConfig<C>(
    val transbs: List<CublasOperation>,
    val transas: List<CublasOperation>,
    val ms: List<Int>,
    val ns: List<Int>,
    val ks: List<Int>,
    val alphas: List<C>,
    val betas: List<C>,
    val ldas: List<Int>,
    val ldbs: List<Int>,
    val ldcs: List<Int>,
    val problemSizes: List<Int>,
) {
    fun problemCount(): Int = problemSizes.sum()

    fun groupCount(): Int = problemSizes.size

    fun validate() {
        val count = groupCount()
        check(transbs.size == count) { "transbs.size != groupCount" }
        check(transas.size == count) { "transas.size != groupCount" }
        check(ms.size == count) { "ms.size != groupCount" }
        check(ns.size == count) { "ns.size != groupCount" }
        check(ks.size == count) { "ks.size != groupCount" }
        check(alphas.size == count) { "alphas.size != groupCount" }
        check(betas.size == count) { "betas.size != groupCount" }
        check(ldas.size == count) { "ldas.size != groupCount" }
        check(ldbs.size == count) { "ldbs.size != groupCount" }
        check(ldcs.size == count) { "ldcs.size != groupCount" }
    }
}

interface GroupedGemm<T : GroupedGemmDtype, C> {
    fun <A : DevicePtr<T>, B : DevicePtr<T>, OutC : DevicePtrMut<T>> groupedGemm(
        config: GroupedGemmConfig<C>,
        aSlices: List<A>,
        bSlices: List<B>,
        cSlices: List<OutC>,
    ): Result<Unit>
}

fun <T : GroupedGemmDtype> CudaBlas.groupedGemmFloat(
    dtype: T,
    config: GroupedGemmConfig<Float>,
    aSlices: List<DevicePtr<T>>,
    bSlices: List<DevicePtr<T>>,
    cSlices: List<DevicePtrMut<T>>,
): Result<Unit> {
    config.validate()
    check(aSlices.size == config.problemCount()) { "aSlices.size != problemCount" }
    check(bSlices.size == config.problemCount()) { "bSlices.size != problemCount" }
    check(cSlices.size == config.problemCount()) { "cSlices.size != problemCount" }

    val aPtrs = aSlices.map { it.devicePtr(stream).first }
    val bPtrs = bSlices.map { it.devicePtr(stream).first }
    val cPtrs = cSlices.map { it.devicePtrMut(stream).first }

    return cublasGemmGroupedBatchedEx(
        handle = handle,
        transa = config.transas,
        transb = config.transbs,
        m = config.ms,
        n = config.ns,
        k = config.ks,
        alpha = config.alphas,
        a = aPtrs,
        aType = dtype.dataType(),
        lda = config.ldas,
        b = bPtrs,
        bType = dtype.dataType(),
        ldb = config.ldbs,
        beta = config.betas,
        c = cPtrs,
        cType = dtype.dataType(),
        ldc = config.ldcs,
        groupCount = config.groupCount(),
        groupSize = config.problemSizes,
        computeType = dtype.computeType(),
    )
}
