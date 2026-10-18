// port-lint: source cublas/safe/gemm.rs
package io.github.kotlinmania.cudarc.cublas.safe

import io.github.kotlinmania.cudarc.cublas.result.CublasError
import io.github.kotlinmania.cudarc.cublas.result.CublasOperation
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import kotlin.Result

data class GemmConfig<T>(
    val transa: CublasOperation,
    val transb: CublasOperation,
    val m: Int,
    val n: Int,
    val k: Int,
    val alpha: T,
    val lda: Int,
    val ldb: Int,
    val beta: T,
    val ldc: Int,
)

data class StridedBatchedConfig<T>(
    val gemm: GemmConfig<T>,
    val batchSize: Int,
    val strideA: Long,
    val strideB: Long,
    val strideC: Long,
)

interface Gemm<T> {
    fun gemm(
        cfg: GemmConfig<T>,
        a: DevicePtr<T>,
        b: DevicePtr<T>,
        c: DevicePtrMut<T>,
    ): Result<Unit>

    fun gemmStridedBatched(
        cfg: StridedBatchedConfig<T>,
        a: DevicePtr<T>,
        b: DevicePtr<T>,
        c: DevicePtrMut<T>,
    ): Result<Unit>
}
