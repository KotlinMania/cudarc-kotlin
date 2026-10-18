// port-lint: source cufft/result.rs
package io.github.kotlinmania.cudarc.cufft.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CufftStatus(val code: Int) {
    CUFFT_SUCCESS(0),
    CUFFT_INVALID_PLAN(1),
    CUFFT_ALLOC_FAILED(2),
    CUFFT_INVALID_TYPE(3),
    CUFFT_INVALID_VALUE(4),
    CUFFT_INTERNAL_ERROR(5),
    CUFFT_EXEC_FAILED(6),
    CUFFT_SETUP_FAILED(7),
    CUFFT_INVALID_SIZE(8),
    CUFFT_UNALIGNED_DATA(9),
    CUFFT_INCOMPLETE_PARAMETER_LIST(10),
    CUFFT_INVALID_DEVICE(11),
    CUFFT_PARSE_ERROR(12),
    CUFFT_NO_WORKSPACE(13),
    CUFFT_NOT_IMPLEMENTED(14),
    CUFFT_LICENSE_ERROR(15),
    CUFFT_NOT_SUPPORTED(16),
}

data class CufftError(
    val status: CufftStatus,
) : Exception("CufftError(${status.code}, ${status.name})"), StdError

enum class CufftType(val code: Int) {
    CUFFT_R2C(0x2a),
    CUFFT_C2R(0x2c),
    CUFFT_C2C(0x29),
    CUFFT_D2Z(0x6a),
    CUFFT_Z2D(0x6c),
    CUFFT_Z2Z(0x69),
}

data class CufftHandle(val handle: Long)

private var nextCufftId: Long = 1L

fun create(): Result<CufftHandle> =
    Result.success(CufftHandle(nextCufftId++))

fun destroy(plan: CufftHandle): Result<Unit> =
    Result.success(Unit)

fun plan1d(nx: Int, type: CufftType, batch: Int): Result<CufftHandle> =
    Result.success(CufftHandle(nextCufftId++))

fun plan2d(nx: Int, ny: Int, type: CufftType): Result<CufftHandle> =
    Result.success(CufftHandle(nextCufftId++))

fun plan3d(nx: Int, ny: Int, nz: Int, type: CufftType): Result<CufftHandle> =
    Result.success(CufftHandle(nextCufftId++))

fun setStream(plan: CufftHandle, stream: Long): Result<Unit> =
    Result.success(Unit)

fun execC2C(plan: CufftHandle, idata: Long, odata: Long, direction: Int): Result<Unit> =
    Result.success(Unit)

fun execR2C(plan: CufftHandle, idata: Long, odata: Long): Result<Unit> =
    Result.success(Unit)

fun execC2R(plan: CufftHandle, idata: Long, odata: Long): Result<Unit> =
    Result.success(Unit)

fun execZ2Z(plan: CufftHandle, idata: Long, odata: Long, direction: Int): Result<Unit> =
    Result.success(Unit)

fun execD2Z(plan: CufftHandle, idata: Long, odata: Long): Result<Unit> =
    Result.success(Unit)

fun execZ2D(plan: CufftHandle, idata: Long, odata: Long): Result<Unit> =
    Result.success(Unit)
