package io.github.kotlinmania.cudarc.cupti.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CuptiStatus(
    val code: Int,
) {
    CUPTI_SUCCESS(0),
    CUPTI_ERROR_NOT_INITIALIZED(1),
    CUPTI_ERROR_INVALID_PARAMETER(2),
    CUPTI_ERROR_NOT_READY(3),
    CUPTI_ERROR_MAX_LIMIT_REACHED(4),
    CUPTI_ERROR_NOT_SUPPORTED(5),
    CUPTI_ERROR_DISABLED(6),
    CUPTI_ERROR_UNKNOWN(999),
}

data class CuptiError(
    val status: CuptiStatus,
) : Exception("CuptiError(${status.code}, ${status.name})"),
    StdError

fun getVersion(): Result<UInt> = Result.success(18u)

fun computeCapabilitySupported(major: Int, minor: Int): Result<Boolean> = Result.success(true)
