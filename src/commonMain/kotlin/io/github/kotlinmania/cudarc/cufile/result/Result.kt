// port-lint: source cufile/result.rs
package io.github.kotlinmania.cudarc.cufile.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CufileOpError(val code: Int) {
    CU_FILE_SUCCESS(0),
    CU_FILE_DRIVER_NOT_INITIALIZED(1),
    CU_FILE_DRIVER_INVALID_PROPS(2),
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT(3),
    CU_FILE_DRIVER_VERSION_MISMATCH(4),
    CU_FILE_DRIVER_VERSION_ERROR(5),
    CU_FILE_CUDA_DRIVER_ERROR(6),
    CU_FILE_CUDA_POINTER_INVALID(7),
    CU_FILE_CUDA_MEMORY_TYPE_INVALID(8),
    CU_FILE_CUDA_ERROR(9),
    CU_FILE_INVALID_VALUE(10),
    CU_FILE_MEMORY_ALREADY_REGISTERED(11),
    CU_FILE_MEMORY_NOT_REGISTERED(12),
    CU_FILE_PERMISSION_DENIED(13),
    CU_FILE_DRIVER_ALREADY_OPEN(14),
    CU_FILE_HANDLE_NOT_REGISTERED(15),
    CU_FILE_HANDLE_ALREADY_REGISTERED(16),
    CU_FILE_DEVICE_NOT_FOUND(17),
    CU_FILE_IO_NOT_SUPPORTED(18),
    CU_FILE_INVALID_MAPPING_SIZE(19),
    CU_FILE_INVALID_MAPPING_OFFSET(20),
    CU_FILE_INVALID_FILE_TYPE(21),
    CU_FILE_INVALID_FILE_OPEN_FLAG(22),
    CU_FILE_DIO_NOT_SET(23),
}

data class CufileError(
    val error: CufileOpError,
) : Exception("CufileError(${error.code}, ${error.name})"), StdError

fun driverOpen(): Result<Unit> = Result.success(Unit)

fun driverClose(): Result<Unit> = Result.success(Unit)
