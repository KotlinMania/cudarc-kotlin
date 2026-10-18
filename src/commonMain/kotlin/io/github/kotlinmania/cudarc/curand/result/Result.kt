// port-lint: source curand/result.rs
package io.github.kotlinmania.cudarc.curand.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class CurandStatus(val code: Int, val description: String) {
    CURAND_STATUS_SUCCESS(0, "Success"),
    CURAND_STATUS_VERSION_MISMATCH(100, "Version mismatch"),
    CURAND_STATUS_NOT_INITIALIZED(101, "Not initialized"),
    CURAND_STATUS_ALLOCATION_FAILED(102, "Allocation failed"),
    CURAND_STATUS_TYPE_ERROR(103, "Type error"),
    CURAND_STATUS_OUT_OF_RANGE(104, "Out of range"),
    CURAND_STATUS_LENGTH_NOT_MULTIPLE(105, "Length not multiple"),
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED(106, "Double precision required"),
    CURAND_STATUS_LAUNCH_FAILURE(201, "Launch failure"),
    CURAND_STATUS_PREEXISTING_FAILURE(202, "Preexisting failure"),
    CURAND_STATUS_INITIALIZATION_FAILED(203, "Initialization failed"),
    CURAND_STATUS_ARCH_MISMATCH(204, "Architecture mismatch"),
    CURAND_STATUS_INTERNAL_ERROR(999, "Internal error"),
    ;

    fun result(): Result<Unit> =
        if (this == CURAND_STATUS_SUCCESS) {
            Result.success(Unit)
        } else {
            Result.failure(CurandError(this))
        }

    companion object {
        fun fromCode(code: Int): CurandStatus =
            entries.find { it.code == code } ?: CURAND_STATUS_INTERNAL_ERROR
    }
}

enum class CurandRngType(val code: Int) {
    CURAND_RNG_TEST(0),
    CURAND_RNG_PSEUDO_DEFAULT(100),
    CURAND_RNG_PSEUDO_XORWOW(101),
    CURAND_RNG_PSEUDO_MRG32K3A(121),
    CURAND_RNG_PSEUDO_MTGP32(141),
    CURAND_RNG_PSEUDO_MT19937(142),
    CURAND_RNG_PSEUDO_PHILOX4_32_10(161),
    CURAND_RNG_QUASI_DEFAULT(200),
    CURAND_RNG_QUASI_SOBOL32(201),
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32(202),
    CURAND_RNG_QUASI_SOBOL64(203),
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64(204),
}

data class CurandGenerator(val handle: Long)

data class CurandError(
    val status: CurandStatus,
) : Exception("CurandError(${status.code}, ${status.name})"), StdError

private var nextGenId: Long = 1L
private val generatorSeeds: MutableMap<Long, ULong> = mutableMapOf()
private val generatorOffsets: MutableMap<Long, ULong> = mutableMapOf()

fun createGenerator(kind: CurandRngType = CurandRngType.CURAND_RNG_PSEUDO_DEFAULT): Result<CurandGenerator> {
    val id = nextGenId++
    generatorSeeds[id] = 0uL
    generatorOffsets[id] = 0uL
    return Result.success(CurandGenerator(id))
}

fun setSeed(generator: CurandGenerator, seed: ULong): Result<Unit> {
    generatorSeeds[generator.handle] = seed
    return Result.success(Unit)
}

fun setOffset(generator: CurandGenerator, offset: ULong): Result<Unit> {
    generatorOffsets[generator.handle] = offset
    return Result.success(Unit)
}

fun setStream(generator: CurandGenerator, streamHandle: Long): Result<Unit> =
    Result.success(Unit)

fun destroyGenerator(generator: CurandGenerator): Result<Unit> {
    generatorSeeds.remove(generator.handle)
    generatorOffsets.remove(generator.handle)
    return Result.success(Unit)
}
