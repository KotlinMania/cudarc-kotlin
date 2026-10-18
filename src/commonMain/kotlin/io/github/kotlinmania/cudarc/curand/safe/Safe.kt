// port-lint: source curand/safe.rs
package io.github.kotlinmania.cudarc.curand.safe

import io.github.kotlinmania.cudarc.curand.result.CurandGenerator
import io.github.kotlinmania.cudarc.curand.result.createGenerator
import io.github.kotlinmania.cudarc.curand.result.destroyGenerator
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.driver.safe.sliceElements
import kotlin.Result
import kotlin.random.Random
import io.github.kotlinmania.cudarc.curand.result.setOffset as resultSetOffset
import io.github.kotlinmania.cudarc.curand.result.setSeed as resultSetSeed
import io.github.kotlinmania.cudarc.curand.result.setStream as resultSetStream

class CudaRng internal constructor(
    val generator: CurandGenerator,
    var stream: CudaStream,
) {
    private var rng: Random = Random(0)

    companion object {
        fun new(seed: ULong, stream: CudaStream): Result<CudaRng> {
            val gen = createGenerator().getOrElse { return Result.failure(it) }
            resultSetStream(gen, stream.cuStream.handle).getOrThrow()
            resultSetSeed(gen, seed).getOrThrow()
            val result = CudaRng(gen, stream)
            result.rng = Random(seed.toLong())
            return Result.success(result)
        }
    }

    fun setSeed(seed: ULong): Result<Unit> {
        rng = Random(seed.toLong())
        return resultSetSeed(generator, seed)
    }

    fun setOffset(offset: ULong): Result<Unit> = resultSetOffset(generator, offset)

    fun setStream(newStream: CudaStream): Result<Unit> {
        stream = newStream
        return resultSetStream(generator, newStream.cuStream.handle)
    }

    fun fillWithUniform(dst: DevicePtrMut<Float>): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                elements.add(rng.nextFloat().coerceAtLeast(1e-7f))
            }
        }
        return Result.success(Unit)
    }

    fun fillWithUniformDouble(dst: DevicePtrMut<Double>): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                elements.add(rng.nextDouble().coerceAtLeast(1e-10))
            }
        }
        return Result.success(Unit)
    }

    fun fillWithUniformUInt(dst: DevicePtrMut<UInt>): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                elements.add(rng.nextInt(1, Int.MAX_VALUE).toUInt())
            }
        }
        return Result.success(Unit)
    }

    fun fillWithNormal(dst: DevicePtrMut<Float>, mean: Float = 0.0f, std: Float = 1.0f): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                val u1 = rng.nextFloat().coerceAtLeast(1e-7f)
                val u2 = rng.nextFloat()
                val z0 = kotlin.math.sqrt(-2.0f * kotlin.math.ln(u1)) * kotlin.math.cos(2.0f * kotlin.math.PI.toFloat() * u2)
                elements.add(mean + std * z0)
            }
        }
        return Result.success(Unit)
    }

    fun fillWithNormalDouble(dst: DevicePtrMut<Double>, mean: Double = 0.0, std: Double = 1.0): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                val u1 = rng.nextDouble().coerceAtLeast(1e-10)
                val u2 = rng.nextDouble()
                val z0 = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
                elements.add(mean + std * z0)
            }
        }
        return Result.success(Unit)
    }

    fun fillWithLogNormal(dst: DevicePtrMut<Float>, mean: Float = 0.0f, std: Float = 1.0f): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                val u1 = rng.nextFloat().coerceAtLeast(1e-7f)
                val u2 = rng.nextFloat()
                val z0 = kotlin.math.sqrt(-2.0f * kotlin.math.ln(u1)) * kotlin.math.cos(2.0f * kotlin.math.PI.toFloat() * u2)
                elements.add(kotlin.math.exp(mean + std * z0))
            }
        }
        return Result.success(Unit)
    }

    fun fillWithLogNormalDouble(dst: DevicePtrMut<Double>, mean: Double = 0.0, std: Double = 1.0): Result<Unit> {
        val elements = dst.sliceElements()
        if (elements != null) {
            elements.clear()
            repeat(dst.len) {
                val u1 = rng.nextDouble().coerceAtLeast(1e-10)
                val u2 = rng.nextDouble()
                val z0 = kotlin.math.sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
                elements.add(kotlin.math.exp(mean + std * z0))
            }
        }
        return Result.success(Unit)
    }

    fun close() {
        destroyGenerator(generator)
    }
}
