// port-lint: source cufft/safe.rs
package io.github.kotlinmania.cudarc.cufft.safe

import io.github.kotlinmania.cudarc.cufft.result.CufftHandle
import io.github.kotlinmania.cudarc.cufft.result.CufftType
import io.github.kotlinmania.cudarc.cufft.result.destroy
import io.github.kotlinmania.cudarc.cufft.result.execC2C
import io.github.kotlinmania.cudarc.cufft.result.execC2R
import io.github.kotlinmania.cudarc.cufft.result.execD2Z
import io.github.kotlinmania.cudarc.cufft.result.execR2C
import io.github.kotlinmania.cudarc.cufft.result.execZ2D
import io.github.kotlinmania.cudarc.cufft.result.execZ2Z
import io.github.kotlinmania.cudarc.cufft.result.plan1d
import io.github.kotlinmania.cudarc.cufft.result.plan2d
import io.github.kotlinmania.cudarc.cufft.result.plan3d
import io.github.kotlinmania.cudarc.cufft.result.setStream
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import kotlin.Result

enum class FftDirection(
    val value: Int,
) {
    Forward(-1),
    Inverse(1),
}

class CudaFft(
    val handle: CufftHandle,
    val stream: CudaStream,
) {
    companion object {
        fun plan1d(
            nx: Int,
            type: CufftType,
            batch: Int,
            stream: CudaStream,
        ): Result<CudaFft> {
            val h = plan1d(nx, type, batch).getOrElse { return Result.failure(it) }
            val setRes = setStream(h, stream.cuStream.handle)
            if (setRes.isFailure) {
                return Result.failure(setRes.exceptionOrNull() ?: Exception("Failed to set stream"))
            }
            return Result.success(CudaFft(h, stream))
        }

        fun plan2d(
            nx: Int,
            ny: Int,
            type: CufftType,
            stream: CudaStream,
        ): Result<CudaFft> {
            val h = plan2d(nx, ny, type).getOrElse { return Result.failure(it) }
            val setRes = setStream(h, stream.cuStream.handle)
            if (setRes.isFailure) {
                return Result.failure(setRes.exceptionOrNull() ?: Exception("Failed to set stream"))
            }
            return Result.success(CudaFft(h, stream))
        }

        fun plan3d(
            nx: Int,
            ny: Int,
            nz: Int,
            type: CufftType,
            stream: CudaStream,
        ): Result<CudaFft> {
            val h = plan3d(nx, ny, nz, type).getOrElse { return Result.failure(it) }
            val setRes = setStream(h, stream.cuStream.handle)
            if (setRes.isFailure) {
                return Result.failure(setRes.exceptionOrNull() ?: Exception("Failed to set stream"))
            }
            return Result.success(CudaFft(h, stream))
        }

        fun planMany(
            n: IntArray,
            inembed: IntArray?,
            istride: Int,
            idist: Int,
            onembed: IntArray?,
            ostride: Int,
            odist: Int,
            type: CufftType,
            batch: Int,
            stream: CudaStream,
        ): Result<CudaFft> {
            val h =
                io.github.kotlinmania.cudarc.cufft.result
                    .planMany(
                        n,
                        inembed,
                        istride,
                        idist,
                        onembed,
                        ostride,
                        odist,
                        type,
                        batch,
                    ).getOrElse { return Result.failure(it) }
            val setRes = setStream(h, stream.cuStream.handle)
            if (setRes.isFailure) {
                return Result.failure(setRes.exceptionOrNull() ?: Exception("Failed to set stream"))
            }
            return Result.success(CudaFft(h, stream))
        }
    }

    fun close() {
        destroy(handle)
    }

    fun execR2C(
        input: DevicePtr<Float>,
        output: DevicePtrMut<Float>,
    ): Result<Unit> {
        val inPtr = input.devicePtr(stream).first
        val outPtr = output.devicePtrMut(stream).first
        return execR2C(handle, inPtr.handle.toLong(), outPtr.handle.toLong())
    }

    fun execC2R(
        input: DevicePtr<Float>,
        output: DevicePtrMut<Float>,
    ): Result<Unit> {
        val inPtr = input.devicePtr(stream).first
        val outPtr = output.devicePtrMut(stream).first
        return execC2R(handle, inPtr.handle.toLong(), outPtr.handle.toLong())
    }

    fun execC2C(
        input: DevicePtr<Float>,
        output: DevicePtrMut<Float>,
        direction: FftDirection,
    ): Result<Unit> {
        val inPtr = input.devicePtr(stream).first
        val outPtr = output.devicePtrMut(stream).first
        return execC2C(handle, inPtr.handle.toLong(), outPtr.handle.toLong(), direction.value)
    }

    fun execD2Z(
        input: DevicePtr<Double>,
        output: DevicePtrMut<Double>,
    ): Result<Unit> {
        val inPtr = input.devicePtr(stream).first
        val outPtr = output.devicePtrMut(stream).first
        return execD2Z(handle, inPtr.handle.toLong(), outPtr.handle.toLong())
    }

    fun execZ2D(
        input: DevicePtr<Double>,
        output: DevicePtrMut<Double>,
    ): Result<Unit> {
        val inPtr = input.devicePtr(stream).first
        val outPtr = output.devicePtrMut(stream).first
        return execZ2D(handle, inPtr.handle.toLong(), outPtr.handle.toLong())
    }

    fun execZ2Z(
        input: DevicePtr<Double>,
        output: DevicePtrMut<Double>,
        direction: FftDirection,
    ): Result<Unit> {
        val inPtr = input.devicePtr(stream).first
        val outPtr = output.devicePtrMut(stream).first
        return execZ2Z(handle, inPtr.handle.toLong(), outPtr.handle.toLong(), direction.value)
    }
}
