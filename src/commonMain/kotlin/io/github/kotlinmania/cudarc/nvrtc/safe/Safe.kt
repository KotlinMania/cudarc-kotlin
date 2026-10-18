// port-lint: source nvrtc/safe.rs
package io.github.kotlinmania.cudarc.nvrtc.safe

import io.github.kotlinmania.cudarc.nvrtc.result.NvrtcError
import io.github.kotlinmania.cudarc.nvrtc.result.compileProgram
import io.github.kotlinmania.cudarc.nvrtc.result.createProgram
import io.github.kotlinmania.cudarc.nvrtc.result.destroyProgram
import io.github.kotlinmania.cudarc.nvrtc.result.getProgramLog
import io.github.kotlinmania.cudarc.nvrtc.result.getPtx
import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

sealed class PtxKind {
    data class Image(
        val bytes: ByteArray,
    ) : PtxKind()

    data class Src(
        val src: String,
    ) : PtxKind()

    data class File(
        val path: String,
    ) : PtxKind()

    data class Binary(
        val data: ByteArray,
    ) : PtxKind()
}

data class Ptx(
    internal val kind: PtxKind,
) {
    companion object {
        fun fromFile(path: String): Ptx = Ptx(PtxKind.File(path))

        fun fromSrc(src: String): Ptx = Ptx(PtxKind.Src(src))

        fun fromBinary(data: ByteArray): Ptx = Ptx(PtxKind.Binary(data))
    }

    fun toSrc(): String =
        when (kind) {
            is PtxKind.Image -> kind.bytes.decodeToString()
            is PtxKind.Src -> kind.src
            is PtxKind.File -> "// File: ${kind.path}"
            is PtxKind.Binary -> error("Cannot convert binary CUBIN data to string. Use fromSrc() or fromFile() for PTX data.")
        }

    fun asBytes(): ByteArray? =
        when (kind) {
            is PtxKind.Image -> kind.bytes
            is PtxKind.Binary -> kind.data
            else -> null
        }
}

sealed class CompileError(
    message: String,
) : Exception(message),
    StdError {
    data class CreationError(
        val error: NvrtcError,
    ) : CompileError("Program creation error: $error")

    data class CompilerError(
        val error: NvrtcError,
        val options: List<String>,
        val log: String,
    ) : CompileError("Compilation error: $error\nLog: $log")

    data class GetLogError(
        val error: NvrtcError,
    ) : CompileError("Get log error: $error")

    data class GetPtxError(
        val error: NvrtcError,
    ) : CompileError("Get PTX error: $error")

    data class DestroyError(
        val error: NvrtcError,
    ) : CompileError("Destroy error: $error")
}

data class CompileOptions(
    val ftz: Boolean? = null,
    val precSqrt: Boolean? = null,
    val precDiv: Boolean? = null,
    val fmad: Boolean? = null,
    val options: List<String> = emptyList(),
    val useFastMath: Boolean? = null,
    val maxrregcount: Int? = null,
    val includePaths: List<String> = emptyList(),
    val arch: String? = null,
    val name: String? = null,
) {
    fun build(): List<String> {
        val result = mutableListOf<String>()
        if (ftz != null) result.add("--ftz=$ftz")
        if (precSqrt != null) result.add("--prec-sqrt=$precSqrt")
        if (precDiv != null) result.add("--prec-div=$precDiv")
        if (fmad != null) result.add("--fmad=$fmad")
        if (useFastMath == true) result.add("--use_fast_math")
        if (maxrregcount != null) result.add("--maxrregcount=$maxrregcount")
        if (arch != null) result.add("--gpu-architecture=$arch")
        for (path in includePaths) {
            result.add("-I$path")
        }
        result.addAll(options)
        return result
    }
}

fun compilePtx(src: String): Result<Ptx> = compilePtxWithOpts(src, CompileOptions())

fun compilePtxWithOpts(src: String, opts: CompileOptions): Result<Ptx> {
    val prog =
        createProgram(src, opts.name).getOrElse {
            return Result.failure(CompileError.CreationError(it as NvrtcError))
        }
    val optionsList = opts.build()
    val compileResult = compileProgram(prog, optionsList)
    if (compileResult.isFailure) {
        val log = getProgramLog(prog).getOrDefault("Unknown error")
        val err =
            compileResult.exceptionOrNull() as? NvrtcError
                ?: NvrtcError(io.github.kotlinmania.cudarc.nvrtc.result.NvrtcResult.NVRTC_ERROR_COMPILATION)
        destroyProgram(prog)
        return Result.failure(CompileError.CompilerError(err, optionsList, log))
    }
    val ptxBytes =
        getPtx(prog).getOrElse {
            destroyProgram(prog)
            return Result.failure(CompileError.GetPtxError(it as NvrtcError))
        }
    destroyProgram(prog)
    return Result.success(Ptx(PtxKind.Image(ptxBytes)))
}
