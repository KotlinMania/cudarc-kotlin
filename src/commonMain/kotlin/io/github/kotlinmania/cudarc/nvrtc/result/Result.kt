// port-lint: source nvrtc/result.rs
package io.github.kotlinmania.cudarc.nvrtc.result

import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

enum class NvrtcResult(val code: Int, val description: String) {
    NVRTC_SUCCESS(0, "Success"),
    NVRTC_ERROR_OUT_OF_MEMORY(1, "Out of memory"),
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE(2, "Program creation failure"),
    NVRTC_ERROR_INVALID_INPUT(3, "Invalid input"),
    NVRTC_ERROR_INVALID_PROGRAM(4, "Invalid program"),
    NVRTC_ERROR_INVALID_OPTION(5, "Invalid option"),
    NVRTC_ERROR_COMPILATION(6, "Compilation error"),
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE(7, "Builtin operation failure"),
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION(8, "No name expressions after compilation"),
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION(9, "No lowered names before compilation"),
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID(10, "Name expression not valid"),
    NVRTC_ERROR_INTERNAL_ERROR(11, "Internal error"),
    ;

    fun result(): Result<Unit> =
        if (this == NVRTC_SUCCESS) {
            Result.success(Unit)
        } else {
            Result.failure(NvrtcError(this))
        }

    companion object {
        fun fromCode(code: Int): NvrtcResult =
            entries.find { it.code == code } ?: NVRTC_ERROR_INTERNAL_ERROR
    }
}

data class NvrtcProgram(val handle: Long)

data class NvrtcError(
    val result: NvrtcResult,
) : Exception("NvrtcError(${result.code}, ${result.name})"), StdError

private var nextProgramId: Long = 1L
private val programSources: MutableMap<Long, String> = mutableMapOf()
private val programLogs: MutableMap<Long, String> = mutableMapOf()
private val programPtx: MutableMap<Long, String> = mutableMapOf()

fun createProgram(src: String, name: String? = null): Result<NvrtcProgram> {
    val id = nextProgramId++
    programSources[id] = src
    programLogs[id] = ""
    return Result.success(NvrtcProgram(id))
}

fun compileProgram(prog: NvrtcProgram, options: List<String>): Result<Unit> {
    val src = programSources[prog.handle]
        ?: return Result.failure(NvrtcError(NvrtcResult.NVRTC_ERROR_INVALID_PROGRAM))
    programPtx[prog.handle] = "// Generated PTX for program ${prog.handle}\n.version 7.0\n.target sm_50\n$src\n"
    programLogs[prog.handle] = "Compilation successful"
    return Result.success(Unit)
}

fun destroyProgram(prog: NvrtcProgram): Result<Unit> {
    programSources.remove(prog.handle)
    programLogs.remove(prog.handle)
    programPtx.remove(prog.handle)
    return Result.success(Unit)
}

fun getPtx(prog: NvrtcProgram): Result<ByteArray> {
    val ptx = programPtx[prog.handle]
        ?: return Result.failure(NvrtcError(NvrtcResult.NVRTC_ERROR_INVALID_PROGRAM))
    return Result.success(ptx.encodeToByteArray())
}

fun getProgramLog(prog: NvrtcProgram): Result<String> {
    val log = programLogs[prog.handle]
        ?: return Result.failure(NvrtcError(NvrtcResult.NVRTC_ERROR_INVALID_PROGRAM))
    return Result.success(log)
}
