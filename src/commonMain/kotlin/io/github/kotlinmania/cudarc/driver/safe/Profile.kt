// port-lint: source driver/safe/profile.rs
package io.github.kotlinmania.cudarc.driver.safe

import kotlin.Result

class Profiler private constructor() {
    companion object {
        fun new(): Result<Profiler> {
            profilerStart().getOrThrow()
            return Result.success(Profiler())
        }
    }

    fun close() {
        profilerStop()
    }
}

fun profilerStart(): Result<Unit> = Result.success(Unit)

fun profilerStop(): Result<Unit> = Result.success(Unit)
