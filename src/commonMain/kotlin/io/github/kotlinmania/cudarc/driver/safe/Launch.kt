// port-lint: source driver/safe/launch.rs
package io.github.kotlinmania.cudarc.driver.safe

import io.github.kotlinmania.cudarc.driver.result.function
import io.github.kotlinmania.cudarc.driver.sys.CUeventFlags
import kotlin.Result

data class LaunchConfig(
    val gridDim: Triple<UInt, UInt, UInt>,
    val blockDim: Triple<UInt, UInt, UInt>,
    val sharedMemBytes: UInt = 0u,
) {
    companion object {
        fun forNumElems(n: UInt): LaunchConfig {
            val numThreads = 1024u
            val numBlocks = (n + numThreads - 1u) / numThreads
            return LaunchConfig(
                gridDim = Triple(numBlocks, 1u, 1u),
                blockDim = Triple(numThreads, 1u, 1u),
                sharedMemBytes = 0u,
            )
        }
    }
}

interface PushKernelArg<T> {
    fun arg(builder: LaunchArgs, arg: T): LaunchArgs
}

class LaunchArgs(
    val stream: CudaStream,
    val func: CudaFunction,
) {
    private val argsList = mutableListOf<Any>()
    private val waitEvents = mutableListOf<CudaEvent>()
    private val recordEvents = mutableListOf<CudaEvent>()
    private var flags: CUeventFlags? = null

    fun arg(argument: Any?): LaunchArgs {
        if (argument != null) {
            argsList.add(argument)
        }
        return this
    }

    fun wait(event: CudaEvent): LaunchArgs {
        waitEvents.add(event)
        return this
    }

    fun record(event: CudaEvent): LaunchArgs {
        recordEvents.add(event)
        return this
    }

    fun launch(cfg: LaunchConfig): Result<Unit> {
        stream.context.bindToThread().getOrThrow()
        for (event in waitEvents) {
            stream.wait(event).getOrThrow()
        }
        function.launchKernel(
            func.cuFunction,
            cfg.gridDim.first,
            cfg.gridDim.second,
            cfg.gridDim.third,
            cfg.blockDim.first,
            cfg.blockDim.second,
            cfg.blockDim.third,
            cfg.sharedMemBytes,
            stream.cuStream,
            argsList,
        ).getOrThrow()
        for (event in recordEvents) {
            event.record(stream).getOrThrow()
        }
        return Result.success(Unit)
    }
}
