// port-lint: tests nvrtc/result.rs
package io.github.kotlinmania.cudarc.nvrtc.result

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class ResultTest {
    @Test
    fun test_compile_program_no_opts() {
        val prog = createProgram("""extern "C" __global__ void kernel() { }""").getOrThrow()
        compileProgram(prog, emptyList()).getOrThrow()
        destroyProgram(prog).getOrThrow()
    }

    @Test
    fun test_compile_program_1_opt() {
        val prog = createProgram("""extern "C" __global__ void kernel() { }""").getOrThrow()
        compileProgram(prog, listOf("--ftz=true")).getOrThrow()
        destroyProgram(prog).getOrThrow()
    }

    @Test
    fun test_compile_program_2_opt() {
        val prog = createProgram("""extern "C" __global__ void kernel() { }""").getOrThrow()
        compileProgram(prog, listOf("--ftz=true", "--fmad=true")).getOrThrow()
        destroyProgram(prog).getOrThrow()
    }

    @Test
    fun test_compile_bad_program() {
        val prog = createProgram("""extern "C" __global__ void kernel(""").getOrThrow()
        val err = compileProgram(prog, emptyList()).exceptionOrNull()
        assertEquals(NvrtcError(NvrtcResult.NVRTC_ERROR_COMPILATION), err)
    }

    @Test
    fun test_get_ptx() {
        val src =
            """
            extern "C" __global__ void sin_kernel(float *out, const float *inp, int numel) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < numel) {
                    out[i] = sin(inp[i]);
                }
            }
            """.trimIndent()
        val prog = createProgram(src).getOrThrow()
        compileProgram(prog, emptyList()).getOrThrow()
        val ptx = getPtx(prog).getOrThrow()
        assertTrue(ptx.isNotEmpty())
    }
}
