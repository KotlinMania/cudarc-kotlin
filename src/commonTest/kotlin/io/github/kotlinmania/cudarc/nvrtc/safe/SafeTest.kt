// port-lint: tests nvrtc/safe.rs
package io.github.kotlinmania.cudarc.nvrtc.safe

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class SafeTest {
    @Test
    fun test_compile_no_opts() {
        val src =
            """
            extern "C" __global__ void sin_kernel(float *out, const float *inp, int numel) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < numel) {
                    out[i] = sin(inp[i]);
                }
            }
            """.trimIndent()
        compilePtxWithOpts(src, CompileOptions()).getOrThrow()
    }

    @Test
    fun test_compile_options_build_none() {
        val opts = CompileOptions()
        assertTrue(opts.build().isEmpty())
    }

    @Test
    fun test_compile_options_build_ftz() {
        val opts = CompileOptions(ftz = true)
        assertEquals(listOf("--ftz=true"), opts.build())
    }

    @Test
    fun test_compile_options_build_multi() {
        val opts =
            CompileOptions(
                precDiv = false,
                maxrregcount = 60,
            )
        assertEquals(listOf("--prec-div=false", "--maxrregcount=60"), opts.build())
    }
}
