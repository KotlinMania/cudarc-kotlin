// port-lint: tests nvtx/safe.rs
package io.github.kotlinmania.cudarc.nvtx.safe

import kotlin.test.Test
import kotlin.test.assertNotNull

class SafeTest {
    @Test
    fun test_nvtx_mark() {
        mark("hello mark")
    }

    @Test
    fun test_nvtx_range() {
        val range = scopedRange("hello range")
        assertNotNull(range)
    }
}
