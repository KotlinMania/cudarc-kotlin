// port-lint: tests cusparse/result.rs
package io.github.kotlinmania.cudarc.cusparse.result

import kotlin.test.Test
import kotlin.test.assertNotNull

class ResultTest {
    @Test
    fun test_create_and_destroy() {
        val handle = create().getOrThrow()
        assertNotNull(handle)
        destroy(handle).getOrThrow()
    }
}
