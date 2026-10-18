// port-lint: source types.rs
package io.github.kotlinmania.cudarc

/**
 * Maps a rust type to its corresponding [name] in cuda C++ land.
 */
interface CudaTypeName {
    val name: String
}
