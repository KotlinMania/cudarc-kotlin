package io.github.kotlinmania.cudarc

internal const val CUDA_MAJOR_VERSION: String = "12"
internal const val CUDA_MINOR_VERSION: String = "0"

internal fun panicNoLibFound(libName: String, choices: List<String>): Nothing {
    error("Unable to dynamically load the \"$libName\" shared library - searched for library names: $choices. Ensure that LD_LIBRARY_PATH has the correct path to the installed library. If the shared library is present on the system under a different name than one of those listed above, please open a GitHub issue.")
}

fun getLibNameCandidates(libName: String): List<String> {
    val pointerWidth = "64"
    val major = CUDA_MAJOR_VERSION
    val minor = CUDA_MINOR_VERSION

    val prefixes = listOf("lib", "")
    val suffixes = listOf(".so", ".dylib", ".dll")

    val candidates = mutableListOf<String>()
    for (prefix in prefixes) {
        for (suffix in suffixes) {
            candidates.add("$prefix$libName$suffix")
            candidates.add("$prefix$libName$pointerWidth$suffix")
            candidates.add("$prefix$libName${pointerWidth}_$major$suffix")
            candidates.add("$prefix$libName$pointerWidth$major$minor$suffix")
            candidates.add("$prefix$libName$pointerWidth${major}${minor}_0$suffix")
            candidates.add("$prefix$libName${pointerWidth}_${major}0_$minor$suffix")
            candidates.add("$prefix$libName${pointerWidth}_10$suffix")
            candidates.add("$prefix$libName${pointerWidth}_11$suffix")
            candidates.add("$prefix$libName${pointerWidth}_12$suffix")
            candidates.add("$prefix$libName${pointerWidth}_${major}0_0$suffix")
            candidates.add("$prefix$libName${pointerWidth}_9$suffix")
            candidates.add("$prefix$libName$suffix.$major")
            candidates.add("$prefix$libName$suffix.12")
            candidates.add("$prefix$libName$suffix.11")
            candidates.add("$prefix$libName$suffix.10")
            candidates.add("$prefix$libName$suffix.9")
            candidates.add("$prefix$libName$suffix.1")
        }
    }
    return candidates.distinct()
}
