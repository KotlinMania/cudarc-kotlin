// port-lint: source nvtx/result.rs
package io.github.kotlinmania.cudarc.nvtx.result

data class NvtxDomainHandle(
    val handle: Long,
)

data class NvtxRangeId(
    val id: Long,
)

data class NvtxEventAttributes(
    val category: UInt = 0u,
    val colorType: Int = 0,
    val color: UInt = 0u,
    val payloadType: Int = 0,
    val messageType: Int = 0,
    val message: String = "",
)

private var nextDomainId: Long = 1L
private var nextRangeId: Long = 1L

fun domainCreate(name: String): NvtxDomainHandle =
    NvtxDomainHandle(nextDomainId++)

fun domainDestroy(domain: NvtxDomainHandle) {
}

fun domainMark(domain: NvtxDomainHandle, eventAttrib: NvtxEventAttributes) {
}

fun domainRangeStart(domain: NvtxDomainHandle, eventAttrib: NvtxEventAttributes): NvtxRangeId =
    NvtxRangeId(nextRangeId++)

fun domainRangeEnd(domain: NvtxDomainHandle, id: NvtxRangeId) {
}

fun domainRangePush(domain: NvtxDomainHandle, eventAttrib: NvtxEventAttributes): Int = 0

fun domainRangePop(domain: NvtxDomainHandle): Int = 0

fun domainNameCategory(domain: NvtxDomainHandle, category: UInt, name: String) {
}

fun mark(eventAttrib: NvtxEventAttributes) {
}

fun rangeStart(eventAttrib: NvtxEventAttributes): NvtxRangeId =
    NvtxRangeId(nextRangeId++)

fun rangeEnd(id: NvtxRangeId) {
}

fun rangePush(eventAttrib: NvtxEventAttributes): Int = 0

fun rangePop(): Int = 0

fun nameCategory(category: UInt, name: String) {
}
