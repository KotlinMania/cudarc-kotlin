// port-lint: source nvtx/safe.rs
package io.github.kotlinmania.cudarc.nvtx.safe

import io.github.kotlinmania.cudarc.nvtx.result.NvtxDomainHandle
import io.github.kotlinmania.cudarc.nvtx.result.NvtxEventAttributes
import io.github.kotlinmania.cudarc.nvtx.result.NvtxRangeId
import io.github.kotlinmania.cudarc.nvtx.result.domainCreate
import io.github.kotlinmania.cudarc.nvtx.result.domainDestroy
import io.github.kotlinmania.cudarc.nvtx.result.domainMark
import io.github.kotlinmania.cudarc.nvtx.result.domainNameCategory
import io.github.kotlinmania.cudarc.nvtx.result.domainRangeEnd
import io.github.kotlinmania.cudarc.nvtx.result.domainRangePop
import io.github.kotlinmania.cudarc.nvtx.result.domainRangePush
import io.github.kotlinmania.cudarc.nvtx.result.domainRangeStart
import io.github.kotlinmania.cudarc.nvtx.result.mark as resultMark
import io.github.kotlinmania.cudarc.nvtx.result.nameCategory as resultNameCategory
import io.github.kotlinmania.cudarc.nvtx.result.rangeEnd as resultRangeEnd
import io.github.kotlinmania.cudarc.nvtx.result.rangePop as resultRangePop
import io.github.kotlinmania.cudarc.nvtx.result.rangePush as resultRangePush
import io.github.kotlinmania.cudarc.nvtx.result.rangeStart as resultRangeStart

fun scopedRange(msg: String): Range =
    Event.message(msg).range()

fun mark(msg: String) {
    Event.message(msg).mark()
}

class Event(
    var category: UInt? = null,
    var argb: UInt? = null,
    var message: String,
) {
    companion object {
        fun message(message: String): Event =
            Event(message = message)
    }

    fun category(category: UInt): Event {
        this.category = category
        return this
    }

    fun argb(argb: UInt): Event {
        this.argb = argb
        return this
    }

    fun toAttributes(): NvtxEventAttributes =
        NvtxEventAttributes(
            category = category ?: 0u,
            color = argb ?: 0u,
            message = message,
        )

    fun mark() {
        resultMark(toAttributes())
    }

    fun range(): Range {
        val id = resultRangeStart(toAttributes())
        return Range(id)
    }

    fun rangePush(): RangePush {
        resultRangePush(toAttributes())
        return RangePush()
    }
}

class Range(val id: NvtxRangeId) {
    fun close() {
        resultRangeEnd(id)
    }
}

class RangePush {
    fun close() {
        resultRangePop()
    }
}

fun nameCategory(category: UInt, name: String) {
    resultNameCategory(category, name)
}

class Domain(
    val domain: NvtxDomainHandle,
) {
    companion object {
        fun new(name: String): Domain =
            Domain(domainCreate(name))
    }

    fun mark(event: Event) {
        domainMark(domain, event.toAttributes())
    }

    fun range(event: Event): DomainRange {
        val id = domainRangeStart(domain, event.toAttributes())
        return DomainRange(domain, id)
    }

    fun rangePush(event: Event): DomainRangePush {
        domainRangePush(domain, event.toAttributes())
        return DomainRangePush(domain)
    }

    fun nameCategory(category: UInt, name: String) {
        domainNameCategory(domain, category, name)
    }

    fun close() {
        domainDestroy(domain)
    }
}

class DomainRange(
    val domain: NvtxDomainHandle,
    val id: NvtxRangeId,
) {
    fun close() {
        domainRangeEnd(domain, id)
    }
}

class DomainRangePush(
    val domain: NvtxDomainHandle,
) {
    fun close() {
        domainRangePop(domain)
    }
}
