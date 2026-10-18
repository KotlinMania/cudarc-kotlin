// port-lint: source cupti/result/activity.rs
package io.github.kotlinmania.cudarc.cupti.result

import kotlin.Result

enum class CuptiActivityKind(
    val code: Int,
) {
    CUPTI_ACTIVITY_KIND_INVALID(0),
    CUPTI_ACTIVITY_KIND_MEMCPY(1),
    CUPTI_ACTIVITY_KIND_MEMSET(2),
    CUPTI_ACTIVITY_KIND_KERNEL(3),
    CUPTI_ACTIVITY_KIND_DRIVER(4),
    CUPTI_ACTIVITY_KIND_RUNTIME(5),
    CUPTI_ACTIVITY_KIND_EVENT(6),
    CUPTI_ACTIVITY_KIND_METRIC(7),
    CUPTI_ACTIVITY_KIND_DEVICE(8),
    CUPTI_ACTIVITY_KIND_CONTEXT(9),
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL(10),
}

fun enable(kind: CuptiActivityKind): Result<Unit> = Result.success(Unit)

fun disable(kind: CuptiActivityKind): Result<Unit> = Result.success(Unit)

fun flushAll(flag: UInt): Result<Unit> = Result.success(Unit)
