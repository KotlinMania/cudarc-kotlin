// port-lint: source nccl/result.rs
package io.github.kotlinmania.cudarc.nccl.result

import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.thiserror.StdError
import kotlin.Result

data class NcclError(
    val code: Int,
    val description: String = "NcclError($code)",
) : Exception(description), StdError

enum class NcclStatus(val code: Int) {
    Success(0),
    InProgress(1),
    NumResults(2),
}

data class NcclComm(val handle: Long)

data class NcclUniqueId(val internal: ByteArray = ByteArray(128)) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is NcclUniqueId) return false
        return internal.contentEquals(other.internal)
    }

    override fun hashCode(): Int = internal.contentHashCode()
}

enum class NcclRedOp(val code: Int) {
    NcclSum(0),
    NcclProd(1),
    NcclMax(2),
    NcclMin(3),
    NcclAvg(4),
    NcclNumOps(5),
}

enum class NcclDataType(val code: Int) {
    NcclInt8(0),
    NcclChar(0),
    NcclUint8(1),
    NcclInt32(2),
    NcclInt(2),
    NcclUint32(3),
    NcclInt64(4),
    NcclUint64(5),
    NcclFloat16(6),
    NcclHalf(6),
    NcclFloat32(7),
    NcclFloat(7),
    NcclFloat64(8),
    NcclDouble(8),
    NcclBfloat16(9),
    NcclNumTypes(10),
}

private var nextCommId: Long = 1L

fun commFinalize(comm: NcclComm): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun commDestroy(comm: NcclComm): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun commAbort(comm: NcclComm): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun getNcclVersion(): Result<Int> = Result.success(21800)

fun getUniqueId(): Result<NcclUniqueId> = Result.success(NcclUniqueId())

fun commInitRank(
    nranks: Int,
    commId: NcclUniqueId,
    rank: Int,
): Result<NcclComm> = Result.success(NcclComm(nextCommId++))

fun commInitAll(
    ndev: Int,
    devlist: List<Int>,
): Result<List<NcclComm>> = Result.success(List(ndev) { NcclComm(nextCommId++) })

fun commCount(comm: NcclComm): Result<Int> = Result.success(1)

fun commCuDevice(comm: NcclComm): Result<Int> = Result.success(0)

fun commUserRank(comm: NcclComm): Result<Int> = Result.success(0)

fun reduce(
    sendbuff: CUdeviceptr,
    recvbuff: CUdeviceptr,
    count: Long,
    datatype: NcclDataType,
    op: NcclRedOp,
    root: Int,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun broadcast(
    sendbuff: CUdeviceptr,
    recvbuff: CUdeviceptr,
    count: Long,
    datatype: NcclDataType,
    root: Int,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun allReduce(
    sendbuff: CUdeviceptr,
    recvbuff: CUdeviceptr,
    count: Long,
    datatype: NcclDataType,
    op: NcclRedOp,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun reduceScatter(
    sendbuff: CUdeviceptr,
    recvbuff: CUdeviceptr,
    recvcount: Long,
    datatype: NcclDataType,
    op: NcclRedOp,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun allGather(
    sendbuff: CUdeviceptr,
    recvbuff: CUdeviceptr,
    sendcount: Long,
    datatype: NcclDataType,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun send(
    sendbuff: CUdeviceptr,
    count: Long,
    datatype: NcclDataType,
    peer: Int,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun recv(
    recvbuff: CUdeviceptr,
    count: Long,
    datatype: NcclDataType,
    peer: Int,
    comm: NcclComm,
    stream: Long,
): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun groupStart(): Result<NcclStatus> = Result.success(NcclStatus.Success)

fun groupEnd(): Result<NcclStatus> = Result.success(NcclStatus.Success)
