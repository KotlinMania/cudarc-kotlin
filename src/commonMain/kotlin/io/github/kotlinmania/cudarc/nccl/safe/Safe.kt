// port-lint: source nccl/safe.rs
package io.github.kotlinmania.cudarc.nccl.safe

import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import io.github.kotlinmania.cudarc.driver.safe.CudaStream
import io.github.kotlinmania.cudarc.driver.safe.DevicePtr
import io.github.kotlinmania.cudarc.driver.safe.DevicePtrMut
import io.github.kotlinmania.cudarc.nccl.result.NcclComm
import io.github.kotlinmania.cudarc.nccl.result.NcclDataType
import io.github.kotlinmania.cudarc.nccl.result.NcclError
import io.github.kotlinmania.cudarc.nccl.result.NcclRedOp
import io.github.kotlinmania.cudarc.nccl.result.NcclStatus
import io.github.kotlinmania.cudarc.nccl.result.NcclUniqueId
import io.github.kotlinmania.cudarc.driver.sys.CUdeviceptr
import io.github.kotlinmania.cudarc.nccl.result.commAbort
import io.github.kotlinmania.cudarc.nccl.result.commInitAll
import io.github.kotlinmania.cudarc.nccl.result.commInitRank
import io.github.kotlinmania.cudarc.nccl.result.getUniqueId
import io.github.kotlinmania.cudarc.nccl.result.groupEnd
import io.github.kotlinmania.cudarc.nccl.result.groupStart
import io.github.kotlinmania.cudarc.nccl.result.allGather as resultAllGather
import io.github.kotlinmania.cudarc.nccl.result.allReduce as resultAllReduce
import io.github.kotlinmania.cudarc.nccl.result.broadcast as resultBroadcast
import io.github.kotlinmania.cudarc.nccl.result.recv as resultRecv
import io.github.kotlinmania.cudarc.nccl.result.reduce as resultReduce
import io.github.kotlinmania.cudarc.nccl.result.reduceScatter as resultReduceScatter
import io.github.kotlinmania.cudarc.nccl.result.send as resultSend
import kotlin.Result

data class Id(val id: NcclUniqueId) {
    companion object {
        fun new(): Result<Id> =
            getUniqueId().map { Id(it) }

        fun uninit(internal: ByteArray): Id =
            Id(NcclUniqueId(internal))
    }

    fun internal(): ByteArray = id.internal
}

enum class ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

private fun convertToNcclReduceOp(op: ReduceOp): NcclRedOp = when (op) {
    ReduceOp.Sum -> NcclRedOp.NcclSum
    ReduceOp.Prod -> NcclRedOp.NcclProd
    ReduceOp.Max -> NcclRedOp.NcclMax
    ReduceOp.Min -> NcclRedOp.NcclMin
    ReduceOp.Avg -> NcclRedOp.NcclAvg
}

interface NcclType {
    fun asNcclType(): NcclDataType
}

class Comm(
    val comm: NcclComm,
    val stream: CudaStream,
    val rank: Int,
    val worldSize: Int,
) {
    fun stream(): CudaStream = stream

    fun context(): CudaContext = stream.context

    fun ordinal(): Int = stream.context.ordinal

    fun rank(): Int = rank

    fun worldSize(): Int = worldSize

    fun close() {
        commAbort(comm)
    }

    fun <S : DevicePtr<T>, T : NcclType> send(
        data: S,
        peer: Int,
        ncclType: T,
    ): Result<Unit> {
        val src = data.devicePtr(stream).first
        return resultSend(
            sendbuff = src,
            count = data.len.toLong(),
            datatype = ncclType.asNcclType(),
            peer = peer,
            comm = comm,
            stream = stream.cuStream.handle,
        ).map { }
    }

    fun <R : DevicePtrMut<T>, T : NcclType> recv(
        buff: R,
        peer: Int,
        ncclType: T,
    ): Result<NcclStatus> {
        val dst = buff.devicePtrMut(stream).first
        return resultRecv(
            recvbuff = dst,
            count = buff.len.toLong(),
            datatype = ncclType.asNcclType(),
            peer = peer,
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <S : DevicePtr<T>, R : DevicePtrMut<T>, T : NcclType> broadcast(
        sendbuff: S?,
        recvbuff: R,
        root: Int,
        ncclType: T,
    ): Result<NcclStatus> {
        val src = sendbuff?.devicePtr(stream)?.first ?: CUdeviceptr(0uL)
        val dst = recvbuff.devicePtrMut(stream).first
        return resultBroadcast(
            sendbuff = src,
            recvbuff = dst,
            count = recvbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            root = root,
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <R : DevicePtrMut<T>, T : NcclType> broadcastInPlace(
        recvbuff: R,
        root: Int,
        ncclType: T,
    ): Result<NcclStatus> {
        val dst = recvbuff.devicePtrMut(stream).first
        return resultBroadcast(
            sendbuff = dst,
            recvbuff = dst,
            count = recvbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            root = root,
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <S : DevicePtr<T>, R : DevicePtrMut<T>, T : NcclType> allGather(
        sendbuff: S,
        recvbuff: R,
        ncclType: T,
    ): Result<NcclStatus> {
        val src = sendbuff.devicePtr(stream).first
        val dst = recvbuff.devicePtrMut(stream).first
        return resultAllGather(
            sendbuff = src,
            recvbuff = dst,
            sendcount = sendbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <S : DevicePtr<T>, R : DevicePtrMut<T>, T : NcclType> allReduce(
        sendbuff: S,
        recvbuff: R,
        reduceOp: ReduceOp,
        ncclType: T,
    ): Result<NcclStatus> {
        val src = sendbuff.devicePtr(stream).first
        val dst = recvbuff.devicePtrMut(stream).first
        return resultAllReduce(
            sendbuff = src,
            recvbuff = dst,
            count = sendbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            op = convertToNcclReduceOp(reduceOp),
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <R : DevicePtrMut<T>, T : NcclType> allReduceInPlace(
        buff: R,
        reduceOp: ReduceOp,
        ncclType: T,
    ): Result<NcclStatus> {
        val dst = buff.devicePtrMut(stream).first
        return resultAllReduce(
            sendbuff = dst,
            recvbuff = dst,
            count = buff.len.toLong(),
            datatype = ncclType.asNcclType(),
            op = convertToNcclReduceOp(reduceOp),
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <S : DevicePtr<T>, R : DevicePtrMut<T>, T : NcclType> reduce(
        sendbuff: S,
        recvbuff: R?,
        reduceOp: ReduceOp,
        root: Int,
        ncclType: T,
    ): Result<NcclStatus> {
        val src = sendbuff.devicePtr(stream).first
        val dst = recvbuff?.devicePtrMut(stream)?.first ?: CUdeviceptr(0uL)
        return resultReduce(
            sendbuff = src,
            recvbuff = dst,
            count = sendbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            op = convertToNcclReduceOp(reduceOp),
            root = root,
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <R : DevicePtrMut<T>, T : NcclType> reduceInPlace(
        recvbuff: R,
        reduceOp: ReduceOp,
        root: Int,
        ncclType: T,
    ): Result<NcclStatus> {
        val dst = recvbuff.devicePtrMut(stream).first
        return resultReduce(
            sendbuff = dst,
            recvbuff = dst,
            count = recvbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            op = convertToNcclReduceOp(reduceOp),
            root = root,
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    fun <S : DevicePtr<T>, R : DevicePtrMut<T>, T : NcclType> reduceScatter(
        sendbuff: S,
        recvbuff: R,
        reduceOp: ReduceOp,
        ncclType: T,
    ): Result<NcclStatus> {
        val src = sendbuff.devicePtr(stream).first
        val dst = recvbuff.devicePtrMut(stream).first
        return resultReduceScatter(
            sendbuff = src,
            recvbuff = dst,
            recvcount = recvbuff.len.toLong(),
            datatype = ncclType.asNcclType(),
            op = convertToNcclReduceOp(reduceOp),
            comm = comm,
            stream = stream.cuStream.handle,
        )
    }

    companion object {
        fun fromDevices(streams: List<CudaStream>): Result<List<Comm>> {
            val nStreams = streams.size
            val ordinals = streams.map { it.context.ordinal }
            val commsResult = commInitAll(nStreams, ordinals)
            return commsResult.map { comms ->
                comms.mapIndexed { rank, comm ->
                    Comm(
                        comm = comm,
                        stream = streams[rank],
                        rank = rank,
                        worldSize = nStreams,
                    )
                }
            }
        }

        fun fromRank(
            stream: CudaStream,
            rank: Int,
            worldSize: Int,
            id: Id,
        ): Result<Comm> {
            val commResult = commInitRank(worldSize, id.id, rank)
            return commResult.map { comm ->
                Comm(
                    comm = comm,
                    stream = stream,
                    rank = rank,
                    worldSize = worldSize,
                )
            }
        }
    }
}

inline fun <R> group(action: () -> R): Result<R> {
    val startRes = groupStart()
    if (startRes.isFailure) {
        return Result.failure(startRes.exceptionOrNull() ?: Exception("Failed to start group"))
    }
    val result = try {
        action()
    } finally {
        groupEnd()
    }
    return Result.success(result)
}
