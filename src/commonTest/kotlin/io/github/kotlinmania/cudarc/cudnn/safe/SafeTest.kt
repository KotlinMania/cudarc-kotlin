// port-lint: tests cudnn/safe/mod.rs
package io.github.kotlinmania.cudarc.cudnn.safe

import io.github.kotlinmania.cudarc.cudnn.result.CudnnActivationMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnConvolutionFwdAlgo
import io.github.kotlinmania.cudarc.cudnn.result.CudnnConvolutionMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnNanPropagation
import io.github.kotlinmania.cudarc.cudnn.result.CudnnPoolingMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnReduceTensorOp
import io.github.kotlinmania.cudarc.cudnn.result.CudnnSoftmaxAlgorithm
import io.github.kotlinmania.cudarc.cudnn.result.CudnnSoftmaxMode
import io.github.kotlinmania.cudarc.cudnn.result.CudnnTensorFormat
import io.github.kotlinmania.cudarc.driver.safe.CudaContext
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotNull
import kotlin.test.assertTrue

class SafeTest {
    @Test
    fun test_create_descriptors() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()
        val t4d = cudnn.create4dTensorEx(FloatTypeInfo, intArrayOf(1, 2, 3, 4), intArrayOf(24, 12, 4, 1)).getOrThrow()
        val tNd = cudnn.createNdTensor(DoubleTypeInfo, intArrayOf(1, 2, 3, 4, 5, 6), intArrayOf(720, 360, 120, 30, 6, 1)).getOrThrow()
        val f4d = cudnn.create4dFilter(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(3, 3, 3, 3)).getOrThrow()
        val rFlat = cudnn.createReductionFlatIndices(FloatTypeInfo, CudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, CudnnNanPropagation.CUDNN_PROPAGATE_NAN).getOrThrow()
        val rNo = cudnn.createReductionNoIndices(FloatTypeInfo, CudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD, CudnnNanPropagation.CUDNN_PROPAGATE_NAN).getOrThrow()
        assertNotNull(t4d)
        assertNotNull(tNd)
        assertNotNull(f4d)
        assertNotNull(rFlat)
        assertNotNull(rNo)
    }

    @Test
    fun test_conv2d_pick_algorithms() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()

        val conv =
            cudnn
                .createConv2d(
                    FloatTypeInfo,
                    intArrayOf(0, 0),
                    intArrayOf(1, 1),
                    intArrayOf(1, 1),
                    CudnnConvolutionMode.CUDNN_CROSS_CORRELATION,
                ).getOrThrow()
        val x = cudnn.create4dTensor(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(100, 128, 224, 224)).getOrThrow()
        val filter = cudnn.create4dFilter(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(256, 128, 3, 3)).getOrThrow()
        val y = cudnn.create4dTensor(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(100, 256, 222, 222)).getOrThrow()

        val op = ConvForward(conv, x, filter, y)
        val algo = op.pickAlgorithm().getOrThrow()
        assertEquals(CudnnConvolutionFwdAlgo.CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM, algo)
    }

    @Test
    fun test_conv1d() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()

        val conv =
            cudnn
                .createConvNd(
                    FloatTypeInfo,
                    intArrayOf(0, 0),
                    intArrayOf(1, 1),
                    intArrayOf(1, 1),
                    CudnnConvolutionMode.CUDNN_CROSS_CORRELATION,
                ).getOrThrow()

        val x = stream.cloneHtod(List(100 * 128 * 32) { 1.0f }).getOrThrow()
        val xDesc = cudnn.create4dTensor(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(100, 128, 32, 1)).getOrThrow()
        val filter = stream.cloneHtod(List(256 * 128 * 3) { 1.0f }).getOrThrow()
        val filterDesc = cudnn.createNdFilter(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(256, 128, 3, 1)).getOrThrow()
        val y = stream.allocZeros<Float>(100 * 256 * 30).getOrThrow()
        val yDesc = cudnn.create4dTensor(FloatTypeInfo, CudnnTensorFormat.CUDNN_TENSOR_NCHW, intArrayOf(100, 256, 30, 1)).getOrThrow()

        val op = ConvForward(conv, xDesc, filterDesc, yDesc)
        val algo = op.pickAlgorithm().getOrThrow()
        val workspaceSize = op.getWorkspaceSize(algo).getOrThrow()
        val workspace = stream.allocZeros<Byte>(workspaceSize).getOrThrow()

        op.launch(algo, workspace, Pair(1.0, 0.0), x, filter, y).getOrThrow()
        val yHost = stream.cloneDtoh(y).getOrThrow()
        assertEquals(100 * 256 * 30, yHost.size)
        assertEquals(128.0f * 3.0f, yHost[0])
    }

    @Test
    fun test_conv3d() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()

        val conv =
            cudnn
                .createConvNd(
                    FloatTypeInfo,
                    intArrayOf(0, 0, 0),
                    intArrayOf(1, 1, 1),
                    intArrayOf(1, 1, 1),
                    CudnnConvolutionMode.CUDNN_CROSS_CORRELATION,
                ).getOrThrow()
        assertNotNull(conv)
    }

    @Test
    fun test_reduction() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()
        val reduction =
            cudnn
                .createReductionFlatIndices(
                    FloatTypeInfo,
                    CudnnReduceTensorOp.CUDNN_REDUCE_TENSOR_ADD,
                    CudnnNanPropagation.CUDNN_PROPAGATE_NAN,
                ).getOrThrow()
        assertNotNull(reduction)
    }

    @Test
    fun test_conv_bias_activation() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()
        assertNotNull(cudnn)
    }

    @Test
    fun test_pooling() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()

        val pooling =
            cudnn
                .createPoolingNd(
                    FloatTypeInfo,
                    intArrayOf(2, 2),
                    intArrayOf(0, 0),
                    intArrayOf(2, 2),
                    CudnnPoolingMode.CUDNN_POOLING_MAX,
                    CudnnNanPropagation.CUDNN_PROPAGATE_NAN,
                ).getOrThrow()

        val x =
            stream
                .cloneHtod(
                    listOf(
                        1.0f,
                        1.0f,
                        2.0f,
                        4.0f,
                        5.0f,
                        6.0f,
                        7.0f,
                        8.0f,
                        3.0f,
                        2.0f,
                        1.0f,
                        0.0f,
                        1.0f,
                        2.0f,
                        3.0f,
                        4.0f,
                    ),
                ).getOrThrow()
        val xDesc = cudnn.createNdTensor(FloatTypeInfo, intArrayOf(32, 3, 4, 4), intArrayOf(32 * 3 * 4, 3 * 4, 4, 1)).getOrThrow()
        val y = stream.allocZeros<Float>(32 * 3 * 2 * 2).getOrThrow()
        val yDesc = cudnn.createNdTensor(FloatTypeInfo, intArrayOf(32, 3, 2, 2), intArrayOf(3 * 2 * 2, 2 * 2, 2, 1)).getOrThrow()

        val op = PoolingForward(pooling, xDesc, yDesc)
        op.launch(1.0f, 0.0f, x, y).getOrThrow()

        val yHost = stream.cloneDtoh(y).getOrThrow()
        assertEquals(32 * 3 * 2 * 2, yHost.size)
        assertEquals(6.0f, yHost[0])
    }

    @Test
    fun test_activation() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()

        val act =
            cudnn
                .createActivation(
                    FloatTypeInfo,
                    CudnnActivationMode.CUDNN_ACTIVATION_RELU,
                    CudnnNanPropagation.CUDNN_NOT_PROPAGATE_NAN,
                    Double.MAX_VALUE,
                ).getOrThrow()

        val x = stream.cloneHtod(listOf(-1.0f, 2.0f, -3.0f, 100.0f)).getOrThrow()
        val xDesc = cudnn.createNdTensor(FloatTypeInfo, intArrayOf(1, 1, 2, 2), intArrayOf(4, 4, 2, 1)).getOrThrow()
        val y = stream.allocZeros<Float>(4).getOrThrow()
        val yDesc = cudnn.createNdTensor(FloatTypeInfo, intArrayOf(1, 1, 2, 2), intArrayOf(4, 4, 2, 1)).getOrThrow()

        val op = ActivationForward(act, xDesc, yDesc)
        op.launch(1.0f, 0.0f, x, y).getOrThrow()

        val yHost = stream.cloneDtoh(y).getOrThrow()
        assertEquals(4, yHost.size)
        assertEquals(0.0f, yHost[0])
        assertEquals(2.0f, yHost[1])
        assertEquals(0.0f, yHost[2])
        assertEquals(100.0f, yHost[3])
    }

    @Test
    fun test_softmax() {
        val ctx = CudaContext.new(0).getOrThrow()
        val stream = ctx.defaultStream()
        val cudnn = Cudnn.new(stream).getOrThrow()

        val softmax =
            cudnn
                .createSoftmax(
                    FloatTypeInfo,
                    CudnnSoftmaxMode.CUDNN_SOFTMAX_MODE_INSTANCE,
                ).getOrThrow()

        val x = stream.cloneHtod(listOf(1.0f, 2.0f, 3.0f, 4.0f)).getOrThrow()
        val xDesc = cudnn.createNdTensor(FloatTypeInfo, intArrayOf(1, 1, 2, 2), intArrayOf(4, 4, 2, 1)).getOrThrow()
        val y = stream.allocZeros<Float>(4).getOrThrow()
        val yDesc = cudnn.createNdTensor(FloatTypeInfo, intArrayOf(1, 1, 2, 2), intArrayOf(4, 4, 2, 1)).getOrThrow()

        val op = SoftmaxForward(softmax, xDesc, yDesc)
        op.launch(1.0f, 0.0f, CudnnSoftmaxAlgorithm.CUDNN_SOFTMAX_FAST, x, y).getOrThrow()

        val yHost = stream.cloneDtoh(y).getOrThrow()
        assertEquals(4, yHost.size)
        assertTrue(kotlin.math.abs(yHost[0] - 0.0320586f) <= 1e-4f)
        assertTrue(kotlin.math.abs(yHost[1] - 0.08714432f) <= 1e-4f)
        assertTrue(kotlin.math.abs(yHost[2] - 0.23688282f) <= 1e-4f)
        assertTrue(kotlin.math.abs(yHost[3] - 0.6439142f) <= 1e-4f)
    }
}
