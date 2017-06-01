import unittest

import numpy

import chainer
from chainer import cuda
from chainer import gradient_check
from chainer import testing
from chainer.testing import attr
from chainer.testing import condition

import cumprod as functions


@testing.parameterize(*testing.product({
    'shape': [3, (2, 3), (4, 2, 3)],
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
    'contain_zero': [True, False],
}))
class TestCumprod(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, self.shape).astype(self.dtype)
        if self.contain_zero:
            index = numpy.random.choice(self.x.size)
            self.x.ravel()[index] = 0

    def check_forward(self, x_data, axis):
        xp = cuda.get_array_module(x_data)
        x = chainer.Variable(x_data)
        y = functions.cumprod(x, axis=axis)
        self.assertEqual(y.data.dtype, self.dtype)
        # TODO: simply use xp.cumprod
        y_expect = xp.asarray(numpy.cumprod(self.x, axis=axis))

        if self.dtype == numpy.float16:
            options = {'atol': 1e-3, 'rtol': 1e-3}
        else:
            options = {}

        testing.assert_allclose(y_expect, y.data, **options)

    @condition.retry(3)
    def test_forward_cpu(self):
        axes = numpy.arange(-self.x.ndim, self.x.ndim).tolist() + [None]
        for axis in axes:
            self.check_forward(self.x, axis)

    @attr.gpu
    @condition.retry(3)
    def test_forward_gpu(self):
        axes = numpy.arange(-self.x.ndim, self.x.ndim).tolist() + [None]
        for axis in axes:
            self.check_forward(self.x, axis)
            self.check_forward(cuda.to_gpu(self.x), axis)

    def check_backward(self, x_data, axis, y_grad):
        gradient_check.check_backward(
            lambda x: functions.cumprod(x, axis), x_data, y_grad,
            atol=1e-3, dtype=numpy.float64)

    @condition.retry(3)
    def test_backward_cpu(self):
        axes = numpy.arange(-self.x.ndim, self.x.ndim).tolist() + [None]
        for axis in axes:
            g_shape = numpy.cumprod(self.x, axis=axis).shape
            gy = numpy.random.uniform(-1, 1, g_shape).astype(self.dtype)
            self.check_backward(self.x, axis, gy)

    @attr.gpu
    @condition.retry(3)
    def test_backward_axis_gpu(self):
        axes = numpy.arange(-self.x.ndim, self.x.ndim).tolist() + [None]
        for axis in axes:
            # TODO: use cupy.cumprod
            g_shape = numpy.cumprod(cuda.to_cpu(self.x), axis=axis).shape
            gy = cuda.cupy.random.uniform(-1, 1, g_shape).astype(self.dtype)
            self.check_backward(
                cuda.to_gpu(self.x), axis, cuda.to_gpu(gy))


@testing.parameterize(*testing.product({
    'dtype': [numpy.float16, numpy.float32, numpy.float64],
}))
class TestCumprodError(unittest.TestCase):

    def setUp(self):
        self.x = numpy.random.uniform(-1, 1, (3, 2, 4)).astype(self.dtype)

    def test_invalid_axis_type(self):
        with self.assertRaises(TypeError):
            functions.cumprod(self.x, [0])
        with self.assertRaises(TypeError):
            functions.cumprod(self.x, (0,))


testing.run_module(__name__, __file__)
