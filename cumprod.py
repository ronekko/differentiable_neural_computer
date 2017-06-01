import numpy
import cupy
import chainer
from chainer import cuda
from chainer import functions
from chainer import Variable
from prod import prod


def cumprod(x, axis=-1):
    if not isinstance(x, Variable):
        x = Variable(x)

    if axis is None:
        x = x.reshape(-1)
        axis = 0
    elif axis < 0:
        axis = x.ndim + axis
    assert axis >= 0 and axis < x.ndim

    xp = cuda.get_array_module(x)
    ndim = x.ndim
    dims = x.shape[axis]
    shape_new = x.shape[:axis] + (dims,) + x.shape[axis:]
    x = functions.expand_dims(x, axis)
    x = functions.broadcast_to(x, shape_new)

    # TODO: use cupy.tril
    mask = numpy.tril(numpy.ones((dims, dims), numpy.bool))
    if xp is cupy:
        mask = cuda.to_gpu(mask)
    expander = [1] * axis + [dims, dims] + [1] * (ndim - axis - 1)
    mask = mask.reshape(expander)
    mask = xp.broadcast_to(mask, shape_new)
    x = functions.where(mask, x, xp.ones_like(x.data))
    return prod(x, axis + 1)


if __name__ == '__main__':
    x = chainer.Variable(
        numpy.arange(1, 1 + 2 * 3 * 4, dtype='f').reshape(2, 3, 4))
    axis = -3
    cp_np = numpy.cumprod(x.data, axis)
    cp_f = cumprod(x, axis).data
    print(x.data)
    print(cp_np)
    print(cp_f)
    assert numpy.array_equal(cp_np, cp_f)
    print(numpy.array_equal(cp_np, cp_f))
