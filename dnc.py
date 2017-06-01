# -*- coding: utf-8 -*-
"""
Created on Thu May 11 18:51:53 2017

@author: sakurai
"""

import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import Variable
import chainer.functions as F
import chainer.links as L

from prod import prod
from cumprod import cumprod


class Controller(chainer.Chain):
    '''
    Args:
        dim_x:
            number of dimensions of the input x
        dim_h:
            number of dimensions of the LSTM output h
        dim_y:
            number of dimensions of the output y
        N:
            number of slots (rows) in the memory matrix M
        W:
            number of dimension of a memory (i.e. a row of M)
        R:
            number of the read heads
        '''

    def __init__(self, dim_x, dim_h, dim_y, N, W, R):
        self.N = N
        self.W = W
        self.R = R
        dim_chi = dim_x + R * W
        xi_lengths = [W * R, R, W, 1, W, W, R, 1, 1, 3 * R]
        xi_length = sum(xi_lengths)
        self._xi_split_indices = np.cumsum(xi_lengths)[:-1]
        super(Controller, self).__init__(
            lstm=L.LSTM(dim_chi, dim_h),
            l_ups=L.Linear(dim_h, dim_y, nobias=True),  # W_upsilon
            l_xi=L.Linear(dim_h, xi_length, nobias=True),  # W_xi
            l_r=L.Linear(R * W, dim_y, nobias=True)  # W_r
        )

    def to_cpu(self):
        super(Controller, self).to_cpu()
        if hasattr(self, 'r') and self.r is not None:
            self.r.to_cpu()
        if hasattr(self, 'L') and self.L is not None:
            self.L.to_cpu()
        if hasattr(self, 'M') and self.M is not None:
            self.M.to_cpu()
        if hasattr(self, 'w_w') and self.w_w is not None:
            self.w_w.to_cpu()
        if hasattr(self, 'w_r') and self.w_r is not None:
            self.w_r.to_cpu()
        if hasattr(self, 'u') and self.u is not None:
            self.u.to_cpu()
        if hasattr(self, 'p') and self.p is not None:
            self.p.to_cpu()
        return self

    def to_gpu(self, device=None):
        super(Controller, self).to_gpu(device)
        if hasattr(self, 'r') and self.r is not None:
            self.r.to_gpu(device)
        if hasattr(self, 'L') and self.L is not None:
            self.L.to_gpu(device)
        if hasattr(self, 'M') and self.M is not None:
            self.M.to_gpu(device)
        if hasattr(self, 'w_w') and self.w_w is not None:
            self.w_w.to_gpu(device)
        if hasattr(self, 'w_r') and self.w_r is not None:
            self.w_r.to_gpu(device)
        if hasattr(self, 'u') and self.u is not None:
            self.u.to_gpu(device)
        if hasattr(self, 'p') and self.p is not None:
            self.p.to_gpu(device)
        return self

    def reset_state(self, batch_size):
        self.lstm.reset_state()
        xp = self.xp
        B, N, W, R = batch_size, self.N, self.W, self.R
        self.r = xp.zeros((B, R * W), dtype=np.float32)
        self.L = xp.zeros((B, N, N), dtype=np.float32)
        self.M = xp.zeros((B, N, W), dtype=np.float32)
        self.w_w = xp.zeros((B, N), dtype=np.float32)
        self.w_r = xp.zeros((B, R, N), dtype=np.float32)
        self.u = xp.zeros((B, N), dtype=np.float32)
        self.p = xp.zeros((B, N), dtype=np.float32)

    def set_state(self, batch_size):
        raise NotImplementedError

    def __call__(self, x):
        r_prev = self.r
        chi = F.concat((x, r_prev))
        h = self.lstm(chi)
        ups = self.l_ups(h)
        xi = self.l_xi(h)
        k_r, beta_r, k_w, beta_w, e, v, free, g_a, g_w, pi = self._parse_xi(xi)

        L_prev = self.L
        M_prev = self.M
        w_w_prev = self.w_w
        w_r_prev = self.w_r
        u_prev = self.u
        p_prev = self.p

        # write memory
        w_w, u = self._write_weighting(k_w, beta_w, free, g_a, g_w,
                                       M_prev, w_w_prev, w_r_prev, u_prev)
        M = self._write_memory(w_w, e, v, M_prev)

        # update temporal link matrix
        L = self._update_temporal_link_matrix(w_w, L_prev, p_prev)
        p = self._precedence_weighting(p_prev, w_w)

        # read memory
        w_r = self._read_weighting(M, L, k_r, beta_r, pi, w_r_prev)
        r = self._read_memory(M, w_r)

        y = ups + self.l_r(r)

        self.r = r
        self.L = L
        self.M = M
        self.w_w = w_w
        self.w_r = w_r
        self.u = u
        self.p = p

        return y

    def _parse_xi(self, xi):
        W = self.W
        R = self.R
        k_r, beta_r, k_w, beta_w, e, v, free, g_a, g_w, pi = F.split_axis(
            xi, self._xi_split_indices, 1)
        k_r = k_r.reshape((-1, R, W))
        beta_r = 1 + F.softplus(beta_r.reshape((-1, R)))
        beta_w = 1 + F.softplus(beta_w)
        e = F.sigmoid(e)
        free = F.sigmoid(free.reshape((-1, R)))
        g_a = F.sigmoid(g_a)
        g_w = F.sigmoid(g_w)
        pi = F.softmax(pi.reshape((-1, R, 3)), axis=2)
        return k_r, beta_r, k_w, beta_w, e, v, free, g_a, g_w, pi

    def _write_memory(self, w_w, e, v, M_prev):
        ones = xp.ones((self.N, self.W), xp.float32)
        we = F.batch_matmul(w_w, e, transb=True)
        wv = F.batch_matmul(w_w, v, transb=True)
        M_new = M_prev * (ones - we) + wv
        return M_new

    def _read_memory(self, M, w_r):
        M_trans = F.transpose(M, (0, 1, 2))  # TODO: check whether M or M.T?
        r = F.batch_matmul(w_r, M_trans)
        concatenated_r = r.reshape(-1, self.R * self.W)  # (B*R, W) -> (B, R*W)
        return concatenated_r

    def _content_weighting(self, M, k, beta):
        '''
        Args:
            M:
                memory matrix of shape (B, N, W)
            k:
                batch of key vectors, the shape must be (B, W) or (B, R, W)
            beta:
                strength parameter
        '''
        ndim = k.ndim
        assert ndim == 2 or ndim == 3

        if ndim == 2:
            B, W = k.shape
            k = k.reshape((B, 1, W))
        B, R, W = k.shape

        M = F.normalize(M, axis=2)
        k = F.normalize(k, axis=2)
        cosine = F.batch_matmul(k, M, transb=True)
        beta = F.expand_dims(beta, 2)
        beta = F.broadcast_to(beta, (B, R, self.N))
        w = F.softmax(cosine * beta, axis=2)

        if ndim == 2:
            w = w.reshape((B, self.N))
        return w

    def _retention_vector(self, free, w_r_prev):
        '''
        Args:
            free:
                (B, R)
            w_r_prev:
                (B, R, N)
        Returns:
            psi:
                (B, N)
        '''
        terms = 1 - F.scale(w_r_prev, free, 0)
        psi = prod(terms, 1)
        return psi

    def _usage_vector(self, psi, u_prev, w_w_prev):
        '''
        Args:
            psi:
                (B, N)
            u_prev:
                (B, M)
            w_w_prev:
                (B, N)
        Returns:
            u_new:
                (B, N)
        '''
        u_new = (u_prev + w_w_prev - u_prev * w_w_prev) * psi
        return u_new

    def _allocation_weighting(self, u):
        '''
        Args:
            u:
                (B, N)
        Returns:
            a:
                (B, N)
        '''
        xp = self.xp
        batch_size = u.shape[0]

        # sort u
        phi = xp.argsort(u.data, axis=1)
        b_indices = xp.tile(xp.arange(batch_size).reshape(-1, 1), (1, self.N))
        u = u[b_indices, phi]

        # calculate `a` using the cumprod of [1; u] = [1, u[0], ..., u[N-1]]
        # -> [1, u[0], u[0]*u[1], ..., u[0]*...*u[N-1]]
        u = F.hstack((xp.ones((batch_size, 1), xp.float32), u))
        cp_u = cumprod(u)
        a = cp_u[:, :-1] - cp_u[:, 1:]  # eq. (1) can be written as like this
        return a

    def _write_weighting(self, k_w, beta_w, free, g_a, g_w,
                         M_prev, w_w_prev, w_r_prev, u_prev):
        '''
        Args:
            k_w:
                (B, W)
            beta_w:
                (B, 1)
            free:
                (B, R)
            g_a:
                (B, 1)
            g_w:
                (B, 1)
            M_prev:
                (B, N, W)
            w_w_prev:
                (B, N)
            w_r_prev:
                (B, R, N)
            u_prev:
                (B, N)
        Returns:
            w_w:
                (B, N)
            u:
                (B, N)
        '''
        psi = self._retention_vector(free, w_r_prev)
        u = self._usage_vector(psi, u_prev, w_w_prev)
        a = self._allocation_weighting(u)
        c_w = self._content_weighting(M_prev, k_w, beta_w)

        g_w, g_a, a, c_w = F.broadcast(g_w, g_a, a, c_w)
        w_w = g_w * (g_a * a + (1 - g_a) * c_w)
        return w_w, u

    def _precedence_weighting(self, p_prev, w_w):
        '''
        Args:
            p_prev:
                (B, N)
            w_w:
                (B, N)
        Returns:
            p_new:
                (B, N)
        '''
        sum_w = F.sum(w_w, axis=1, keepdims=True)
        sum_w, p_prev, w_w = F.broadcast(sum_w, p_prev, w_w)
        return (1 - sum_w) * p_prev + w_w

    def _update_temporal_link_matrix(self, w_w, L_prev, p_prev):
        '''
        Args:
            w_w:
                (B, N)
            L_prev:
                (B, N, N)
            p_prev:
                (B, N)
        Returns:
            L_new:
                (B, N, N)
        '''
        w_w_row = F.expand_dims(w_w, 1)
        w_w_col = F.expand_dims(w_w, 2)
        w_w_row_b, w_w_col_b = F.broadcast(w_w_row, w_w_col)
        p_row = F.expand_dims(p_prev, 1)
        tmp = F.batch_matmul(w_w_col, p_row)
        L = (1 - w_w_col_b - w_w_row_b) * L_prev + tmp

        B, N = w_w.shape
        mask = xp.ones((N, N), dtype=np.float32)
        xp.fill_diagonal(mask, 0)
        mask = xp.tile(mask, (B, 1, 1))
        return L * mask

    def _forward_backward_weighting(self, L, w_r_prev):
        '''
        Args:
            L:
                (B, N, N)
            w_r_prev:
                (B, R, N)
        Returns:
            f:
                (B, R, N)
            b:
                (B, R, N)
        '''
        f = F.batch_matmul(w_r_prev, L, transb=True)
        b = F.batch_matmul(w_r_prev, L, transb=False)
        return f, b

    def _read_weighting(self, M, L, k_r, beta_r, pi, w_r_prev):
        '''
        Args:
            M:
                (B, N, W)
            L:
                (B, N, N)
            k_r:
                (B, R, W)
            beta_r:
                (B, R)
            pi:
                (B, R, 3)
            w_r__prev:
                (B, R, N)
        Returns:
            w_r:
                (B, R, N)
        '''
        B = k_r.shape[0]  # batch size
        R = self.R
        N = self.N
        f, b = self._forward_backward_weighting(L, w_r_prev)

        c = self._content_weighting(M, k_r, beta_r)  # (B, R, N)
        bcf = F.stack((b, c, f), 2)  # (B, R, 3, N)
        bcf_matrices = bcf.reshape((B * R), 3, N)
        pi_vectors = pi.reshape((B * R, 1, 3))
        w_vectors = F.batch_matmul(pi_vectors, bcf_matrices)
        w_r = w_vectors.reshape((B, R, N))
        return w_r


if __name__ == '__main__':
    xp = np
    controller = Controller(100, 200, 50, 40, 64, 4)
    if chainer.cuda.available and xp is chainer.cuda.cupy:
        controller.to_gpu()

    batch_size = 16
    x = xp.ones((batch_size, 100), xp.float32)
    controller.reset_state(batch_size)
    y = controller(x)
