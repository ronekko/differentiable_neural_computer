# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 07:18:57 2017

@author: ryuhei
"""

import multiprocessing as mp
from queue import Empty, Full

import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F

from copy_dataset import generate_copy_data
from dnc import Controller


def worker(batch_size, seq_len, dim_x, receive_queue, send_queue):
    pid = mp.current_process()._identity[0]
    np.random.seed(pid)
    model = receive_queue.get()
    while True:
        try:
            model = receive_queue.get(block=False)
        except Empty:
            pass
        x, t = generate_copy_data(batch_size, seq_len, dim_x)
        x = x.transpose((1, 0, 2))
        t = t.transpose((1, 0, 2))

        model.reset_state(batch_size)
        for x_t in x:
            model(x_t)

        y = []
        for t_t in t:
            dummy_input = np.zeros_like(x_t)
            y_t = model(dummy_input)
            y.append(y_t)
        y = F.stack(y)
        loss = F.sigmoid_cross_entropy(y, t)
        model.cleargrads()
        loss.backward()
        try:
            send_queue.put(model, block=False)
        except Full:
            pass


if __name__ == '__main__':
    batch_size = 1
    seq_len = 20
    dim_x = 9
    dim_y = dim_x

    dim_h = 100
    num_memory_slots = 128
    dim_memory_vector = 20
    num_read_heads = 1

    num_processes = 5
    receive_queue_size = 5

    num_updates = 1000000
    learning_rate = 0.0001

    model_master = Controller(dim_x, dim_y, dim_h, num_memory_slots,
                              dim_memory_vector, num_read_heads)

    optimizer = chainer.optimizers.RMSprop(learning_rate)
    optimizer.setup(model_master)
    optimizer.zero_grads()

    # Call the forward once in order to resolve uninitialized variables
    x, c = generate_copy_data(batch_size, seq_len, dim_x)
    x = x.transpose((1, 0, 2))
    model_master.reset_state(batch_size)
    model_master(x[0])

    # Create worker processes
    processes = []
    send_queues = []  # Interface to send the master model to workers
    receive_queue = mp.Queue(maxsize=receive_queue_size)  # To receive models
    for p in range(num_processes):
        send_queue = mp.Queue(maxsize=1)
        process = mp.Process(target=worker,
                             args=(batch_size, seq_len, dim_x,
                                   send_queue, receive_queue))
        process.start()
        processes.append(process)
        send_queue.put(model_master)
        send_queues.append(send_queue)

    count_updates = 0
    evaluate_at = 0
    try:
        while count_updates < num_updates:
            try:
                model = receive_queue.get(block=False)
                model_master.cleargrads()
                model_master.addgrads(model)
                optimizer.update()
                count_updates += 1

                for p in np.random.permutation(num_processes):
                    send_queues[p].put(model_master, block=False)

            except Empty:
                pass
            except Full:
                pass

            # Evaluation
            if count_updates % 20 == 0 and evaluate_at != count_updates:
                evaluate_at = count_updates
                x, t = generate_copy_data(batch_size, seq_len, dim_x)
                x = x.transpose((1, 0, 2))
                t = t.transpose((1, 0, 2))

                model_master.reset_state(batch_size)
                for x_t in x:
                    model_master(x_t)

                y = []
                for t_t in t:
                    dummy_input = np.zeros_like(x_t)
                    y_t = model_master(dummy_input)
                    y.append(y_t)
                y = F.stack(y)
                loss = F.sigmoid_cross_entropy(y, t)
                loss_data = cuda.to_cpu(loss.data)
                acc = F.binary_accuracy(y, t)
                acc_data = cuda.to_cpu(acc.data)
                print('{}: {:0.5},\t{:1.5}'.format(
                    count_updates, float(acc_data), float(loss_data)))

    except KeyboardInterrupt:
        print('Ctrl+c')

    for process in processes:
        process.terminate()
