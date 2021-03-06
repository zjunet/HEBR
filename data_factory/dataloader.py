# -*- coding: utf-8 -*-


import numpy as np


class BatchLoader():
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.x = None
        self.y = None
        self.pointer = 0
        self.num_batch = 0

    # Shuffle the data
    def Shuffle(self, datalength):
        shuffle_indices = np.random.permutation(np.arange(datalength))
        return shuffle_indices

    def SplitBatches(self, data):
        datas = data[:self.num_batch * self.batch_size]
        reminder = data[self.num_batch * self.batch_size:]
        data_batches = np.split(datas, self.num_batch, 0)
        if reminder.shape[0] != 0:
            data_batches.append(reminder)
        return data_batches

    def load_data(self, x=None, y=None, shuffle=False):
        self.x = np.asarray(x, dtype=np.float)
        self.y = np.asarray(y, dtype=np.int)

        # Shuffle the data
        if shuffle:
            shuffle_indices = self.Shuffle(self.x.shape[0])
            self.x = self.x[shuffle_indices]
            self.y = self.y[shuffle_indices]

        # Split batches
        self.num_batch = int(self.x.shape[0] / self.batch_size)
        self.pointer = 0

        self.x_batches = self.SplitBatches(self.x)
        self.y_batches = self.SplitBatches(self.y)
        self.num_batch = len(self.x_batches)

    def next_batch(self):
        x_batch = self.x_batches[self.pointer]
        y_batch = self.y_batches[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batch
        return x_batch, y_batch

    def reset_pointer(self):
        self.pointer = 0