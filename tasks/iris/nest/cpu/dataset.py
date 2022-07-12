################################
# Dataset and data elaboration #
################################
import numpy as np


class Dataset:
    def __init__(self, dataset, time_stimulus):
        self.train_set = [Data(element, time_stimulus) for element in dataset['train_set']]
        self.test_set = [Data(element, time_stimulus) for element in dataset['test_set']]
        self.input_shape = np.array(dataset['train_set'][0][0]).shape

        # number of cells in the input data
        cells = self.input_shape[0] * self.input_shape[1]

        # creating a unique variable for spike train with source from training set
        self.spike_train = []
        samples = len(dataset['train_set'])
        for cell in range(cells):
            tmp = np.array([])
            for sample in range(samples):
                tmp = np.concatenate((tmp, self.train_set[sample].spikes_AER[cell] + ((time_stimulus['duration'] + time_stimulus['silence']) * np.array(sample))))
            self.spike_train.append(tmp)

        # creating a unique variable for spike train with source from test set
        self.spike_test = []
        samples = len(dataset['test_set'])
        for cell in range(cells):
            tmp = np.array([])
            for sample in range(samples):
                tmp = np.concatenate((tmp, self.test_set[sample].spikes_AER[cell] + ((time_stimulus['duration'] + time_stimulus['silence']) * np.array(sample))))
            self.spike_test.append(tmp)


class Data:
    def __init__(self, sample, time_stimulus):
        import math
        import random

        # separation data and label
        self.data = np.array(sample[0])  # image
        self.label = sample[1]  # label

        # spike generation with RATE encoding
        random.seed(0)

        self.spikes_AER = []

        rows, cols = self.data.shape
        for row in range(rows):
            for col in range(cols):
                rate = self.data[row, col]  # intensity data in cell
                if rate == 0:
                    self.spikes_AER.append(np.array([]))  # no stimulus
                else:
                    spike_sequence = []
                    poisson_isi = -math.log(1.0 - random.random()) / rate * 1000.0  # ms tau
                    spike_time = poisson_isi
                    while spike_time < time_stimulus['duration']:
                        spike_sequence.append(spike_time)
                        poisson_isi = -math.log(1.0 - random.random()) / rate * 1000.0  # ms tau
                        spike_time += poisson_isi
                    self.spikes_AER.append(np.array(spike_sequence))
        self.spinnaker_AER = {i: self.spikes_AER[i].tolist() for i in range(len(self.spikes_AER))}

    # plot function for:
    # direct data
    def plot_sample(self):
        import matplotlib.pyplot as plt
        plt.imshow(self.data[:, :])
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')
        plt.show()

    # spike train
    def plot_spike_train(self):
        import matplotlib.pyplot as plt
        plt.eventplot(self.spike_AER)
        plt.xlabel('Time (ms)')
        plt.ylabel('Pixel')
        plt.show()
