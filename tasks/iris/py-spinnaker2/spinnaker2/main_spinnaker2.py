import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import random
from dataset import *
import tensorflow as tf
from ANN import *
# from SNN import *
from spinnaker2 import snn, hardware
import seaborn as sns

##########################################
# Load IRIS dataset and spike conversion #
##########################################
n_class = 3
n_samples = 150
n_features = 4

dataset = tfds.as_numpy(tfds.load(name='iris', split='train'))

data = np.array([sample['features'] for sample in dataset])
data[:, -1] = 8 * data[:, -1] / data[:, -1].max()
data = 255 * data / data.max()
data = data.astype(int)
data = data.reshape((n_samples, n_features, 1))
label = np.array([sample['label'] for sample in dataset])
sort_index = np.argsort(label)
dataset = list(zip(data[sort_index], label[sort_index]))

train_set = []
test_set = []
for i in range(n_class):
    tmp = dataset[0 + 50 * i:10 + 50 * i]
    for sample in tmp:
        test_set.append(sample)
    tmp = dataset[10 + 50 * i:50 + 50 * i]
    for sample in tmp:
        train_set.append(sample)
del dataset, data, label, sort_index, tmp, i, sample
random.seed(0)
random.shuffle(train_set)
random.shuffle(test_set)

# formatting dataset in convenient way for spike conversion
dataset = {'train_set': train_set,
           'test_set': test_set}

# duration of stimulus per sample in ms for spike conversion
time_stimulus = {'duration': 1000.0, 'silence': 20.0}
spike_IRIS = Dataset(dataset, time_stimulus)

#############################
# Load pretrained CNN model #
#############################

ann_name = 'iris_dnn_4l128n_95_93.h5'
ann_directory = f'ANN_model/{ann_name}'


def sReLU(inputs):
    S = 201.0
    tau_syn = 0.005
    S_tausyn = tf.multiply(S, tau_syn)
    return tf.multiply(S_tausyn, tf.math.maximum(0.0, inputs))


model_ann = tf.keras.models.load_model(ann_directory, custom_objects={'sReLU': sReLU})
model_ann = DNN(model_ann)
del ann_name, ann_directory

##############################
# Simulation with Spinnaker2 #
##############################
# parameter of neuron LIF
lif_parameter = {'cm': 0.25,  # nF
                 'i_offset': 0.1,  # nA
                 'tau_m': 20.0,  # ms
                 'tau_refrac': 1.0,  # ms
                 'tau_syn_E': 5.0,  # ms
                 'tau_syn_I': 5.0,  # ms
                 'v_reset': -65.0,  # mV
                 'v_rest': -65.0,  # mV
                 'v_thresh': -50.0  # mV
                 }
time_step = 1e-3
neuron_params = {'reset': "reset_to_v_reset",
                 'threshold': lif_parameter['v_thresh'],
                 'alpha_decay': float(np.exp(-time_step / lif_parameter['tau_m'])),
                 'i_offset': lif_parameter['i_offset'],
                 'v_reset': lif_parameter['v_reset'],
                 'exc_decay': float(np.exp(-time_step / lif_parameter['tau_syn_E'])/lif_parameter['cm']),
                 'inh_decay': float(np.exp(-time_step / lif_parameter['tau_syn_I'])/lif_parameter['cm']),
                 't_refrac': int(lif_parameter['tau_refrac'])
                 }

# train set inference
number_samples = len(spike_IRIS.train_set)
prediction = []
for index_sample in range(number_samples):
    model_snn = []
    model_snn.append(snn.Population(size=n_features, neuron_model="spike_list",
                                    params=spike_IRIS.train_set[index_sample].spinnaker_AER, name="inputs"))
    index_layer = 0
    for layer in model_ann.layers:
        spike_layer = snn.Population(size=layer.output, neuron_model="lif_curr_exp", params=neuron_params,
                                     name=f'{layer.type}{index_layer}')
        index_layer += 1
        conn_synapses = []
        for i in range(layer.input):
            for o in range(layer.output):
                conn_synapses.append([i, o, layer.weights[o, i], 1.0])
        spike_proje = snn.Projection(pre=model_snn[-1], post=spike_layer, connections=conn_synapses)
        model_snn.append(spike_proje)
        model_snn.append(spike_layer)

    net = snn.Network("sDNN")
    for component in model_snn:
        net.add(component)

    timesteps = len(range(int(time_stimulus['duration'])))

    hw = hardware.SpiNNaker2Chip()
    hw.run(net, timesteps)

    output_spikes = model_snn[-1].get_spikes()
    prediction.append(np.argmax([len(output_spikes[k]) for k in output_spikes.keys()]))

accuracy = np.array([prediction[i] == spike_IRIS.train_set[i].label for i in range(number_samples)])
accuracy = accuracy.sum() / accuracy.size * 100
print(f'Accuracy train set {accuracy}% with {number_samples} sample')

confusion_matrix = np.zeros(shape=(n_class, n_class), dtype=int)
for i in range(len(spike_IRIS.train_set)):
    confusion_matrix[spike_IRIS.train_set[i].label, prediction[i]] += 1

recall = np.zeros(shape=n_class)
precision = np.zeros(shape=n_class)
f1 = np.zeros(shape=n_class)
for i in range(n_class):
    recall[i] = confusion_matrix[i, i] / sum(confusion_matrix[:, i])
    precision[i] = confusion_matrix[i, i] / sum(confusion_matrix[i, :])
    f1[i] = 2 * (recall[i] * precision[i]) / (recall[i] + precision[i])
recall = recall.reshape((1, n_class))
precision = precision.reshape((1, n_class))
f1 = f1.reshape((1, n_class))

normalization = confusion_matrix.sum(axis=1)
confusion_matrix = confusion_matrix / normalization[:, np.newaxis]
plt.close('all')

size_title = 15
figsize = np.array((6.4, 4.8)) * 1.2

plt.figure(figsize=figsize)
plt.suptitle('Training set metrics', fontsize=size_title + 2)
plt.subplot(3, 1, 1)
sns.heatmap(recall, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Recall', fontsize=size_title - 3)
plt.subplot(3, 1, 2)
sns.heatmap(precision, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Precision', fontsize=size_title - 3)
plt.subplot(3, 1, 3)
sns.heatmap(f1, annot=True)
plt.yticks([])
plt.ylabel('F1 score', fontsize=size_title - 3)
plt.xlabel('Class', fontsize=size_title - 3)
plt.savefig('performance/metrics_trainset_spinnaker2.png')

plt.figure(figsize=figsize)
plt.title(f'Confusion matrix Train set\nAccuracy Score: {accuracy:.2f}%', fontdict={'fontsize': size_title})
sns.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted', fontsize=size_title - 3)
plt.ylabel('Actual', fontsize=size_title - 3)
plt.savefig('performance/confusion_matrix_trainset_spinnaker2.png')

# test set inference
number_samples = len(spike_IRIS.test_set)
prediction = []
for index_sample in range(number_samples):
    model_snn = []
    model_snn.append(snn.Population(size=n_features, neuron_model="spike_list",
                                    params=spike_IRIS.test_set[index_sample].spinnaker_AER, name="inputs"))
    index_layer = 0
    for layer in model_ann.layers:
        spike_layer = snn.Population(size=layer.output, neuron_model="lif_curr_exp", params=neuron_params,
                                     name=f'{layer.type}{index_layer}')
        index_layer += 1
        conn_synapses = []
        for i in range(layer.input):
            for o in range(layer.output):
                conn_synapses.append([i, o, layer.weights[o, i], 1.0])
        spike_proje = snn.Projection(pre=model_snn[-1], post=spike_layer, connections=conn_synapses)
        model_snn.append(spike_proje)
        model_snn.append(spike_layer)

    net = snn.Network("sDNN")
    for component in model_snn:
        net.add(component)

    timesteps = len(range(int(time_stimulus['duration'])))

    hw = hardware.SpiNNaker2Chip()
    hw.run(net, timesteps)

    output_spikes = model_snn[-1].get_spikes()
    prediction.append(np.argmax([len(output_spikes[k]) for k in output_spikes.keys()]))

accuracy = np.array([prediction[i] == spike_IRIS.test_set[i].label for i in range(number_samples)])
accuracy = accuracy.sum() / accuracy.size * 100
print(f'Accuracy test set {accuracy}% with {number_samples} sample')

confusion_matrix = np.zeros(shape=(n_class, n_class), dtype=int)
for i in range(len(spike_IRIS.test_set)):
    confusion_matrix[spike_IRIS.test_set[i].label, prediction[i]] += 1

recall = np.zeros(shape=n_class)
precision = np.zeros(shape=n_class)
f1 = np.zeros(shape=n_class)
for i in range(n_class):
    recall[i] = confusion_matrix[i, i] / sum(confusion_matrix[:, i])
    precision[i] = confusion_matrix[i, i] / sum(confusion_matrix[i, :])
    f1[i] = 2 * (recall[i] * precision[i]) / (recall[i] + precision[i])
recall = recall.reshape((1, n_class))
precision = precision.reshape((1, n_class))
f1 = f1.reshape((1, n_class))

normalization = confusion_matrix.sum(axis=1)
confusion_matrix = confusion_matrix / normalization[:, np.newaxis]
plt.close('all')

size_title = 15
figsize = np.array((6.4, 4.8)) * 1.2

plt.figure(figsize=figsize)
plt.suptitle('Test set metrics', fontsize=size_title + 2)
plt.subplot(3, 1, 1)
sns.heatmap(recall, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Recall', fontsize=size_title - 3)
plt.subplot(3, 1, 2)
sns.heatmap(precision, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Precision', fontsize=size_title - 3)
plt.subplot(3, 1, 3)
sns.heatmap(f1, annot=True)
plt.yticks([])
plt.ylabel('F1 score', fontsize=size_title - 3)
plt.xlabel('Class', fontsize=size_title - 3)
plt.savefig('performance/metrics_testset_spinnaker2.png')

plt.figure(figsize=figsize)
plt.title(f'Confusion matrix Test set\nAccuracy Score: {accuracy:.2f}%', fontdict={'fontsize': size_title})
sns.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted', fontsize=size_title - 3)
plt.ylabel('Actual', fontsize=size_title - 3)
plt.savefig('performance/confusion_matrix_testset_spinnaker2.png')

# plt.show()
