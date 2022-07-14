import tensorflow as tf
from dataset import *
from ANN import *
from SNN import *
import seaborn as sns

###########################################
# Load MNIST dataset and spike conversion #
###########################################

n_class = 10
(train_data, train_label), (test_data, test_label) = tf.keras.datasets.mnist.load_data()

train_data = train_data[0:5, :, :]
train_label = train_label[0:5]
test_data = test_data[0:5, :, :]
test_label = test_label[0:5]

# formatting dataset in convenient way for spike conversion
dataset = {'train_set': list(zip(train_data.tolist(), train_label.tolist())),
           'test_set': list(zip(test_data.tolist(), test_label.tolist()))}

# delete unused variable in RAM
del train_data, train_label, test_data, test_label

# duration of stimulus per sample in ms for spike conversion
time_stimulus = {'duration': 1000.0, 'silence': 20.0}
spike_MNIST = Dataset(dataset, time_stimulus)

#############################
# Load pretrained CNN model #
#############################

ann_name = 'mnist_cnn.h5'
ann_directory = f'ANN_model/{ann_name}'


def sReLU(inputs):
    S = 201.0
    tau_syn = 0.005
    S_tausyn = tf.multiply(S, tau_syn)
    return tf.multiply(S_tausyn, tf.math.maximum(0.0, inputs))


model_cnn = tf.keras.models.load_model(ann_directory, custom_objects={'sReLU': sReLU})
model_cnn = CNN(model_cnn)
del ann_name, ann_directory

#######################
# Simulation with SNN #
#######################

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

# creation of SNN from CNN model and insert the spike train in the input layer
model_snn = SNN(spike_MNIST.input_shape, spike_MNIST.spike_train, model_cnn, lif_parameter)
# NEST simulation with spike train input
number_samples = len(spike_MNIST.train_set)
spike_train = model_snn.start_simulation(number_samples, time_stimulus)

# prediction and accuracy of sample data
result = spike_train[-1]
prediction = []
time_sample = time_stimulus['duration'] + time_stimulus['silence']
for i in range(number_samples):
    tmp = []
    for cell in range(n_class):
        tmp.append(result[cell][(result[cell] > i * time_sample) & (result[cell] < (i + 1) * time_sample)].size)
    prediction.append(np.array(tmp).argmax())
accuracy = np.array([prediction[i] == spike_MNIST.train_set[i].label for i in range(number_samples)])
accuracy = accuracy.sum() / accuracy.size * 100
print(f'Accuracy train set {accuracy}% with {number_samples} sample')

confusion_matrix = np.zeros(shape=(n_class, n_class), dtype=int)
for i in range(len(spike_MNIST.train_set)):
    confusion_matrix[spike_MNIST.train_set[i].label, prediction[i]] += 1

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
confusion_matrix = confusion_matrix/normalization[:, np.newaxis]
plt.close('all')

size_title = 15
figsize = np.array((6.4, 4.8)) * 1.2

plt.figure(figsize=figsize)
plt.suptitle('Training set metrics', fontsize=size_title+2)
plt.subplot(3, 1, 1)
sns.heatmap(recall, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Recall', fontsize=size_title-3)
plt.subplot(3, 1, 2)
sns.heatmap(precision, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Precision', fontsize=size_title-3)
plt.subplot(3, 1, 3)
sns.heatmap(f1, annot=True)
plt.yticks([])
plt.ylabel('F1 score', fontsize=size_title-3)
plt.xlabel('Class', fontsize=size_title-3)
plt.savefig('metrics_trainset.png')

plt.figure(figsize=figsize)
plt.title(f'Confusion matrix Train set\nAccuracy Score: {accuracy:.2f}%', fontdict={'fontsize': size_title})
sns.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted', fontsize=size_title-3)
plt.ylabel('Actual', fontsize=size_title-3)
plt.savefig('confusion_matrix_trainset.png')


# creation of SNN from CNN model and insert the spike test in the input layer
model_snn = SNN(spike_MNIST.input_shape, spike_MNIST.spike_test, model_cnn, lif_parameter)
# NEST simulation with spike train input
number_samples = len(spike_MNIST.test_set)
spike_test = model_snn.start_simulation(number_samples, time_stimulus)

# prediction and accuracy of sample data
result = spike_test[-1]
prediction = []
time_sample = time_stimulus['duration'] + time_stimulus['silence']
for i in range(number_samples):
    tmp = []
    for cell in range(n_class):
        tmp.append(result[cell][(result[cell] > i * time_sample) & (result[cell] < (i + 1) * time_sample)].size)
    prediction.append(np.array(tmp).argmax())
accuracy = np.array([prediction[i] == spike_MNIST.test_set[i].label for i in range(number_samples)])
accuracy = accuracy.sum() / accuracy.size * 100
print(f'Accuracy test set {accuracy}% with {number_samples} sample')

confusion_matrix = np.zeros(shape=(n_class, n_class), dtype=int)
for i in range(len(spike_MNIST.test_set)):
    confusion_matrix[spike_MNIST.test_set[i].label, prediction[i]] += 1

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
confusion_matrix = confusion_matrix/normalization[:, np.newaxis]

size_title = 15
figsize = np.array((6.4, 4.8)) * 1.2

plt.figure(figsize=figsize)
plt.suptitle('Test set metrics', fontsize=size_title+2)
plt.subplot(3, 1, 1)
sns.heatmap(recall, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Recall', fontsize=size_title-3)
plt.subplot(3, 1, 2)
sns.heatmap(precision, annot=True)
plt.xticks([])
plt.yticks([])
plt.ylabel('Precision', fontsize=size_title-3)
plt.subplot(3, 1, 3)
sns.heatmap(f1, annot=True)
plt.yticks([])
plt.ylabel('F1 score', fontsize=size_title-3)
plt.xlabel('Class', fontsize=size_title-3)
plt.savefig('metrics_testset.png')

plt.figure(figsize=figsize)
plt.title(f'Confusion matrix Test set\nAccuracy Score: {accuracy:.2f}%', fontdict={'fontsize': size_title})
sns.heatmap(confusion_matrix, annot=True)
plt.xlabel('Predicted', fontsize=size_title-3)
plt.ylabel('Actual', fontsize=size_title-3)
plt.savefig('confusion_matrix_testset.png')

print()
