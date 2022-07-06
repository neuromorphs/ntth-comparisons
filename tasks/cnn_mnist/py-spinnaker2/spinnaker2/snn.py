import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # use CPU for TF
import numpy as np
from spinnaker2 import snn, hardware
import tensorflow as tf
import tensorflow_datasets as tfds
import itertools
import helpers
import matplotlib.pyplot as plt


# Network definitions
conv2d_out_shape = (4,12,12)
conv2d_size = np.prod(conv2d_out_shape) # 576
dense1_size = 16
dense2_size = 10
delay = 0
timesteps = 50

# Load dataset
(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    as_supervised=True,
    with_info=True,
    )
print(ds_info)
ds = ds_test.take(10)
index = 8
input_image, label = next(itertools.islice(iter(tfds.as_numpy(ds)), index, None))
print("Actual label:", label)

# re-format input image
input_image_uint8 = input_image.astype(np.uint8)
input_image_uint8 = np.moveaxis(input_image_uint8,-1,0)
assert input_image_uint8.shape == (1,28,28) # HWC

####################################
# Load weights and convert to int8 #
####################################
tf_model = tf.keras.models.load_model('data/tf_model_v2.h5')

# conv2d
conv2d_weights = np.array(tf_model.weights[0])
conv2d_weights_scaled_int8 = helpers.scale_and_convert_weights_to_int8(conv2d_weights)

# dense
dense_weights = np.array(tf_model.weights[1])
dense_weights_scaled_int8 = helpers.scale_and_convert_weights_to_int8(dense_weights)
# reshape dense weights
dense_weights_scaled_int8 = helpers.reorder_dense_weights_after_conv2d(
        dense_weights_scaled_int8, channels=conv2d_out_shape[0],
        height=conv2d_out_shape[1], width=conv2d_out_shape[2])

# dense2
dense2_weights = np.array(tf_model.weights[2])
dense2_weights_scaled_int8 = helpers.scale_and_convert_weights_to_int8(dense2_weights)

######################
# Network Definition #
######################

##########
# Conv2D #
##########
max_theo_output_value = helpers.mla_conv2d_max_output_value(5,5,1,np.int8, np.uint8)
threshold = 100.
scale = threshold/max_theo_output_value

params = {
    "image": input_image_uint8, # CHW
    "weights": conv2d_weights_scaled_int8, # (H,W,CI,CO) format
    "scale": scale*5, # multiplier of weighted sum to I_offset
    "threshold": threshold, # spike threshold
    "stride_x" : 2,
    "stride_y" :2
    }

input_pop = snn.Population(
        size=conv2d_size,
        neuron_model="conv2d_if_neuron_rate_code",
        params=params,
        name="input_pop")

######################
# Dense Hidden Layer #
######################

max_theo_output_value = dense_weights_scaled_int8.shape[0]*np.iinfo(np.int8).max
neuron_params = {
        "threshold":max_theo_output_value/50,
        "alpha_decay":1.,
        }

pop1 = snn.Population(
        size=dense1_size,
        neuron_model="lif_no_delay",
        params=neuron_params,
        name="pop1")

conns = helpers.connection_list_from_dense_weights(dense_weights_scaled_int8, delay)

proj1 = snn.Projection(
        pre=input_pop,
        post=pop1,
        connections=conns)

######################
# Dense Output Layer #
######################

neuron_params_out = {
        "threshold":1.e9, # very high threshold so that it is never reached
        "alpha_decay":1., # no leakage
        }

pop2 = snn.Population(
        size=dense2_size,
        neuron_model="lif_no_delay",
        params=neuron_params_out,
        name="pop2", record=["spikes", "v"])

conns2 = helpers.connection_list_from_dense_weights(dense2_weights_scaled_int8, delay)

proj2 = snn.Projection(
        pre=pop1,
        post=pop2,
        connections=conns2)

net = snn.Network("my network")
net.add(input_pop, pop1, pop2, proj1, proj2)

#####################
# Run on SpiNNaker2 #
#####################

hw = hardware.SpiNNaker2Chip()
hw.run(net, timesteps)


# get voltage of last timestep of output neurons to find winner
voltages = pop2.get_voltages()
v_last_timestep = [vs[-1] for vs in voltages.values()]
index_max_value = np.argmax(v_last_timestep)
print("Predicted label:", index_max_value)
print("Actual label:", label)
if label == index_max_value:
    print("CORRECT PREDICTION")


# compare output of each layer to DNN
compare_layer_output = False
if compare_layer_output:
    # reference DNN in TF/numpy
    input_image_float = input_image.astype(float)/255.
    input_image_float = np.expand_dims(input_image_float, 0)
    conv2d_out = tf.nn.conv2d(input_image_float, conv2d_weights, strides=[1,2,2,1], padding="VALID")
    conv2d_out = np.array(conv2d_out)
    conv2d_out = conv2d_out.clip(min=0) # ReLU

    dense_out = np.matmul(conv2d_out.flatten(),dense_weights)
    dense_out = dense_out.clip(min=0) # ReLU

    dense2_out = np.matmul(dense_out, dense2_weights)

    # compare to SpiNNaker2
    # Conv2D
    spike_times = input_pop.get_spikes() # dict with neuron_ids as keys and list of spike times as value
    spike_counts = np.zeros(input_pop.size, dtype=np.int32)
    for neuron_id, spikes in spike_times.items():
        spike_counts[neuron_id] = len(spikes)
    print("Conv2D max spikes:", spike_counts.max())

    conv2d_out = np.moveaxis(conv2d_out,-1,1) # change from NHWC to NCHW
    out_flat = conv2d_out.flatten()

    plt.figure()
    plt.plot(out_flat, spike_counts, ".")
    plt.xlabel("TensorFlow activation")
    plt.ylabel("SpiNNaker2 spike count")
    plt.title("Conv2D")

    # Dense
    spike_times_dense = pop1.get_spikes() # dict with neuron_ids as keys and list of spike times as value
    spike_counts_dense = np.zeros(pop1.size, dtype=np.int32)
    for neuron_id, spikes in spike_times_dense.items():
        spike_counts_dense[neuron_id] = len(spikes)
    print("Dense max spikes:", spike_counts_dense.max())

    plt.figure()
    plt.plot(dense_out, spike_counts_dense, ".")
    plt.xlabel("TensorFlow activation")
    plt.ylabel("SpiNNaker2 spike count")
    plt.title("Dense (hidden layer)")

    # output layer
    plt.figure()
    times = np.arange(timesteps)
    for i, vs in voltages.items():
        plt.plot(times, vs, label=str(i))
    plt.xlim(0,timesteps)
    plt.xlabel("time step")
    plt.ylabel("voltage")
    plt.legend(title="Neuron")
    plt.title(f"SpiNNaker output (True label={label})")

    plt.figure()
    plt.plot(dense2_out, v_last_timestep, ".")
    plt.xlabel("TensorFlow: activation of output layer")
    plt.ylabel("SpiNNaker2: voltage at last timestep")
    plt.title("Output layer")
    plt.show()
