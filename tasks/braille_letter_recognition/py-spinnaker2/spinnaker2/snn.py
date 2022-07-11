import torch
import braille_helpers
import helpers
from spinnaker2 import snn, hardware
import numpy as np

file_weights = "tasks/braille_letter_recognition/data/trained/layers_th1.npz"
file_dataset = "tasks/braille_letter_recognition/data/reading/data_th1_rpNull"

# Network definitions
input_size = 96
population_size = 450
output_size = 28

# params
delay = 0
scale = 5
time_bin_size = 8
n_input_copies = 4
tau_mem = 0.06
tau_ratio = 10
fwd_weight_scale = 1
weight_scale_factor = 0.01
reg_spikes = 0.004
reg_neurons = 0.000001

#w_input_pop, w_pop_out, w_pop_pop = braille_helpers.load_pt_weights(file_weights)
w_input_pop, w_pop_out, w_pop_pop = braille_helpers.load_np_weights(file_weights)

# Get 1 random sample from the dataset
max_time = int(54 * 25)  # ms
spikes_dict_AER, label = braille_helpers.get_random_braille_sample(
    file_dataset, time_bin_size, n_input_copies, max_time
)

# Network Definition
# The input is repeated 4 times "vertically"; 24 x 4 = 96
# Input needs to be AER for spike_list
pop_input = snn.Population(
    size=input_size, neuron_model="spike_list", params=spikes_dict_AER, name="pop_input"
)


time_step = time_bin_size * 0.001
tau_syn = tau_mem / tau_ratio
neuron_params = {
    "alpha_decay": float(np.exp(-time_step / tau_syn)),
    "threshold": 1.0,
}

pop_rsnn = snn.Population(
    size=population_size,
    neuron_model="lif_no_delay",
    params=neuron_params,
    name="pop_rsnn",
)

pop_output = snn.Population(
    size=output_size,
    neuron_model="lif_no_delay",
    params=neuron_params,
    name="pop_output",
)

conn_input_rsnn = helpers.connection_list_from_dense_weights(w_input_pop, delay=0)
conn_recurrent = helpers.connection_list_from_dense_weights(w_pop_pop, delay=0)
conn_rsnn_output = helpers.connection_list_from_dense_weights(w_pop_out, delay=0)

proj_input_rsnn = snn.Projection(
    pre=pop_input, post=pop_rsnn, connections=conn_input_rsnn
)
proj_recurrent = snn.Projection(pre=pop_rsnn, post=pop_rsnn, connections=conn_recurrent)
proj_rsnn_output = snn.Projection(
    pre=pop_rsnn, post=pop_output, connections=conn_rsnn_output
)

net = snn.Network("my network")
net.add(
    pop_input, pop_rsnn, pop_output, proj_input_rsnn, proj_recurrent, proj_rsnn_output
)

timesteps = len(range(0, max_time, time_bin_size))

hw = hardware.SpiNNaker2Chip()
hw.run(net, timesteps)


# get voltage of last timestep of output neurons to find winner
voltages = pop_output.get_voltages()
v_last_timestep = [vs[-1] for vs in voltages.values()]
index_max_value = np.argmax(v_last_timestep)
print("Predicted label:", braille_helpers.letters[int(index_max_value)])
print("Actual label:", label)
if label == index_max_value:
    print("CORRECT PREDICTION")
