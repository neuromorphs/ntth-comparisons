import braille_helpers
import helpers
from spinnaker2 import snn, hardware, mapper

import spinnaker2.neuron_models.lif_neuron 
spinnaker2.neuron_models.lif_neuron.LIFApplication.profiling = True

import numpy as np

file_weights = "tasks/braille_letter_recognition/data/trained/layers_th1.npz"
file_dataset = "tasks/braille_letter_recognition/data/reading/data_th1_rpNull"

# Network definitions
input_size = 96
population_size = 450
#population_size = 50
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
weight_scale_factor = 0.001
reg_spikes = 0.004
reg_neurons = 0.000001

#w_input_pop, w_pop_out, w_pop_pop = braille_helpers.load_pt_weights(file_weights)
w_input_pop, w_pop_out, w_pop_pop = braille_helpers.load_np_weights(file_weights)
print(w_input_pop)
print("input", w_input_pop.shape)
print("ouput", w_pop_out.shape)
print("rec", w_pop_pop.shape)
w_input_pop = w_input_pop[:,:population_size]
w_pop_out = w_pop_out[:population_size,:]
w_pop_pop = w_pop_pop[:population_size,:population_size]
print(f"input_weights: max={w_input_pop.max()}, min={w_input_pop.min()}")
print(f"recurrent_weights: max={w_pop_pop.max()}, min={w_pop_pop.min()}")
print(f"output_weights: max={w_pop_out.max()}, min={w_pop_out.min()}")


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
'''
neuron_params = {
    "alpha_decay": float(np.exp(-time_step / tau_syn)),
    "threshold": 1.0,
}
'''

neuron_params_rsnn = {
    "reset":"reset_to_v_reset",
    "threshold":1.0/weight_scale_factor,
    "alpha_decay":float(np.exp(-time_step/tau_mem)),
    "i_offset":0,
    "v_reset":0,
    "exc_decay":float(np.exp(-time_step / tau_syn)),
    "inh_decay":float(np.exp(-time_step / tau_syn)),
    "t_refrac":0
}
print(neuron_params_rsnn)

neuron_params_output = {
    "reset":"reset_to_v_reset",
    "threshold":1e9,  # set to non-spiking
    "alpha_decay":float(np.exp(-time_step/tau_mem)),
    "i_offset":0,
    "v_reset":0,
    "exc_decay":float(np.exp(-time_step / tau_syn)),
    "inh_decay":float(np.exp(-time_step / tau_syn)),
    "t_refrac":0
}

pop_rsnn = snn.Population(
    size=population_size,
    neuron_model="lif_curr_exp_no_delay",
    params=neuron_params_rsnn,
    name="pop_rsnn",
)
pop_rsnn.set_max_atoms_per_core(10)

pop_output = snn.Population(
    size=output_size,
    neuron_model="lif_curr_exp_no_delay",
    params=neuron_params_output,
    name="pop_output",
    record=["v", "spikes"]
)

pop_output.set_max_atoms_per_core(10)

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
hw.run(net, timesteps, debug=False, sys_tick_in_s=10.e-3) # use larger time step to make sure all spikes are processed within sys_tick


# get voltage of last timestep of output neurons to find winner
voltages = pop_output.get_voltages()
v_last_timestep = [vs[-1] for vs in voltages.values()]
index_max_value = np.argmax(v_last_timestep)
print("Predicted label:", braille_helpers.letters[int(index_max_value)])
print("Actual label:", label)
if label == index_max_value:
    print("CORRECT PREDICTION")

spikes_rsnn = pop_rsnn.get_spikes()


import matplotlib.pyplot as plt
# voltage output layer
plt.figure()
times = np.arange(timesteps)
for vs,letter in zip(voltages.values(), braille_helpers.letters):
    if letter == label:
        plt.plot(times, vs, label=letter, lw=3)
    else:
        plt.plot(times, vs, label=letter, lw=1)
plt.legend()
plt.xlabel("time step")
plt.ylabel("voltage")
plt.title("voltage output layer")

# spikes recurrent layer
import spinnaker2
plt.figure()
indices, times = spinnaker2.helpers.spike_times_dict_to_arrays(spikes_rsnn)
plt.plot(times, indices, ".")
plt.xlim(0,timesteps)
plt.ylim(0,pop_rsnn.size)
plt.xlabel("time step")
plt.ylabel("neuron")
plt.title("spikes recurrent layer")

plt.show()
