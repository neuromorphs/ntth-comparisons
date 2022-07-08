import torch
import pickle
import helpers
from spinnaker2 import snn, hardware

letter_written = [
    "Space",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]


def load_layers(file, map_location, requires_grad=True, variable=False):

    if variable:

        lays = file

        for ii in lays:
            ii.requires_grad = requires_grad

    else:

        lays = torch.load(file, map_location=map_location)

        for ii in lays:
            ii.requires_grad = requires_grad

    return lays


file_weights = "tasks/braille_letter_recognition/data/trained/layers_th1.pt"
file_dataset = "tasks/braille_letter_recognition/data/reading/data_th1_rpNull"

# Network definitions
input_size = 96
population_size = 450
output_size = 28

# params
delay = 0
scale = 5
time_bin_size = 8
nb_input_copies = 4
tau_mem = 0.06
tau_ratio = 10
fwd_weight_scale = 1
weight_scale_factor = 0.01
reg_spikes = 0.004
reg_neurons = 0.000001

dataset = pickle.load(open(file_dataset, "rb"))
nchan = 12
# 12 taxels (channels)

import random
import numpy as np

max_time = int(54 * 25)  # ms
bins = 1000  # [ms] # rescaling?

# read 1 sample

idx = random.randrange(0, len(dataset) - 1)
dat = dataset[idx]["events"][:]
events_array = np.zeros([nchan, round((max_time / time_bin_size) + 0.5), 2])
for taxel in range(len(dat)):
    for event_type in range(len(dat[taxel])):
        if dat[taxel][event_type]:
            indx = bins * (np.array(dat[taxel][event_type]))
            indx = np.array((indx / time_bin_size).round(), dtype=int)
            events_array[taxel, indx, event_type] = 1

events_array = np.reshape(
    np.transpose(events_array, (1, 0, 2)), (events_array.shape[1], -1)
)
selected_chans = 2 * nchan
label = letter_written[letter_written.index(dataset[idx]["letter"])]

spikes_dict_AER = {}

# build AER dict from the events_array
for timestep in range(len(events_array)):
    for channel in range(len(events_array[timestep])):
        if events_array[timestep][channel] > 0:
            # tile the input 4 times
            for i in range(nb_input_copies):
                key = int(i * channel)
                if key not in spikes_dict_AER:
                    spikes_dict_AER[key] = []
                spikes_dict_AER[key].append(timestep)

# Load weights and convert to int8
weights = torch.load(file_weights, map_location=torch.device("cpu"))

# weights[0] = input -> population [96,450]
w_input_pop = weights[0].cpu().detach().numpy()
w_input_pop = helpers.scale_and_convert_weights_to_int8(w_input_pop)
# weights[1] = population -> output [450, 28]
w_pop_out = weights[1].cpu().detach().numpy()
w_pop_out = helpers.scale_and_convert_weights_to_int8(w_pop_out)
# weights[2] = population -> population [450, 450]
w_pop_pop = weights[2].cpu().detach().numpy()
w_pop_pop = helpers.scale_and_convert_weights_to_int8(w_pop_pop)

# Network Definition
# The input is repeated 4 times "vertically"; 24 x 4 = 96
# Input needs to be AER for spike_list
pop_input = snn.Population(size=input_size, neuron_model="spike_list", params=spikes_dict_AER, name="pop_input")


time_step = time_bin_size*0.001
tau_syn = tau_mem/tau_ratio
neuron_params = {
    "alpha_decay": float(np.exp(-time_step/tau_syn)),
    "threshold"  : 1.0,
}

pop_rsnn = snn.Population(size=population_size, neuron_model="lif_no_delay", params=neuron_params, name="pop_rsnn")

pop_output = snn.Population(size=output_size, neuron_model="lif_no_delay", params=neuron_params, name="pop_output")

conn_input_rsnn = helpers.connection_list_from_dense_weights(w_input_pop, delay=0)
conn_recurrent = helpers.connection_list_from_dense_weights(w_pop_pop, delay=0)
conn_rsnn_output = helpers.connection_list_from_dense_weights(w_pop_out, delay=0)

proj_input_rsnn = snn.Projection(pre=pop_input, post=pop_rsnn, connections=conn_input_rsnn)
proj_recurrent = snn.Projection(pre=pop_rsnn, post=pop_rsnn, connections=conn_recurrent)
proj_rsnn_output = snn.Projection(pre=pop_rsnn, post=pop_output, connections=conn_rsnn_output)

net = snn.Network("my network")
net.add(pop_input, pop_rsnn, pop_output, proj_input_rsnn, proj_recurrent, proj_rsnn_output)

timesteps = len(range(0, max_time, time_bin_size))

hw = hardware.SpiNNaker2Chip()
hw.run(net, timesteps)


# get voltage of last timestep of output neurons to find winner
voltages = pop_output.get_voltages()
v_last_timestep = [vs[-1] for vs in voltages.values()]
index_max_value = np.argmax(v_last_timestep)
print("Predicted label:", letter_written[int(index_max_value)])
print("Actual label:", label)
if label == index_max_value:
    print("CORRECT PREDICTION")



