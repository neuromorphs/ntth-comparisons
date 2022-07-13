import pickle
import random
import numpy as np
import helpers

# import torch

letters = [
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


def get_random_braille_sample(file_dataset, time_bin_size, n_input_copies, max_time):
    dataset = pickle.load(open(file_dataset, "rb"))

    n_channels = 12
    # 12 taxels (channels)

    bins = 1000  # [ms] rescaling

    # read 1 sample

    idx = random.randrange(0, len(dataset) - 1)
    dat = dataset[idx]["events"][:]
    events_array = np.zeros([n_channels, round((max_time / time_bin_size) + 0.5), 2])
    for taxel in range(len(dat)):
        for event_type in range(len(dat[taxel])):
            if dat[taxel][event_type]:
                indx = bins * (np.array(dat[taxel][event_type]))
                indx = np.array((indx / time_bin_size).round(), dtype=int)
                events_array[taxel, indx, event_type] = 1

    events_array = np.reshape(
        np.transpose(events_array, (1, 0, 2)), (events_array.shape[1], -1)
    )
    selected_channels = 2 * n_channels
    label = letters[letters.index(dataset[idx]["letter"])]

    spikes_dict_AER = {}

    # build AER dict from the events_array
    for timestep in range(len(events_array)):
        for channel in range(len(events_array[timestep])):
            if events_array[timestep][channel] > 0:
                # tile the input 4 times
                for i in range(n_input_copies):
                    key = int(i * channel)
                    if key not in spikes_dict_AER:
                        spikes_dict_AER[key] = []
                    spikes_dict_AER[key].append(timestep)

    return spikes_dict_AER, label


def scale_and_convert_weights_to_int8(*args, scaling_factor=None):
    """
    convert weights from float to int8 with normalization
    - now supports multiple arrays for input.
    - can specify scaling factor
    """
    if scaling_factor == None:
        max_abs_weight = 0
        for tf_weights in args:
            tmp = np.abs(tf_weights).max()
            if tmp > max_abs_weight:
                max_abs_weight = tmp
        scaling_factor = 127 / max_abs_weight

    if len(args) > 1:
        weights_scaled = []

    for tf_weights in args:
        tmp = np.rint(tf_weights * scaling_factor)
        if len(args) > 1:
            weights_scaled.append(tmp.astype(np.int8))
        else:
            weights_scaled = tmp.astype(np.int8)

    return weights_scaled, scaling_factor


def load_np_weights(file_weights):
    weights = np.load(file_weights)
    w_input_pop = weights["w_input_pop"]
    w_pop_out = weights["w_pop_out"]
    w_pop_pop = weights["w_pop_pop"]

    (w_input_pop, w_pop_pop), pop_th_scaling = scale_and_convert_weights_to_int8(
        w_input_pop, w_pop_pop
    )

    w_pop_out, out_th_scaling = scale_and_convert_weights_to_int8(w_pop_out)

    return w_input_pop, w_pop_out, w_pop_pop, pop_th_scaling, out_th_scaling


"""
def load_pt_weights(file_weights):
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

    return w_input_pop, w_pop_out, w_pop_pop
"""
