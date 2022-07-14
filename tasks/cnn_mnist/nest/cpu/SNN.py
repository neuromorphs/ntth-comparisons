import pyNN.nest as pynn


class SNN:
    def __init__(self, data_shape, spike_train, model_ann, neuron_set):
        self.spike_train = spike_train

        layers_dim = []
        self.layers = []

        # initializing pyNN
        pynn.setup(timestep=1.0)

        # build input layer
        input_data_row, input_data_col = data_shape  # dimension input data
        number_neuron = input_data_row * input_data_col  # number neuron in the input layer
        layer_input = pynn.Population(number_neuron, pynn.SpikeSourceArray, {'spike_times': spike_train})

        # adding layer and data for next
        self.layers.append([layer_input])
        layers_dim.append({'data_shape': data_shape, 'number_neuron': number_neuron})

        layer_index = 0  # previous index layer
        for layer in model_ann.layers:
            print(layer.type)

            if layer.type == 'conv2d':
                # build convolutional layer
                kernel_rows, kernel_cols = layer.kernel_size

                # dimension previous layer
                input_data_row, input_data_col = layers_dim[layer_index]['data_shape']
                output_data_row, output_data_col = input_data_row - kernel_rows + 1, input_data_col - kernel_cols + 1
                number_neuron = output_data_row * output_data_col

                tmp_layer = []
                for filter in range(layer.filters):
                    layer_conv = pynn.Population(number_neuron, pynn.IF_curr_exp, neuron_set)  # build feature map

                    # feature map in the previously layer
                    feature_maps = range(len(self.layers[layer_index]))
                    for feature_map in feature_maps:
                        kernel_weights = layer.weights[filter][feature_map]
                        synapse_excit = []
                        synapse_inhib = []
                        for row in range(output_data_row):
                            for col in range(output_data_col):
                                index_neuron = col + row * output_data_col
                                for kernel_row in range(kernel_rows):
                                    for kernel_col in range(kernel_cols):
                                        # weight = kernel_weights[kernel_cols - 1 - kernel_col, kernel_rows - 1 - kernel_row]
                                        weight = kernel_weights[kernel_row, kernel_col]
                                        index_receptive_field = col + kernel_col + (row + kernel_row) * input_data_col
                                        if weight > 0:
                                            synapse_excit.append((index_receptive_field, index_neuron, weight, 1.0))
                                        elif weight < 0:
                                            synapse_inhib.append((index_receptive_field, index_neuron, weight, 1.0))
                        layer_input = self.layers[layer_index][feature_map]
                        if synapse_excit:
                            pynn.Projection(layer_input, layer_conv, pynn.FromListConnector(synapse_excit), receptor_type='excitatory')
                        if synapse_inhib:
                            pynn.Projection(layer_input, layer_conv, pynn.FromListConnector(synapse_inhib), receptor_type='inhibitory')
                    tmp_layer.append(layer_conv)
                self.layers.append(tmp_layer)
                layers_dim.append({'data_shape': (output_data_row, output_data_col), 'number_neuron': number_neuron})
            elif layer.type == 'pool2d':
                # build pooling layer
                pool_rows, pool_cols = layer.pool_size

                # dimension previous layer
                input_data_row, input_data_col = layers_dim[layer_index]['data_shape']
                output_data_row, output_data_col = int(input_data_row / pool_rows), int(input_data_col / pool_cols)
                number_neuron = output_data_row * output_data_col

                # feature map in the previously layer
                tmp_layer = []
                feature_maps = range(len(self.layers[layer_index]))
                for feature_map in feature_maps:
                    layer_pool = pynn.Population(number_neuron, pynn.IF_curr_exp, neuron_set)
                    weight = layer.weights[0, 0]
                    synapse = []

                    for row in range(output_data_row):
                        for col in range(output_data_col):
                            index_neuron = col + row * output_data_col
                            for pool_row in range(pool_rows):
                                for pool_col in range(pool_cols):
                                    index_receptive_field = col * pool_cols + pool_col + (
                                            row * pool_rows + pool_row) * input_data_col
                                    synapse.append((index_receptive_field, index_neuron, weight, 1.0))

                    layer_input = self.layers[layer_index][feature_map]
                    pynn.Projection(layer_input, layer_pool, pynn.FromListConnector(synapse),
                                    receptor_type='excitatory')
                    tmp_layer.append(layer_pool)
                self.layers.append(tmp_layer)
                layers_dim.append({'data_shape': (output_data_row, output_data_col), 'number_neuron': number_neuron})
            elif layer.type == 'dense':
                # build dense layer
                number_neuron = layer.output
                layer_dense = pynn.Population(number_neuron, pynn.IF_curr_exp, neuron_set)

                input_data_row, input_data_col = layers_dim[layer_index]['data_shape']
                number_input = input_data_row * input_data_col
                if number_input == 0:
                    number_input = input_data_row
                    weights = layer.weights

                    synapse_excit = []
                    synapse_inhib = []
                    for output in range(number_neuron):
                        for input in range(number_input):
                            weight = weights[output, input]
                            if weight > 0:
                                synapse_excit.append((input, output, weight, 1.0))
                            elif weight < 0:
                                synapse_inhib.append((input, output, weight, 1.0))
                    layer_input = self.layers[layer_index][0]
                    if synapse_excit:
                        pynn.Projection(layer_input, layer_dense, pynn.FromListConnector(synapse_excit), receptor_type='excitatory')
                    if synapse_inhib:
                        pynn.Projection(layer_input, layer_dense, pynn.FromListConnector(synapse_inhib), receptor_type='inhibitory')
                else:
                    dense_weights = layer.weights

                    feature_maps = range(len(self.layers[layer_index]))
                    for feature_map in feature_maps:
                        weights = dense_weights[:, feature_map * number_input:(feature_map + 1) * number_input]

                        synapse_excit = []
                        synapse_inhib = []
                        for output in range(number_neuron):
                            for input in range(number_input):
                                weight = weights[output, input]
                                if weight > 0:
                                    synapse_excit.append((input, output, weight, 1.0))
                                elif weight < 0:
                                    synapse_inhib.append((input, output, weight, 1.0))

                        layer_input = self.layers[layer_index][feature_map]
                        if synapse_excit:
                            pynn.Projection(layer_input, layer_dense, pynn.FromListConnector(synapse_excit), receptor_type='excitatory')
                        if synapse_inhib:
                            pynn.Projection(layer_input, layer_dense, pynn.FromListConnector(synapse_inhib), receptor_type='inhibitory')
                self.layers.append([layer_dense])
                layers_dim.append({'data_shape': (number_neuron, 0), 'number_neuron': number_neuron})
            layer_index += 1

    def plot_spike_train(self):
        # plot spike train in input of SNN
        import matplotlib.pyplot as plt
        plt.eventplot(self.spike_train)
        plt.show()

    def start_simulation(self, number_samples, time_stimulus):
        spike_layers = [layer[0] for layer in self.layers]

        [layer.record('spikes') for layer in spike_layers]

        time = time_stimulus['duration'] + time_stimulus['silence']  # time for sigle sample
        pynn.run(time * number_samples)

        spike_train = [layer.get_data().segments[0].spiketrains for layer in spike_layers]

        pynn.end()
        return spike_train
