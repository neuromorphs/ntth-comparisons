import matplotlib.pyplot as plt
import numpy as np


##########################
# CNN feature extraction #
##########################

class CNN:
    def __init__(self, model):
        # initialization layer CNN
        self.layers = []

        flag = False
        for layer in model.layers:
            if 'flatten' not in layer.name and 'input' not in layer.name:
                self.layers.append(Layer(layer))
            elif 'flatten' in layer.name:
                flag = True
            if 'dense' in layer.name and flag:
                """
                In order to use the dense layer in a correct way for transfer learning,
                is necessary to rearrange the column of weigths after the flatten layer
                """
                flag = False
                input_neuron = self.layers[-1].input  # input neuron from the feature maps
                output_neuron = self.layers[-1].output  # neuron in the first dense layer
                number_filters = self.layers[-3].filters  # number of filter in the last convolutional layer

                dimension_map = int(input_neuron / number_filters)  # dimension of single feature map

                dense_weights = self.layers[-1].weights
                tmp = np.zeros((output_neuron, input_neuron))

                for feature_map in range(number_filters):
                    for cell in range(dimension_map):
                        tmp[:, (cell + feature_map * dimension_map)] = dense_weights[:, feature_map + cell * number_filters]
                self.layers[-1].weights = tmp  # update structure weights


class DNN:
    def __init__(self, model):
        # initialization layer DNN
        self.layers = []

        for layer in model.layers:
            if 'input' not in layer.name:
                self.layers.append(Layer(layer))

class Layer:
    def __init__(self, layer):
        layer_type = layer.name
        # extraction information base on type of layer
        if 'conv2d' in layer_type:
            self.type = 'conv2d'  # type of layer

            shape = layer.get_weights()[0].shape  # structure convolutional layer

            self.kernel_size = (shape[0], shape[1])  # kernel size dimension rows, cols
            self.input_map = shape[2]  # number input feature map
            self.filters = shape[3]  # number output feature map

            # get weigths layer
            self.weights = []
            tmp_weights = layer.get_weights()[0]
            for index_filter in range(self.filters):
                tmp = []
                for index_input in range(self.input_map):
                    tmp.append(np.array(tmp_weights[:, :, index_input, index_filter]))
                self.weights.append(tmp)
            self.boxplot = boxplot_data_extraction(tmp_weights)

        elif 'max_pooling2d' in layer_type or 'average_pooling2d' in layer_type:
            self.type = 'pool2d'  # type of layer

            self.pool_size = layer.pool_size
            self.weights = np.ones(self.pool_size) * 1 / (self.pool_size[0] * self.pool_size[1])
        elif 'dense' in layer_type:
            self.type = 'dense'  # type of layer

            shape = layer.get_weights()[0].shape  # structure dense layer

            self.input = shape[0]
            self.output = shape[1]

            # get weigths layer
            tmp_weights = layer.get_weights()[0]
            self.weights = np.transpose(tmp_weights)
            self.boxplot = boxplot_data_extraction(tmp_weights)


def boxplot_data_extraction(weights):
    boxplot_data = plt.boxplot(np.abs(weights.flatten()))
    whisker_lower = boxplot_data['whiskers'][0].get_ydata()[1]
    quartile_lower = boxplot_data['boxes'][0].get_ydata()[1]
    median = boxplot_data['medians'][0].get_ydata()[1]
    quartile_upper = boxplot_data['boxes'][0].get_ydata()[2]
    whisker_upper = boxplot_data['whiskers'][1].get_ydata()[1]
    parameters = [whisker_lower, quartile_lower, median, quartile_upper, whisker_upper]
    return parameters
