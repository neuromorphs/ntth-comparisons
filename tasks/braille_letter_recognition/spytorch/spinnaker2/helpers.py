import numpy as np

def scale_and_convert_weights_to_int8(tf_weights):
    """
    convert weights from float to int8 with normalization
    """
    max_abs_weight = np.abs(tf_weights).max()
    weights_scaled = tf_weights*(127/max_abs_weight)
    return weights_scaled.astype(np.int8)

def reorder_dense_weights_after_conv2d(dense_weights, channels, height, width):
    """
    reorder dense weights that follow a conv2d layer.

    The MLA outputs the conv2d results in NCHW format. Instead, TensorFlow uses
    NHWC per default. Hence, after Flatten() the neuron order is different,
    which affects the weights of the follow-up dense layer.

    This function re-orders the 0-th dimension of the dense weights accordingly
    """
    assert dense_weights.shape[0] == channels*height*width

    indices = np.arange(dense_weights.shape[0], dtype=int)
    indices = indices.reshape(height, width, channels) # TF uses HWC
    indices = np.moveaxis(indices,-1,0) # turn to CHW
    indices = indices.flatten()

    return dense_weights[indices,:]

def connection_list_from_dense_weights(dense_weights, delay):
    conns = []
    for i in range(dense_weights.shape[0]):
        for j in range(dense_weights.shape[1]):
            conns.append([i,j,dense_weights[i,j],delay])
    return conns

def mla_conv2d_max_output_value(kernel_height, kernel_width, in_channels, dtype_in, dtype_kernel):
    max_in_value = np.iinfo(dtype_in).max
    max_kernel_value = np.iinfo(dtype_kernel).max
    return kernel_height * kernel_width * in_channels * max_in_value * max_kernel_value

