import numpy as np
from scipy.interpolate import interp1d
import scipy.sparse.linalg as sla
import matplotlib.pyplot as plt
from neuron import *

def get_fir_curve(currents, neuron_pop, timesteps=1000):
    spikes = np.zeros((len(currents),))

    for u_idx, u in enumerate(currents):
        num_spikes = 0
        for t in range(timesteps):
            spk = neuron_pop(u)
            num_spikes += spk[0]
        ### end for
        spikes[u_idx] = num_spikes / timesteps
    ### end for u_idx
    response_curve = interp1d(currents, spikes, 
                               bounds_error=False,
                               fill_value=(0,1.))
    return response_curve


def weight_dist(size=(250,2),low=-15,high=15,dtype=np.int8):
    return np.random.uniform(low=low, 
                             high=high,
                             size=size).astype(dtype)

def neuron_model(pop_size, mem_decay=0.9, exc_decay=0.5):
    return LifCurExp(pop_size, alpha_decay=mem_decay, exc_decay=exc_decay)

# def freq(hz, dt=1e-3):
#     return 2*np.pi*hz / dt

# freq_hz = 0.02
# osc_func = np.array([[1.0, freq(freq_hz)], [-freq(freq_hz), 1.0]])

def get_weights(pop_size, train_inputs, damp_val=0, freq_val=1):
    '''
    parameters:
    -----------

    neuron_model: Callable

        The neuron model being simulated.

    train_inputs : np.ndarray

        A num_samples x data_dim sampling of the input space

    weight_dist : Callable

       Distribution that can sample from the domain of the input weights.
    '''
    assert pop_size <= 250, f'SpiNNaker2 neuron populations have max 250 neurons, got {pop_size}'
    input_dim = train_inputs.shape[1]

    # 1. get the firing curve    
    training_currents = np.linspace(0,25,200)
    exc_decay=0.01
    training_pop = neuron_model(1, mem_decay=0.999, exc_decay=exc_decay)
    firing_rate = get_fir_curve(training_currents, training_pop)

    # 2. Randomly sample the input weights and compute the currents
    # we expect to see over the domain.
    input_weights = weight_dist(size=(pop_size, input_dim))
    exc_currents = np.einsum('pd,nd->pn', input_weights, train_inputs)

    # 3. Get the firing rates of the neurons (estimated) and
    # compute the decoding and feedback matrices.

    activity = firing_rate(exc_currents)

#     decoder_weights = train_inputs.T @ np.linalg.pinv(activity)
    decoder_weights = [sla.lsqr(activity.T, train_inputs[:,i], damp=damp_val)[0] for i in range(input_dim)]
    decoder_weights = np.array(decoder_weights)

    print('decoding error:', np.sqrt(np.mean(np.linalg.norm(train_inputs - (decoder_weights @ activity).T, axis=1)**2)))

    osc_func = np.array([[1.0, freq_val], [-freq_val, 1.0]])
    feedback_weights = input_weights @ osc_func @ decoder_weights
    feedback_weights = np.round(feedback_weights).astype(np.int8)

    return input_weights, decoder_weights, feedback_weights


def simulate(input_weights, decode_weights, feedback_weights,T=1e3):
   
    T = int(T)
    pop_size, input_dim  = input_weights.shape
    exc_decay = 0.5
    neuron_pop = neuron_model(pop_size, mem_decay=0.9, exc_decay=exc_decay)

    feedback_current = np.zeros((pop_size,))
    decoded_state = np.zeros((T, input_dim))

    times = np.arange(T)
    for t in times:
        stim_cur = np.zeros((pop_size,))
        if t < 2: # less than 15 msec, assuming dt=1e-3
            stim_cur = input_weights @ np.array([[0],[1]])

        spikes = neuron_pop(feedback_current/exc_decay + stim_cur.flatten())

        decoded_state[t,:] = decode_weights @ spikes
        feedback_current = feedback_weights @ spikes

    return times, decoded_state


if __name__=='__main__':

    thetas = np.linspace(0,2*np.pi, 1000)
    xs = np.vstack((np.cos(thetas), np.sin(thetas))).T

    pop_size = 100 
    np.random.seed(0)
    input_weights, decode_weights, feedback_weights = get_weights(pop_size, 
                                                                  xs,
                                                                  damp_val=2,
                                                                  freq_val=0.4)


    times, decoded_state = simulate(input_weights, 
                                    decode_weights,
                                    feedback_weights,
                                    T=1e3)



    est_freq=8e-2
    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.plot(times, decoded_state[:,0], label=r'$\hat{x}_{0}$')
    plt.plot(times, decoded_state[:,1], label=r'$\hat{x}_{1}$')
    plt.xticks([])
    plt.ylabel('SpiNNaker2 Neuron Model')
    plt.legend()

    plt.subplot(2,1,2)
    plt.plot(times, np.sin(est_freq*times), label=r'$x_{0}$')
    plt.plot(times, np.cos(est_freq*times), label=r'$x_{1}$')
    plt.xlabel('Timestep (integer)')
    plt.ylabel('Desired Behaviour')
    plt.legend()
    plt.suptitle('Simple Oscillator with Signed 8-bit weights')

    plt.figure()
    plt.plot(decoded_state[:,0], decoded_state[:,1])
    plt.title('Orbit of Simple Oscillator with Signed 8-bit weights')

    plt.show()
### end main
