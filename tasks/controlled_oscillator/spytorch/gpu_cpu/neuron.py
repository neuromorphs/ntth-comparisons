import numpy as np

class LifCurExp:
    def __init__(self, pop_size, alpha_decay=0.9, threshold=10., v_reset=-0.2,
            exc_decay = 0.5, inh_decay = 0.5, t_refrac=0):
        assert pop_size <= 250, f'Population max size is 250, got {pop_size}'

        self.mem_decay = alpha_decay
        self.exc_decay = exc_decay
        self.inh_decay = inh_decay

        self.v = np.zeros((pop_size,), dtype=np.float32) 
        self.i_syn_exc =  np.zeros((pop_size,), dtype=np.float32)
        self.threshold = threshold
        self.v_reset = v_reset

    def __call__(self, input_current):

        self.i_syn_exc = self.exc_decay * self.i_syn_exc + input_current
        self.v = self.mem_decay * self.v + self.i_syn_exc

        spikes = self.v > self.threshold

        self.v[spikes] = self.v_reset

        return spikes
