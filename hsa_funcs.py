import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numba
from time import time


def pure_python_encoder(sgnl, filter_response, step):
    assert isinstance(sgnl, np.ndarray), "'input_signal' must be of type numpy.ndarray"
    assert isinstance(filter_response, np.ndarray), "'filter_response' must be of type numpy.ndarray"
    assert isinstance(step, int), "'step' must be of type int"

    filter_size = filter_response.shape[0]
    sgnl_size = sgnl.shape[0]
    windowed_signal = sgnl[:filter_size].copy()
    spike_times = list()
    for pointer in range(0, sgnl_size, step):
        if pointer > sgnl_size - filter_size - 1:
            break
        else:
            if np.all(windowed_signal >= filter_response):
                windowed_signal -= filter_response
                spike_times.append(pointer)
            windowed_signal = np.array(windowed_signal[step:].tolist() +
                                       sgnl[filter_size + pointer: filter_size + pointer + step].tolist())
    return spike_times


@numba.jit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64), nopython=True, cache=True)
def numba_encoder(sgnl, filter_response, step):
    filter_size = filter_response.shape[0]
    sgnl_size = sgnl.shape[0]
    windowed_signal = sgnl[:filter_size].copy()
    spike_times = np.zeros(sgnl.shape)
    for pointer in range(0, sgnl_size, step):
        if pointer > sgnl_size - filter_size - 1:
            break
        else:
            if np.all(windowed_signal >= filter_response):
                windowed_signal -= filter_response
                spike_times[pointer] = 1.0
            windowed_signal = np.concatenate((windowed_signal[step:],
                                              sgnl[filter_size + pointer: filter_size + pointer + step]))
    return spike_times


sgnl = np.sin(np.linspace(0, 10, 1000))
f_response = 0.1 * signal.gaussian(M=51, std=7)
t = time()
for _ in range(100):
    pure_python_encoder(sgnl=sgnl, filter_response=f_response, step=1)
print((time() - t)/100)

t = time()
for _ in range(100):
    numba_encoder(sgnl=sgnl, filter_response=f_response, step=1)
print((time() - t)/100)