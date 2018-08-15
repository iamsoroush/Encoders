# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numba import jit, float64, int8


class HSAEncoder:
    """Implementation of HSA encoding algorithm.

    Init an object from this class by passing parameters, and get
    encoded signal in term of spike times by calling 'encode' method,
    and passing original signal to this method.
    """

    def __init__(self, filter_response=None, step=1, filter_amp=0.2):
        """Init an encoder object.

        Parameters
        ----------
        filter_response : :obj: 'np.ndarray' , 1d array. optional.
            Default: A gaussian signal with M=51, std=7.
            FIR filter as a window.

        step : int , optional.
            Default: 1.
            Steping used when moving filter on the signal.

        filter_amp : float or int, optional.
            Default: 1
            Amplitude of filter response, by increasing this parameter, number of spikes will decrease.
        """

        self.filter_response = filter_response
        self.step = step
        self.filter_amp = filter_amp
        self._last_spike_times = None
        self._last_signal = None

    @property
    def filter_response(self):
        return self._filter_response

    @filter_response.setter
    def filter_response(self, new_response):
        if new_response is None:
            self._filter_response = signal.gaussian(M=51, std=7)
        else:
            assert isinstance(new_response, np.ndarray), "'filter_response' must be of np.ndarray type."
            assert new_response.ndim == 1, "'filter_response must be a 1d array."
            self._filter_response = new_response

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, new_step):
        assert isinstance(new_step, int), "'step' should be of type 'int'."
        self._step = new_step

    @property
    def filter_amp(self):
        return self._filter_amp

    @filter_amp.setter
    def filter_amp(self, new_amp):
        assert isinstance(new_amp, (int, float)), "'filter_amp' must be of types [int, float]"
        self._filter_amp = new_amp
        self.filter_response = self.filter_response * new_amp
        return

    def encode(self, sgnl):
        """Encode a given signal based on init parameters.

        Parameters
        ----------
        sgnl : :obj: 'np.ndarray' , 1d array.
            Signal to be encoded.

        Notes
        -----
        The spike times will be save in self._last_spikes for later plottings.
        The encoding procedure is written in compiled mode using numba.jit decorator.
        """

        @jit(int8[:](float64[:], float64[:], int8), nopython=True, cache=True)
        def calc_spike_times(sgnl, filter_response, step):
            filter_size = filter_response.shape[0]
            sgnl_size = sgnl.shape[0]
            windowed_signal = sgnl[:filter_size].copy()
            spike_times = np.zeros(sgnl.shape, dtype=np.int8)
            for pointer in range(0, sgnl_size, step):
                if pointer > sgnl_size - filter_size - 1:
                    break
                else:
                    if np.all(windowed_signal >= filter_response):
                        windowed_signal -= filter_response
                        spike_times[pointer] = 1
                    windowed_signal = np.concatenate((windowed_signal[step:],
                                                      sgnl[filter_size + pointer: filter_size + pointer + step]))
            return spike_times

        assert isinstance(sgnl, np.ndarray), "'sgnl' must be of type numpy.ndarray"
        assert sgnl.ndim == 1, "'sgnl' must be 1d array."
        filter_response = self.filter_response
        step = self.step
        self._last_signal = sgnl.copy()
        spikes = calc_spike_times(sgnl=sgnl, filter_response=filter_response, step=step)
        self._last_spike_times = np.where(spikes == 1)[0]
        return spikes

    def plot(self):
        """Plot encoded version and original version of last signal."""

        assert self._last_signal is not None, "You must encode at least one signal to perform plotting."
        fig, [ax0, ax1] = plt.subplots(nrows=2, sharex=True)
        fig.figsize = (18, 12)
        ax0.plot(self._last_signal)
        ax1.eventplot(self._last_spike_times)
        ax1.set_yticks([1])
        plt.show()

    def decode(self):
        """Decodes last encoded signal and plots two signals together."""

        orig = self._last_signal
        encoded = self._last_spike_times
        decoded = np.zeros(orig.shape)
        for spike_time in encoded:
            decoded[spike_time: spike_time + len(self.filter_response)] += self.filter_response
        plt.plot(orig)
        plt.plot(decoded)
        plt.show()
