# -*- coding: utf-8 -*-
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

class HSAEncoder:
    """Implementation of HSA encoding algorithm.

    Init an object from this class by passing parameters, and get
    encoded signal in term of spike times by calling 'encode' method,
    and passing original signal to this method.
    """

    def __init__(self, filter_response=None, step=1, filter_amp=1):
        """Init an encoder object.

        Parameters
        ----------
        filter_response : :obj: 'np.ndarray' , optional.
            Default: A gaussian signal with M=51, std=7.
            FIR filter as a window.

        windowing_step : int , optional.
            Default: 1.
            Steping used when moving filter on the signal.
        """

        self.filter_response = filter_response
        self.step = step
        self.filter_amp = filter_amp
        self._last_spikes = None
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
        sgnl : :obj: 'np.ndarray'.
            Signal to be encoded.

        Notes
        -----

        """

        assert isinstance(sgnl, np.ndarray), "'sgnl' must be of type numpy.ndarray"
        filter_size = self.filter_response.shape[0]
        filter_response = self.filter_response
        step = self.step
        self._last_signal = sgnl.copy()
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
        self._last_spikes = spike_times
        spikes = np.zeros(sgnl.shape)
        spikes[spike_times] = 1
        return spikes

    def plot_spikes(self):
        """Plot encoded version and original version of last signal."""

        assert self._last_signal is not None, "You must encode at least one signal to perform plotting."
        fig, [ax0, ax1] = plt.subplots(2, 1)
        fig.figsize = (18, 12)
        ax0.plot(self._last_signal)
        ax1.eventplot(self._last_spikes)
        ax1.set_xlim(-0.1 * len(self._last_signal), 1.1 * len(self._last_signal))
        ax1.set_yticks([1])
        plt.show()
