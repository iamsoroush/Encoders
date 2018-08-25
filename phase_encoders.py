# -*- coding: utf-8 -*-
"""Phase encoders implementation.

Each encoder's source paper is mentioned in encoder's docstring.

"""


import numpy as np
import matplotlib.pyplot as plt


class LPEncoder:
    """Latency-Phase encoder unit.

    This is implementation of encoding scheme proposed in:
     https://www.mitpressjournals.org/doi/abs/10.1162/NECO_a_00395


    Notes
    -----
        Inspired by the information processing in the retina, the visual information
         is encoded into the responses of neurons using precisely timed action potentials.
        The intensity value of each pixel is converted to a precisely timed spike via a
         latency encoding scheme.
        strong stimulation leads to short spike latency, and weak stimulation results
         in a long reaction time.
        Initialize the encoder, and use 'encode' method, encoder will output n_input/n_rf spike trains
         with spike times in miliseconds.

    """

    def __init__(self, n_rf=4, t_max=1., alpha=1., amp=1., freq=40., phi_0=0., delta_phi=None):
        """Instantiate an encoder object.

        Args
        ----
        n_rf (int): Number of photoreceptors in each ganglion cell.
        t_max (float): Length of encoding window used for creating temporal data from static input, in seconds.
        alpha (float): Scaling factor used in logarithmic transformation function.
        amp (float): Amplitude of SMO function.
        freq (float): Sub-threshold membrane oscillation's frequency in Hz. Default is gamma frequency, 40 Hz.
        phi_0 (float): Initial phase for calculating SNO phases.
        delta_phi (float): SMO phase difference between two adjacent photoreceptors, must be lower than 2*pi/n_rf .
            default value is 2*pi/n_rf .

        """
        self.n_rf = n_rf
        self.t_max = t_max
        self.alpha = alpha
        self.amp = amp
        self.freq = freq
        self.phi_o = phi_0
        if delta_phi is None:
            self.delta_phi = 2 * np.pi / n_rf
        else:
            self.delta_phi = delta_phi

    @property
    def n_rf(self):
        return self._n_rf

    @n_rf.setter
    def n_rf(self, new_nrf):
        assert isinstance(new_nrf, (int, float)), "'n_rf' must be of type int or float."
        self._n_rf = new_nrf

    @property
    def t_max(self):
        return self._t_max

    @t_max.setter
    def t_max(self, new_tmax):
        assert isinstance(new_tmax, float), "'t_max' must be of type float."
        self._t_max = new_tmax

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha):
        assert isinstance(new_alpha, float), "'alpha' must be of type float."
        self._alpha = new_alpha

    @property
    def amp(self):
        return self._amp

    @amp.setter
    def amp(self, new_amp):
        assert isinstance(new_amp, float), "'amp' must be of type float."
        self._amp = new_amp

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, new_freq):
        assert isinstance(new_freq, (int, float)), "'freq' must be of type int or float."
        self._freq = new_freq

    @property
    def phi_0(self):
        return self._phi_0

    @phi_0.setter
    def phi_0(self, new_phi0):
        assert isinstance(new_phi0, float), "'phi_0' must be of type float"
        self._phi_0 = new_phi0

    @property
    def delta_phi(self):
        return self._delta_phi

    @delta_phi.setter
    def delta_phi(self, new_deltaphi):
        assert isinstance(new_deltaphi, float), "'delta_phi' must be of type float."
        self._delta_phi = new_deltaphi



    def encode(self, input_array):
        """Encodes input array to output spike trains.

        Args
        ----
        input_array (:obj: 'np.ndarray'): Shape must be shape(n_input,) .

        :returns Encoded spike train in shape (n_output_cells, n_rf) which contains
         spike time occurrences for each ganglion cell.
        """
        assert isinstance(input_array, np.ndarray), "'input_array' must be of type np.ndarray"
        assert input_array.ndim == 1, "'input_array' must be 1-d tensor"
        assert input_array.shape[0] % self.n_rf == 0, "'input_array' dimensionality should match with n_rf."
        assert input_array.dtype == np.float64, "'input_array' must be of dtype np.float64"
        g_cell = GanglionCell(n_rf=self.n_rf, t_max=self.t_max, alpha=self.alpha, amp=self.amp,
                              freq=self.freq, phi_0=self.phi_o, delta_phi=self.delta_phi)
        n_input = input_array.shape[0]
        fields = np.split(input_array, n_input/self.n_rf)
        encoded = list()
        for row in fields:
            encoded.append(g_cell.encode(stimulation=row))
        encoded = np.array(encoded, dtype=np.int64)
        plt.style.use('ggplot')
        plt.eventplot(encoded)
        plt.yticks([i for i in range(0, len(encoded))])
        plt.xlabel('Simulation interval')
        plt.ylabel('#Encoder')
        plt.title('Encoded spike trains')
        plt.show()
        return encoded


class GanglionCell:
    def __init__(self, n_rf, t_max=1., alpha=1., amp=1., freq=40., phi_0=0., delta_phi=None):
        self.n_rf = n_rf
        self.t_max = t_max
        self.alpha = alpha
        self.amp = amp
        self.freq = freq
        self.omega = 2 * np.pi * freq
        self.phi_0 = phi_0
        if delta_phi is None:
            self.delta_phi = 2 * np.pi / n_rf
        else:
            self.delta_phi = delta_phi

    def encode(self, stimulation):
        """Encode input stimulation.

        Args
        ----
        stimulation (:obj: np.ndarray): Must be of of shape (n_rf,) .

        :returns :obj: np.array of shape(n_rf,) and dtype int64
        """
        receptive_field = [PhotoReceptor(t_max=self.t_max, alpha=self.alpha) for _ in range(self.n_rf)]
        out_spike_times = np.zeros(self.n_rf, dtype=np.int64)
        # fig, ax = plt.subplots(nrows=4, figsize=(14, 18))
        for (ind, intensity) in enumerate(stimulation):
            pr_spike_time = receptive_field[ind].get_spike_time(intensity=intensity)
            phi = self.phi_0 + ind*self.delta_phi
            k = np.arange(0, int(self.freq * self.t_max) + 1)
            peak_times = np.round((2*k*np.pi - phi) / self.omega * 1000) # In miliseconds
            peak_times = peak_times[peak_times < self.t_max*1000]
            differences = np.abs(peak_times - pr_spike_time*1000)
            spike_time = peak_times[np.argmin(differences)].astype(np.int64)
            out_spike_times[ind] = spike_time
        #     ax[ind].plot(np.cos(self.omega * np.arange(0, 1, 0.001) + phi))
        #     ax[ind].axvline(spike_time, color='b')
        #
        # fig.show()
        return out_spike_times


class PhotoReceptor:
    def __init__(self, t_max, alpha):
        """PhotoReceptor unit.

        Args
        ----
        t_max (float): Max output interval is seconds.
        alpha (float): Scaling factor used in logarithmic transformation function.

        """
        assert isinstance(t_max, float)
        assert isinstance(alpha, float)
        self.t_max = t_max
        self.alpha = alpha

    def get_spike_time(self, intensity):
        """Returns spike time in seconds.

        Args
        ----
        intensity (float): Must be float and of normalized distribution.

        :returns spike time in seconds, of type float.
        """

        assert isinstance(intensity, float), "'intensity' must be of type float"
        spike_time = self.t_max - np.log(self.alpha * intensity + 1)
        return spike_time


class TemporalPhaseEncoder:
    """Phase encoder unit, for temporal input encoding.

    Proposed by https://ieeexplore.ieee.org/abstract/document/7086059 .
    Note that each encoder unit can encode a single temporal signal .

    Instantiate an object by passing parameters, then encode desired signal using 'encode' method .

    Notes from paper
    ----------------
    *The specific activity patterns considered in this paper are in a spatiotemporal
     form where the precise timing of spikes is used for carrying information.
    *Encoding unit contains a positive neuron (Pos), a negative neuron (Neg),
     and an output neuron (Eout).
    *Each encoding unit is connected to an input signal x and an SMO.
    *The potentials of the Pos and Neg neurons are the summation of x and SMO.
    *Whenever the membrane potential first crosses the threshold (θE), the
     neuron will fire a spike.
    *The neuron is allowed to fire only once within the whole oscillation period T .
     Note: In this implementation, this limitation is performed via subtraction of a
     exponentially decaying potential.
    *The firing of either the Pos neuron or the Neg neuron will immediately
     cause a spike from the Eout neuron.

    """
    def __init__(self, magnitude=1., freq=40., phi=0., threshold=1.5, resetting_freq_frac=0.1):
        """Initialize an  instance.

        Args
        ----------
        magnitude (float): magnitude of SMO function.

        freq (float): frequency of SMO function in Hz.
            Default is gamma band, 40Hz.

        phi (float): phase of this unit's SMO, in radian.
            Default value is 0. User should determine every unit's phase.

        threshold (float): threshold for Pos and Neg neurons.
            default value: 1.5

        resetting_freq_frac (float): frequency of subtraction exponential term for firing times.
            When firing a spike, unit's threshold will be subtracted from unit's potential,
             and effect of this subtraction will decay exponentially by a time constant,
             this time constant is (omega * resetting_freq_frac) where omega is 2 * pi / period
            default value: 0.1
        """
        self.magnitude = magnitude
        self.freq = freq
        self.phi = phi
        self.threshold = threshold
        self.resetting_freq_frac = resetting_freq_frac

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, new_magnitude):
        assert isinstance(new_magnitude, float), "'magnitude' must be of type float."
        self._magnitude = new_magnitude
        return

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, new_freq):
        assert isinstance(new_freq, float), "'freq' must be of type float."
        self._freq = new_freq

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        assert isinstance(new_phi, float), "'phi' must be of type float."
        self._phi = new_phi

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        assert isinstance(new_threshold, float), "'threshold' must be of type float."
        self._threshold = new_threshold

    @property
    def resetting_freq_frac(self):
        return self._resetting_freq_frac

    @resetting_freq_frac.setter
    def resetting_freq_frac(self, new_rff):
        assert isinstance(new_rff, float), "'resetting_freq_frac' must be of type float."
        self._resetting_freq_frac = new_rff


    def encode(self, sgnl):
        """Encode input signal.

        Args
        ----
        sgnl (:obj: 'np.ndarray'): input rank 1 signal, each sample point is one milisecond.
            Encoding will implemented on normalized signal which scaled on miliseconds, i.e. datapoints
            start at 0 milisecond and end in nth milisecond while time step is 1 milisecond.


        :returns a binary array with len=len(sgnl) containing 1s for spikes and 0s for not spikes.

        Note: sgnl steps must be milisecond!

        """

        assert isinstance(sgnl, np.ndarray), "'sgnl' must be of type np.ndarray ."
        assert sgnl.ndim == 1, "'sgnl' must be of rank 1 ."
        assert sgnl.dtype == np.float64, "'sgnl' must be of dtype np.float64"
        normalized_signal = (sgnl - np.mean(sgnl)) / np.std(sgnl)
        length = normalized_signal.shape[0]
        interval = np.arange(0, sgnl.shape[0])/1000
        omega = 2 * np.pi * self.freq
        SMO = self.magnitude * np.cos(omega * interval + self.phi)
        pos = SMO + normalized_signal
        neg = -SMO - normalized_signal
        spikes = np.zeros(sgnl.shape)
        resetting_potential = np.zeros(length, dtype=np.float64)
        resetting_signal = self.threshold * np.exp(-omega *
                                                   self.resetting_freq_frac *
                                                   np.arange(0, length) / 1000)
        for step in range(0, length):
            if pos[step] >= self.threshold:
                spikes[step] = 1
                resetting_potential[:] = 0
                resetting_potential[step:] = resetting_signal[: length - step]
                pos -= resetting_potential
            if neg[step] >= self.threshold:
                spikes[step] = 1
                resetting_potential[:] = 0
                resetting_potential[step:] = resetting_signal[: length - step]
                neg -= resetting_potential
        spike_times = np.where(spikes == 1)[0]
        plt.style.use('ggplot')
        fig, [ax0, ax1, ax2, ax3] = plt.subplots(nrows=4, sharex=True, figsize=(14, 10))
        fig.figsize = (18, 12)
        ax0.plot(normalized_signal)
        ax0.set_title('Original -normalized- signal')
        ax1.eventplot(spike_times)
        ax1.set_title('Encoded signal')
        ax1.set_yticks([1])
        ax2.plot(pos)
        ax2.set_title("Pos neuron's potential")
        ax3.plot(neg)
        ax3.set_title("Neg neuron's potential")
        plt.show()
        return spike_times
