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

    Notes
    -----
    Initialize the encoder, and use 'encode' method, encoder will output n_input/n_rf spike trains
     with spike times in miliseconds.

    """

    def __init__(self, n_rf=4, t_max=1000, alpha=1, amp=1, freq=40, phi_0=0, delta_phi=None):
        """Initialize an encoder object.

        Args
        ----
        n_rf : float.
            Number of photoreceptors in each ganglion cell.

        t_max : int.
            Length of encoding window used for creating temporal data from static input, in mili-seconds.

        alpha : float.
            Scaling factor used in logarithmic transformation function.

        amp : float.
            Amplitude of SMO function.

        freq : float.
            Sub-threshold membrane oscillation's frequency in Hz. Default is gamma frequency, 40 Hz.

        phi_0 : float.
            Initial phase for calculating SNO phases.

        delta_phi: float.
            SMO phase difference between two adjacent photoreceptors, must be lower than 2*pi/n_rf .
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

    def encode(self, input_array):
        """Encodes input array to output spike trains.

        Args
        ----
        input_array: :obj: np.ndarray of shape(n_input,)

        Notes
        -----
        Encoded input will be in shape (n_output_cells, t_max) which has binary values
         for spike time occurrence.
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
        plt.eventplot(encoded)
        plt.yticks([i for i in range(0, len(encoded))])
        plt.show()
        return encoded


class GanglionCell:
    def __init__(self, n_rf, t_max=1000, alpha=1, amp=1, freq=40, phi_0=0, delta_phi=None):
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

        Arg 'stimulation' must be of type np.ndarray and of shape (n_rf,) .

        :returns :obj: np.array of shape(n_rf,) and dtype int64
        """
        receptive_field = [PhotoReceptor(t_max=self.t_max, alpha=self.alpha) for _ in range(self.n_rf)]
        out_spike_times = np.zeros((self.n_rf), dtype=np.int64)
        # fig, ax = plt.subplots(nrows=4, figsize=(14, 18))
        for (ind, intensity) in enumerate(stimulation):
            pr_spike_time = receptive_field[ind].get_spike_time(intensity=intensity)
            phi = self.phi_0 + ind*self.delta_phi
            k = np.arange(0, int(self.freq * self.t_max / 1000) + 1)
            peak_times = np.round((2*k*np.pi - phi) / self.omega * 1000).astype(np.int)
            peak_times = peak_times[peak_times < 1000]
            differences = np.abs(peak_times - pr_spike_time)
            spike_time = peak_times[np.argmin(differences)]
            out_spike_times[ind] = spike_time
        #     ax[ind].plot(np.cos(self.omega * np.arange(0, 1, 0.001) + phi))
        #     ax[ind].axvline(spike_time, color='b')
        #
        # fig.show()
        return out_spike_times


class PhotoReceptor:
    def __init__(self, t_max, alpha):
        self.t_max = t_max
        self.alpha = alpha

    def get_spike_time(self, intensity):
        """Returns spike time in mili seconds.

        intensity must be float and of normalized distribution.
        """
        assert isinstance(intensity, (np.float64)),\
            "'intensity' must be of type np.float64 or int, not {}".format(type(intensity))
        spike_time = self.t_max - 1000 * np.log(self.alpha * intensity + 1)
        return np.int64(spike_time)


class DynamicLPEncoder:
    """Phase encoder unit, for temporal input encoding.

    Proposed by https://ieeexplore.ieee.org/abstract/document/7086059 .
    Note that each encoder unit can encode a single temporal signal .

    First instantiate an object by passing parameters, then encode desired signal using 'encode' method .
    """
    def __init__(self, threshold=1.5, phi=0, period=0.03, magnitude=1, resetting_freq_frac=0.1):
        """Initialize an PhaseEncoderUnit instance.

        Args
        ----------
        threshold : float. Optional.
            default value: 1.5

        phi : float. Optional.
            Phase of SMO function: SMO = magnitude * cos(omega*t + phi)
            default value : 0.

        period : float. Optional.
            omega = 2*pi/period .
            default value : 0.03 for generating a 33Hz (gamma) signal in SMO.

        magnitude : float. Optional.
            Magnitude of SMO function.
            default value : 1 , by threshold=1.5, and normalized signal.

        resetting_freq_frac : float. Optional.
            When firing a spike, unit's threshold will be subtracted from unit's potential,
             and effect of this subtraction will decay exponentially by a time constant,
             this time constant is (omega * resetting_freq_frac) where omega is 2 * pi / period
            default value: 0.1
        """
        self.threshold = threshold
        self.phi = phi
        self.period = period
        self.magnitude = magnitude
        self.resetting_freq_frac = resetting_freq_frac

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, new_threshold):
        self._threshold = new_threshold
        return

    @property
    def phi(self):
        return self._phi

    @phi.setter
    def phi(self, new_phi):
        self._phi = new_phi
        return

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, new_period):
        self._period = new_period
        return

    @property
    def magnitude(self):
        return self._magnitude

    @magnitude.setter
    def magnitude(self, new_magnitude):
        self._magnitude = new_magnitude
        return


    def encode(self, sgnl):
        """Encode input signal.

        Args
        ----
        sgnl : :obj: 'np.ndarray', 1d array.
            Encoding will implemented on normalized signal, that is scaled on miliseconds, i.e. datapoints
            start at 0 milisecond and end in nth milisecond while time step is 1 milisecond.

        Return
        ------
        :returns a binary array with len=len(sgnl) containing 1s for spikes and 0s for not spikes.

        Note: sgnl steps must be Milisecond!
        """

        normalized_signal = (sgnl - np.mean(sgnl)) / np.std(sgnl)
        length = normalized_signal.shape[0]
        interval = np.arange(0, sgnl.shape[0])/1000
        omega = 2 * np.pi / self.period
        SMO = self.magnitude * np.cos(omega * interval + self.phi)
        pos = SMO + normalized_signal
        neg = -SMO -normalized_signal
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
        self._last_signal = normalized_signal.copy()
        self._last_spike_times = np.where(spikes == 1)[0]
        fig, [ax0, ax1, ax2, ax3] = plt.subplots(nrows=4, sharex=True)
        fig.figsize = (18, 12)
        ax0.plot(self._last_signal)
        ax0.set_title('Original signal')
        ax1.eventplot(self._last_spike_times)
        ax1.set_title('Encoded signal')
        ax1.set_yticks([1])
        ax2.plot(pos)
        ax2.set_title("Pos neuron's potential")
        ax3.plot(neg)
        ax3.set_title("Neg neuron's potential")
        plt.show()
        return spikes


encoder = LPEncoder(n_input=12)
print(encoder.encode(np.random.random(12)))
