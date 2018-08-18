import numpy as np
import matplotlib.pyplot as plt


class PhaseEncoderUnit:
    """Phase encoder unit, for temporal input encoding.

    Proposed by https://ieeexplore.ieee.org/abstract/document/7086059 .
    Note that each encoder unit can encode a single temporal signal .

    First instantiate an object by passing parameters, then encode desired signal using 'encode' method .
    """
    def __init__(self, threshold=1.5, phi=0, period=0.001, magnitude=1, resetting_freq_frac=0.1):
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
            default value : 0.001 for generating a 1000Hz signal in SMO.

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
