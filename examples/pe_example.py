from phase_encoders import PhaseEncoderUnit
import numpy as np


s = np.sin(2 * np.pi * 50 * np.linspace(0, 1, 1000))
noise = np.random.rand(1000)/5
s += noise
encoder = PhaseEncoderUnit()
encoder.encode(sgnl=s)
