from phase_encoders import TemporalPhaseEncoder
import numpy as np


sample_points = np.linspace(0, 1, 1000)
sig = np.cos(2 * np.pi * 50 * sample_points) + np.cos(2 * np.pi * 70 * sample_points) + np.random.random(1000)/2
encoder = TemporalPhaseEncoder()
encoder.encode(sgnl=sig)
