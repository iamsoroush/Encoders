import numpy as np
import bsa

s = np.sin(np.linspace(0, 10, 1000))
noise = np.random.rand(1000)/5
s += noise
encoder = bsa.BSAEncoder()
encoder.encode(sgnl=s)
encoder.plot()
