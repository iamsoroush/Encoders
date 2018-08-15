import numpy as np
import hsa

s = np.sin(np.linspace(0, 10, 1000))
noise = np.random.rand(1000)/5
s += noise
s = (1 + s)/2
encoder = hsa.HSAEncoder()
encoder.encode(sgnl=s)
encoder.plot()
encoder.decode()
