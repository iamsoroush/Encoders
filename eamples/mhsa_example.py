import numpy as np
import modified_hsa

s = np.sin(np.linspace(0, 10, 1000))
noise = np.random.rand(1000)/5
s += noise
encoder = modified_hsa.ModifiedHSA()
encoder.encode(sgnl=s)
encoder.plot()
