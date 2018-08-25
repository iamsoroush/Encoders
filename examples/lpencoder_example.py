import numpy as np
from phase_encoders import LPEncoder

encoder = LPEncoder()
encoder.encode(input_array=np.random.random(12))