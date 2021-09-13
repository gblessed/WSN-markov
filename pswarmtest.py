'''
I create a PSO with 2 variables (time step and markov interval)
with the cbjective to minimize... <<energy + error>>

'''
# Import modules
import numpy as np

# Import PySwarms
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

def f(x):
    """Higher-level method to run an instance of the network with certain parameters
    """
    n_particles = x.shape[0]
    j = [forward_prop(x[i]) for i in range(n_particles)]
    return np.array(j)