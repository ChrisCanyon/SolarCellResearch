from library.config.exhaustive import *
from library.config.genetic import *
from library.mio import *
import scipy.io as sio

print(genetic_config(20, 5, "datasets/N3dataset10k.mat", 25))

