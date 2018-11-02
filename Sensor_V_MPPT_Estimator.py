from library.config.exhaustive import *
from library.config.genetic import *
from library.mio import *
import scipy.io as sio

#genetic_config(100, 5, "dataset10k.mat", "dataset1k.mat")

trainingSet = sio.loadmat("dataset100k.mat")
evaluateSet = sio.loadmat("dataset10k.mat")

CVtrain([98,9], trainingSet ,evaluateSet, folds=10, name="Genetic2", batch=10, verbose=0, epochs=1000)
print(getMSE('Genetic2'))