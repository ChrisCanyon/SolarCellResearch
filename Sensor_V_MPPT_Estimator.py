from library.config.exhaustive import *
from library.config.genetic import *
from library.mio import *
import scipy.io as sio


N = 6

TrainFile = "datasets/N" + str(N) + "dataset10k.mat"
EvalFile = "datasets/N" + str(N) + "dataset100k.mat"
ModeName = "N"  + str(N) + "geneticResult"

layers = genetic_config(20, 5, TrainFile, 25)
print(layers)

evalSet = sio.loadmat(EvalFile)
CVtrain([20,2,5], evalSet, 10, ModeName, batch=10, verbose=1, epochs=1000)
