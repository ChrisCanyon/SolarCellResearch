from library.config.exhaustive import *
from library.config.genetic import *
from library.mio import *

print("Exhausting")
exhuastive_config(2, 3, "dataset10k.mat", "dataset1k.mat", batchSize=1000, verbose=0, epochs=100)

print("Geneticing")
genetic_config(2, 3, "dataset10k.mat", "dataset1k.mat", batchSize=1000, verbose=0, epochs=100)
