# TODO: determine if batchsize can be higher for genetic_config and still find best shape so that I can train faster
# TODO: determine if epochs can be lower for genetic_config and still find best shape so that I can train faster

from library.CVtrain import *
import scipy.io as sio
import math
from math import ceil

# adds 0s to make layer arrays all the same size
def append_zeros(layers, L):
    x = len(layers)
    for i in range(L - x):
        layers.append(0)
    return layers

# removes dangling zeros from being the child of different depth NNs
def handle_zeros(layers):
    x = layers[:]
    try:
        while(1):
            x.remove(0)
    except:
        return x

# returns an array of layer arrays
def generate_initial_generation(N, L, size):
    generation = []
    for i in range(size):
        layers = []
        # choose a depth
        x = random.randint(1, L)
        for j in range(x):
            layers.append(random.randint(1, N))
        layers = append_zeros(layers, L)
        generation.append(layers)

    return generation

# model1 and model2 are same length
def reproduce(model1, model2):
    x = len(model1)

    child = []

    for i in range(x):
        child.append( ceil( (model1[i] + model2[i]) / 2))

    return (child)

def train_generation(gen, trainingSet, batchSize, verbose, epochs):
    MSEs = []
    runtimeComplexities = []
    for i in range(len(gen)):
        trainFriendlyLayer = handle_zeros(gen[i])
        [MSE, weightsAndBiases] = CVtrain(trainFriendlyLayer, trainingSet, 5, "trash", batchSize, verbose, epochs)
        MSEs.append(MSE)
        runtimeComplexities.append(weightsAndBiases)

    return [MSEs, runtimeComplexities, compute_errors(MSEs, runtimeComplexities)]

def mutate(layers, N, L, force=False):
    if force:
        layers[random.randint(0,len(layers)-1)] = random.randint(1,N)
    
    for i in range(len(layers)):
        if random.randint(0,100) < 5:
            layers[i] = random.randint(1,N)
    
    # mutate more rarely
    if (random.randint(0,100) < 33):
        return mutate(layers, N, L)

    if (random.randint(0,100) < 10):
        for i in range(len(layers)):
            layers[i] = random.randint(1,N)

    return layers

# how im using fitness scores
def select_parent(fitness, totalFitness):
    chosen = random.randint(0, int(totalFitness)) # casting to int may cut off largest value but that is okay
                                                  # because the highest value is the worst NN
    for i in range(len(fitness)):
        chosen = chosen - fitness[i]
        if chosen <= 0:
            return i

def compute_errors(MSEs, runtimeComplexities, lam = 0.001):
    errors = []
    i = 0
    while i < len(MSEs):
        errors.append(MSEs[i] * math.exp(lam * runtimeComplexities[i]))
        i = i + 1
    
    return errors

# convert errors to a fitness value
def fitness_score(errors):
    maxError = max(errors)
    fitness = []
    for e in errors:
        if e == 0:
            print("Zero Error in fitness_score")
        # square to punish low values more
        fitness.append(math.pow(maxError/e, 2)) #maxError/e makes smaller errors end up with larger fitness scores
    return fitness

def generate_next_generation(lastGen, MSEs, N, L, populationSize):
    # sort MSEs and lastGen so they are in order of best MSE to worst
    errors, structures = zip(*sorted(zip(MSEs, lastGen), key=lambda x: x[0], reverse=False))
    errors = list(errors)
    structures = list(structures)

    fitness = fitness_score(errors)
    totalFitness = sum(fitness)
    # Save top X structures for next gen
    numToSave = ceil(populationSize * .10) #save 10% of the population rounding up 
    nextGen = structures[0:numToSave]

    # loop to fill rest of the generation
    for i in range(0, populationSize - numToSave, 1):
        # get mom and dad
        momIndex = select_parent(fitness, totalFitness)
        dadIndex = select_parent(fitness, totalFitness)
        mom = structures[momIndex]
        dad = structures[dadIndex]

        # make baby
        child = reproduce(mom, dad)

        child = mutate(child, N, L)
        # avoid duplicates
        if child in nextGen:
            child = mutate(child, N, L, force=True)

        
        # Remove 0 node layers between non 0 node layers: [10,0,10] -> [10,10,0]
        formatedChild = append_zeros(handle_zeros(child), L) # TODO: look into more efficient operations 
        nextGen.append(formatedChild)

    return nextGen

def genetic_config(N, L, trainingSetFile, populationSize, batchSize=250, verbose=0, epochs=500):
    # create first list of models
    generations = 20
    currentGen = generate_initial_generation(N, L, populationSize)

    trainingSet = sio.loadmat(trainingSetFile)

    minError = None
    bestStructure = None
    stop = 0
    first = True
    #train a generation
    for i in range(generations):
        t0 = time.time()
        [MSEs, runtimeComplexities, errors] = train_generation(currentGen, trainingSet, batchSize, verbose, epochs)
        t1 = time.time()
        totalMSE = sum(MSEs)
        totalError = sum(errors)
        avgMSE = totalMSE/len(MSEs)
        avgErrors = totalError/len(errors)

        print("Gen {0} Train time: {1} minutes:".format(i, (t1-t0)/60))
        print(" Average MSE:        {0}".format(avgMSE))
        print(" Average Error:      {0}".format(avgErrors))
        print(" {0:20}|{1:20}|{2:20}|{3:10}".format("Structure", "Error", "MSE", "weights&biases"))

        for j in range(len(MSEs)):
            print(" {0:20}|{1:20}|{2:20}|{3:10}".format(str(currentGen[j]), str(errors[j]), str(MSEs[j]), str(runtimeComplexities[j])))

        #stop early conditions
        genMinError = min(errors)
        if minError == None:
            minError = genMinError
            bestStructure = currentGen[errors.index(genMinError)]

        if (bestStructure == currentGen[errors.index(genMinError)]): #generation has same best structure
            stop = stop + 1            
        else: #new best found. reset stop condition
            stop = 0

        if not first and (minError < genMinError): #generation didnt improve enough
            if(epochs < 800):
                print("Increasing Epochs")
                epochs = epochs + 100 #increase training strength
            else:
                stop = 4

        # update stop condition variables
        first = False
        if genMinError < minError:
            minError = genMinError
        bestStructure = handle_zeros(currentGen[errors.index(genMinError)])

        if stop == 4: #if same structure wins for 4 generations in a row, stop
            return bestStructure
 
        currentGen = generate_next_generation(currentGen, errors, N, L, populationSize)
    
    return bestStructure
