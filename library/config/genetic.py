# TODO: determine if batchsize can be higher for network config and still find best shape so that I can train faster

from library.mio import *
import scipy.io as sio

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

    flipIndex = random.randint(0, x-1)

    model3 = model2[:]

    for i in range(flipIndex):
        model3[i] = model1[i]
    
    return model3

def train_generation(gen, trainingSet, evaluateSet, batchSize, verbose, epochs):
    MSEs = []

    for i in range(len(gen)):
        trainFriendlyLayer = handle_zeros(gen[i])
        CVtrain(trainFriendlyLayer, trainingSet, evaluateSet, 5, "trash", batchSize, verbose, epochs)
        MSE = getMSE('trash')
        MSEs.append(MSE)
    
    print("MSEs after training:", MSEs)
    return MSEs

# TODO: look into mutate options
# Mutate Options:
# Add/subtract layer
# Update more than 1 node
# Call itself recursively rarely for major mutations
# currently just chooses a random index and changes the nodes to a random number
def mutate(layers, N, L):
    x = len(layers)

    i = random.randint(0,x-1)

    lower = 1 - layers[i] 
    upper = N - layers[i]

    layers[i] += random.randint(lower, upper)

    # mutate more rarely
    if (random.randint(0,100) < 33):
        return mutate(layers, N, L)

    return layers

def select_parent(fitness):
    total = sum(fitness)

    chosen = random.randint(0, int(total)) # casting to int may cut off largest value but that is okay
                                           # because the highest value is the worst NN
    for i in range(len(fitness)):
        chosen = chosen - fitness[i]
        if chosen <= 0:
            return i

def generate_next_generation(lastGen, MSEs, N, L):
    # sort MSEs and lastGen so they are in order of best MSE to worst
    errors, structures = zip(*sorted(zip(MSEs, lastGen), key=lambda x: x[0], reverse=False))
    errors = list(errors)
    structures = list(structures)

    # convert errors to some fitness value
    maxError = max(errors)
    fitness = []
    for e in errors: #make sure i dont divide by zero
        # possibly square to punish low values more
        fitness.append(math.pow(maxError/e, 2)) #dividing maxError/e makes smaller errors end up with larger fitness scores

    # Save top X structures for next gen
    # TODO: make this based on population size
    nextGen = structures[0:2]

    # loop to fill rest of the generation
    x = len(MSEs)
    for i in range(x - 2):
        # get mom and dad
        momIndex = select_parent(fitness)
        dadIndex = select_parent(fitness)

        mom = structures[momIndex]
        dad = structures[dadIndex]
        # make baby
        child = reproduce(mom, dad)
        # roll some chance to mutate
        if (random.randint(0,100) < 25):
            child = mutate(child, N, L)
        
        # avoid duplicates
        if child in nextGen:
            child = mutate(child, N, L)
        
        nextGen.append(child)

    return nextGen

def genetic_config(N, L, trainingSetFile, evaluateSetFile, batchSize=250, verbose=0, epochs=500):
    # create first list of models
    generations = 20
    genepoolSize = 25
    currentGen = generate_initial_generation(N, L, genepoolSize)
    print("Gen 0:", currentGen)

    trainingSet = sio.loadmat(trainingSetFile)
    evaluateSet = sio.loadmat(evaluateSetFile)

    #train a generation
    for i in range(generations):
        t0 = time.time()
        MSEs = train_generation(currentGen, trainingSet, evaluateSet, batchSize, verbose, epochs)
        t1 = time.time()
        totalMSE = 0
        for j in range(len(MSEs)):
            totalMSE += MSEs[j]
        print("Gen {0} Train time: {1} minutes, Average MSE: {2}".format(i, (t1-t0)/60, totalMSE/j))
        currentGen = generate_next_generation(currentGen, MSEs, N, L)
        print("Next Gen:", currentGen)
