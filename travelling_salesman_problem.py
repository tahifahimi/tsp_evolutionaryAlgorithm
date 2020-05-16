""" solving travelling salesman problem  with genetic algorithm
    Tahere Fahimi 9539045
    main problem start at : genetic_algorithm function
        step 1) initiate first population randomly
        step 2) create next generation with :
            . rank Routes --> calculate fitness for the population and sort them
            .. selection --> select the parents from their fitness
            ... crossover the parents
            .... mutation over the chromosomes
        step 3) Go to step 2 until all number of generation is created
        step 4) find the best route and return it
"""
import numpy as np, random, operator, pandas as pd, matplotlib.pyplot as plt

"""city : save the coordinate of each city """
class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # calculate distance of this city to another city
    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance


""" Fitness : save the route, distance and fitness for each chromosome"""
class Fitness:
    def __init__(self, route):
        self.route = route  # route = array of cities
        self.distance = 0
        self.fitness = 0.0

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(0, len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                pathDistance += fromCity.distance(toCity)
            self.distance = pathDistance
        return self.distance

    """fitness is 1/distance --> we are going to maximize the fitness, so we have to minimize the distance"""
    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness


"""generate each chromosome and return it"""
def create_chromosome(cityList):
    # choose list of random items from cityList
    chro = random.sample(cityList, len(cityList))
    return chro


"""generate the first population"""
def initialPopulation(popSize, cityList):
    population = []
    for i in range(0, popSize):
        population.append(create_chromosome(cityList))
    # print("len of first : ", len(population))
    return population


"""calculate fitness for the population,
add ID to each route and then sort them """
def calculate_fitness(population):
    fitnessResults = {}
    for i in range(0, len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key=operator.itemgetter(1), reverse=True)


"""select new population as parents : the selection function returns a list of route IDs, 
which we can use to create the mating pool in the matingPool function"""
def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns=["Index", "Fitness"])
    df['cum_sum'] = df.Fitness.cumsum()
    df['cum_perc'] = 100 * df.cum_sum / df.Fitness.sum()    # calculate the proability for each chromosome

    for i in range(0, eliteSize):
        selectionResults.append(popRanked[i][0])
    # use rollet while to choose other parents
    for i in range(0, len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(0, len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


"""select all parents and add them in a pool"""
def matingPool(population, selectionResults):
    matingpool = []
    for i in range(0, len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


"""crossover on the parents and create one child"""
def crossOver(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent1))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child


"""crossover on all parents in mating pool"""
def crossOverPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(0, eliteSize):
        children.append(matingpool[i])

    for i in range(0, length):
        child = crossOver(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)
    return children


"""mutate a single route"""
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city2
            individual[swapWith] = city1
    return individual


"""mutate over all population"""
def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop


"""create children by selection, crossover, mutation"""
def nextGeneration(current_population, eliteSize, mutationRate):
    # calculate fitness for the population and sort them
    popRanked = calculate_fitness(current_population)
    # select best parents
    selected_population = selection(popRanked, eliteSize)
    # parents are children and ????
    matingpool = matingPool(current_population, selected_population)
    children = crossOverPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration


def geneticAlgorithmPlot(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("Initial distance: " + str(1 / calculate_fitness(pop)[0][1]))

    progress = []
    progress.append(1 / calculate_fitness(pop)[0][1])

    acceptable_distance = 40000
    for i in range(0, generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)
        final_fitness = 1 / calculate_fitness(pop)[0][1]
        progress.append(1 / calculate_fitness(pop)[0][1])
        if final_fitness < acceptable_distance:
            print("closed in ", i)
            print(final_fitness)
            break

    print("Final distance: " + str(1 / calculate_fitness(pop)[0][1]))
    bestRouteIndex = calculate_fitness(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    print(bestRoute)
    plt.plot(progress)
    plt.ylabel('Distance')
    plt.xlabel('Generation')
    plt.savefig('tsp.png')
    plt.show()

if __name__ == "__main__":
    cityList = []
    """read file and assign to city list"""
    with open("tsp_data.txt", "r+") as file:
        for line in file:
            d = line.split(" ")
            cityList.append(City(x=float(d[0]), y=float(d[1])))
            
    # solving the problem with genetic algorithm and draw the answer
    geneticAlgorithmPlot(population=cityList, popSize=100, eliteSize=20, mutationRate=0.01, generations=120)
