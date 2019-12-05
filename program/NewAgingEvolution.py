import collections
import copy
import random
import Model
import opration
import pickle


def regularized_evolution(cycles, population_size, sample_size):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    Args:
      cycles: the number of cycles the algorithm should run for.
      population_size: the number of individuals to keep in the population.
      sample_size: the number of individuals that should participate in each
          tournament.

    Returns:
      history: a list of `Model` instances, representing all the models computed
          during the evolution experiment.
    """
    # population = collections.deque()
    population = []
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    # while len(population) < population_size:
    # for i in range(population_size):
    while len(population)<population_size:
        model = Model.Model()
        model.arch = opration.random_architecture()
        model.accuracy = Model.train_and_eval(model.arch)
        model.age = population_size - len(population)
        model.life = population_size
        population.append(model)
        history.append(model)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model.Model()
        child.arch = opration.mutate_arch(parent.arch)
        child.accuracy = Model.train_and_eval(child.arch)
        child.life = population_size
        population.append(child)
        history.append(child)

        # Remove the died model.
        population.sort(key=lambda i: i.accuracy)

        if child.accuracy > population[int(0.8 * len(population))].accuracy:
            parent.life += int(population_size * 0.01)

        for p in population:
            p.age += 1
            if p.age >= p.life:
                population.remove(p)

    return history


def NAS_evolution(pop,cycles, population_size, sample_size,dir,history):
    # population = collections.deque()
    crossover_rate=0.15
    extend_lift_rate=0.1
    print('cross rate: '+str(crossover_rate))
    print('new age: '+str(extend_lift_rate))
    population = pop
    history = history
    # for p in pop:
        # history.append(p)

    #history = copy.deepcopy(pop)  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.

    # for i in range(population_size):
    # # while len(population)<population_size:
    #     model = Model.NASModel()
    #     model.normal_arch = opration.random_NAS_architecture()
    #     model.reduction_arch = opration.random_NAS_architecture()
    #     model.accuracy = model.train_NAS()
    #     model.age = population_size - i
    #     model.life = population_size
    #     population.append(model)
    #     history.append(model)

    #     with open(dir+'gen '+len(history),"wb") as f:
    #         pickle.dump(history,f)

        # for i in range(population_size):
    while len(population)<population_size:
        model = Model.NASModel()
        # if len(population)>0:
            # model.normal_arch=opration.NAS_mutate_arch(population[0].normal_arch)
            # model.reduction_arch=opration.NAS_mutate_arch(population[0].reduction_arch)
        # else:
        model.normal_arch = opration.random_NAS_architecture()
            # model.reduction_arch = opration.random_NAS_architecture()
        # model.accuracy = random.random()
        model.accuracy = model.train_NAS()
        model.age = population_size - len(population)
        model.life = population_size
        population.append(model)
        history.append(model)

        with open(dir+'gen_'+str(len(history)),"wb") as f:
            pickle.dump(history,f)

    # Carry out evolution in cycles. Each cycle produces a model and removes
    # another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            # Inefficient, but written this way for clarity. In the case of neural
            # nets, the efficiency of this line is irrelevant because training neural
            # nets is the rate-determining step.
            candidate = random.choice(list(population))
            sample.append(candidate)

        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.accuracy)

        # Create the child model and store it.
        child = Model.NASModel()

        legal_crossover=False
        if random.random()<crossover_rate:
            sorted_pop=sorted(population,key=lambda x:x.accuracy,reverse=True)
            parent_A=sorted_pop[0].normal_arch
            parent_B=sorted_pop[1].normal_arch

            child.normal_arch=opration.NAS_crossover(parent_A,parent_B)
            if child.normal_arch!=parent_A and child.normal_arch!=parent_B:
                legal_crossover=True
                print('successfully crossover')

        if  not legal_crossover:
            if random.random() < 0:
                child.normal_arch = parent.normal_arch
                child.reduction_arch = opration.NAS_mutate_arch(parent.reduction_arch)
            else:
                child.reduction_arch = parent.reduction_arch
                child.normal_arch = opration.NAS_mutate_arch(parent.normal_arch)
        
        # child.accuracy = random.random()
        child.accuracy = child.train_NAS()
        child.life = population_size
        population.append(child)
        history.append(child)

        # Remove the died model.
        population.sort(key=lambda i: i.accuracy)
        print("---> best:", population[len(population)-1].normal_arch, "\n---> with acc: ",population[len(population)-1].accuracy)
        if child.accuracy > population[int(0.8 * len(population))].accuracy:
            parent.life += int(population_size * extend_lift_rate)

        pop_next=[]
        for p in population:
            p.age += 1
            # if p.age >= p.life:
            #     population.remove(p)
            if p.age < p.life:
                pop_next.append(p)

        population=pop_next

        with open(dir+'gen_'+str(len(history)),"wb") as f:
            pickle.dump(history,f)

        # with open(dir+'gen_'+str(len(history))+'_pop',"wb") as f:
            # pickle.dump(population,f)

    return history
