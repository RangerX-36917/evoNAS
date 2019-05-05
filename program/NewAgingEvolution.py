import collections
import random
import Model
import opration


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
    for i in range(population_size):
        model = Model.Model()
        model.arch = opration.random_architecture()
        model.accuracy = Model.train_and_eval(model.arch)
        model.age = population_size - i
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

def NAS_evolution(cycles, population_size, sample_size):

    # population = collections.deque()
    population = []
    history = []  # Not used by the algorithm, only used to report results.

    # Initialize the population with random models.
    # while len(population) < population_size:
    for i in range(population_size):
        model = Model.NASModel()
        model.arch = opration.random_NAS_architecture()
        model.accuracy = model.train_NAS()
        model.age = population_size - i
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
        child = Model.NASModel()
        child.arch = opration.NAS_mutate_arch(parent.arch)
        child.accuracy = child.train_NAS()
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
