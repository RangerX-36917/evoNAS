import copy
import random
import numpy as np
import pickle


class Individual:
    DNA = None
    fitness = 0

    def __lt__(self, other):
        return self.fitness < other.fitness


STRING_LENGTH = np.power(2, 9)
POP_SIZE = 1000
N_GENERATIONS = 100000


class BitString(Individual):
    def __init__(self, length, life_time, age):
        self.DNA = [[0, 1][random.random() > 0.5] for i in range(length)]
        self.age = age
        self.life_time = life_time

    def cal_fitness(self):
        self.fitness = 0
        noise = np.random.normal(loc=0, scale=0.01)
        # noise = 0
        for j in self.DNA:
            self.fitness += (j + noise)
        # self.fitness +

    def cal_acc_fitness(self):
        self.fitness = 0
        for j in self.DNA:
            self.fitness += j


class GA:
    def __init__(self, pop_size, string_length=STRING_LENGTH):
        self.string_length = string_length
        self.pop_size = pop_size

        # initial DNAs for winner and loser
        self.pop = [BitString(string_length, life_time=pop_size, age=pop_size - i) for i in range(pop_size)]
        for p in self.pop:
            p.cal_fitness()

    def select(self):
        idx = np.random.choice(np.arange(len(self.pop)), size=10, replace=True)
        # p_set = []
        # max_idx = 0
        parent = self.pop[0]
        for i in idx:
            if self.pop[i].fitness > parent.fitness:
                parent = self.pop[i]
                # max_idx = i
            # p_set.append(self.pop[i])
        return copy.deepcopy(parent), parent

    def mutate(self, parent):
        m_idx = random.randint(0, self.string_length - 1)
        parent.DNA[m_idx] = 1 - parent.DNA[m_idx]
        parent.cal_fitness()
        return parent

    def NAE_evolve(self):
        parent = self.select()[0]

        self.mutate(parent)

        self.pop.append(parent)
        self.pop.sort()
        self.pop.pop(0)

    def AE_evolve(self):
        parent = self.select()[0]
        self.mutate(parent)

        self.pop.append(parent)
        # self.pop.sort()
        self.pop.pop(0)

    def new_AE_evolve(self):
        child, parent = self.select()
        child.age = 1
        child.life_time = self.pop_size
        self.mutate(child)

        self.pop.sort()
        if child.fitness > self.pop[int(len(self.pop) * 0.90)].fitness:
            parent.life_time += 50

        for p in self.pop:
            p.age += 1
            if p.age > p.life_time:
                self.pop.remove(p)

        self.pop.append(child)


if __name__ == '__main__':

    re = {}
    data = {}
    for _ in range(12, 19):
        STRING_LENGTH = int(np.power(2, _ / 2))

        re[_ / 2] = {}
        time = 50
        for j in range(time):
            re[_ / 2][j] = {}
            ga = GA(POP_SIZE, STRING_LENGTH)
            for i in range(N_GENERATIONS):
                ga.NAE_evolve()
            best = max(ga.pop)
            best.cal_acc_fitness()
            print("non age:" + str(best.fitness))
            re[_ / 2][j]["non age:"] = best.fitness

            ga = GA(POP_SIZE, STRING_LENGTH)
            for i in range(N_GENERATIONS):
                ga.AE_evolve()
            best = max(ga.pop)
            best.cal_acc_fitness()
            print("age:" + str(best.fitness))
            re[_ / 2][j]["age:"] = best.fitness

            # for i in range(5):
            ga = GA(POP_SIZE, STRING_LENGTH)
            for i in range(N_GENERATIONS):
                ga.new_AE_evolve()
            best = max(ga.pop)
            best.cal_acc_fitness()
            print("new age:" + str(best.fitness))
            re[_ / 2][j]["new age:"] = best.fitness

        a = re[_ / 2]
        non_age = 0
        age = 0
        new_age = 0
        for i in a:
            non_age += a[i]['non age:']
            age += a[i]['age:']
            new_age += a[i]['new age:']
        non_age /= time
        age /= time
        new_age /= time
        data[_ / 2] = {}
        data[_ / 2]['non_age'] = non_age
        data[_ / 2]['age'] = age
        data[_ / 2]['new_age'] = new_age

    pickle.dump(re, open('re2.dt', 'wb'))
    pickle.dump(data, open('data2.dt', 'wb'))
