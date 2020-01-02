import random
import Model
import copy

hidden_layer_range = [i for i in range(100)]

# [(0,2),(0,3),(0,4),(0,5),(0,6),(1,2),(1,3),(1,4),(1,5),(1,6),(2,3),(2,4),(2,5),(2,6),
#   (3,4),(3,5),(3,6),(4,5),(4,6),(5,6)]
N = 20
hidden_layer_num = 5
max_op = 7


def random_architecture():
    """Returns a random architecture (bit-string) represented as an int."""
    return random.randint(0, 2 ** Model.DIM - 1)


def mutate_arch(parent_arch):
    """Computes the architecture for a child of the given parent architecture.

    The parent architecture is cloned and mutated to produce the child
    architecture. The child architecture is mutated by flipping a randomly chosen
    bit in its bit-string.

    Args:
      parent_arch: an int representing the architecture (bit-string) of the
          parent.

    Returns:
      An int representing the architecture (bit-string) of the child.
    """
    position = random.randint(0, Model.DIM - 1)  # Index of the bit to flip.

    # Flip the bit at position `position` in `child_arch`.
    child_arch = parent_arch ^ (1 << position)

    return child_arch


def cal_output_layer(arch):
    output_layer = []

    for i in range(0, hidden_layer_num + 2):
        output_layer.append((i, 1))
    for i in range(2, hidden_layer_num + 2):
        current_layer = arch[i]
        for c in current_layer:
            if c[1] != 0:
                output_layer[c[0]] = (c[0], 0)

    arch[hidden_layer_num + 2] = output_layer



def random_NAS_architecture():
    """

    :return: arch is a vector []
    """
    arch = {}
    for i in range(2, hidden_layer_num + 2):
        count = 0
        arch[i] = []
        for j in range(i):
            time=[1,2][j<2]
            for _ in range(time):
                if random.random() < 0.35:
                    rand_op = random.randint(1, max_op)
                else:
                    rand_op = 0
                count += rand_op
                arch[i].append((j, rand_op))
        if count == 0:
            sample = random.randint(0, i + 1)
            arch[i][sample] = (arch[i][sample][0], random.randint(1, max_op))

    cal_output_layer(arch)

    return arch

def choose_op():
    op=-1

    if random.random()<0.3:
        op=1
    else:
        if random.random()<0.6:
            op = random.randint(4, 7)
        else:
            op = random.randint(2, 3)

    return op

def NAS_mutate_arch(arch):
    arch = copy.deepcopy(arch)

    while True:
        mutate_layer = random.randint(2, hidden_layer_num + 1)
        mutate_position = random.randint(0, mutate_layer + 1)

        mutate_cell=arch[mutate_layer][mutate_position]

        if random.random()<0.7:
            if mutate_cell[1]==0:
                op_mutated=choose_op()
            else:
                op_mutated=0
        else:
            if mutate_cell[1]==0:
                continue
            else:
                op_mutated=choose_op()

        count = 0
        for i in range(mutate_layer):
            if arch[mutate_layer][i][1] > 0:
                count += 1

        # op_mutated = (arch[mutate_layer][mutate_position][1] + random.randint(1, max_op)) % (max_op + 1)
        if (count == 1 and mutate_cell[1] > 0 and op_mutated == 0) or (op_mutated==mutate_cell[1]):
            # op_mutated = random.randint(1, max_op)
            continue
        else:
            break


    arch[mutate_layer][mutate_position] = (mutate_cell[0], op_mutated)
    cal_output_layer(arch)
    print(arch)
    return arch

def NAS_crossover(parent_A, parent_B):
    """
    交叉算子
    选取种群中acc最高的两个个体A，B
    1.直接继承acc较高的个体A的结构，随机将B中不同的部分加入结构
    2.继承A，B两个个体结构中相同的部分，两个个体结构不同的部分随机继承（需要保证结构合法）
    """
    child = []
    if random.random()>0.5:
        child = copy.deepcopy(parent_A)
        for i in range(2, hidden_layer_num + 2):
            current_layer=parent_A[i]
            for j in range(len(current_layer)):
                if current_layer[j][1] ==0:
                    if random.random()<0.5:
                        child[i][j]=parent_B[i][j]
    else:
        child={}
        for i in range(2, hidden_layer_num + 2):
            count=0

            while count==0:
                child[i]=[]
                current_layer = parent_A[i]
                for j in range(len(current_layer)):
                    if random.random()>0.5:
                        child[i].append(parent_A[i][j])
                    else:
                        child[i].append(parent_B[i][j])
                for j in range(len(current_layer)):
                    if child[i][j][1]!=0:
                        count+=1
    cal_output_layer(child)
    print('A:'+str(parent_A))
    print('B:'+str(parent_B))
    print('C:'+str(child))
    return child





def NAS_mutate_arch_old(arch):
    mutate_position = random.randint(0, len(arch) - 1)
    arch = copy.deepcopy(arch)
    tmp = arch[mutate_position]
    if random.random() < 0.2:
        mutate_range = hidden_layer_range[:mutate_position + 2]
        mutate_range.remove(tmp[0])
        mutate_range.remove(tmp[1])
        if mutate_range:
            tmp[random.randint(0, 1)] = random.sample(mutate_range, 1)[0]
        else:
            tmp[random.randint(2, 3)] = random.randint(1, 13)

    else:
        tmp[random.randint(2, 3)] = random.randint(1, 13)
    return arch


def random_NAS_architecture_old():
    arch = []
    for i in range(5):
        arch.append(random.sample(hidden_layer_range[:i + 2], 2))
        arch[i].append(random.randint(1, 13))
        arch[i].append(random.randint(1, 13))
    return arch
