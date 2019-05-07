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


def random_NAS_architecture():
    """

    :return: arch is a vector []
    """
    arch = {}
    for i in range(2, hidden_layer_num + 2):
        count = 0
        arch[i] = []
        for j in range(i):
            rand_op = random.randint(0, max_op)
            count += rand_op
            arch[i].append((j, rand_op))
        if count == 0:
            sample = random.randint(0, i - 1)
            arch[i][sample] = (sample, random.randint(1, max_op))
    return arch


def NAS_mutate_arch(arch):
    arch = copy.deepcopy(arch)

    mutate_layer = random.randint(2, hidden_layer_num + 1)
    mutate_position = random.randint(0, mutate_layer - 1)

    count = 0
    for i in range(mutate_layer):
        if arch[mutate_layer][i][1] > 0:
            count += 1
    op_mutated = (arch[mutate_layer][mutate_position][1] + random.randint(1, max_op)) % (max_op + 1)
    if count == 1 and arch[mutate_layer][mutate_position][1] > 0 and op_mutated == 0:
        op_mutated = random.randint(1, max_op)
    arch[mutate_layer][mutate_position] = (mutate_position, op_mutated)
    return arch


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
