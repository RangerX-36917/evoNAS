import random
import Model
import copy

hidden_layer_range = [i for i in range(100)]


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
    arch = []
    for i in range(5):
        arch.append(random.sample(hidden_layer_range[:i + 2], 2))
        arch[i].append(random.randint(1, 13))
        arch[i].append(random.randint(1, 13))
    return arch


def NAS_mutate_arch(arch):
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
