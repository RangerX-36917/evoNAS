import random
import Model

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