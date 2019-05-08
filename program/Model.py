import copy
import random
import cnn.train_cnn as train_cnn
import cnn.CNN as CNN

DIM = 1000  # Number of bits in the bit strings (i.e. the "models").
NOISE_STDEV = 0.01  # Standard deviation of the simulated training noise.


class Model(object):
    """A class representing a model.

    It holds two attributes: `arch` (the simulated architecture) and `accuracy`
    (the simulated accuracy / fitness). See Appendix C for an introduction to
    this toy problem.

    In the real case of neural networks, `arch` would instead hold the
    architecture of the normal and reduction cells of a neural network and
    accuracy would be instead the result of training the neural net and
    evaluating it on the validation set.

    We do not include test accuracies here as they are not used by the algorithm
    in any way. In the case of real neural networks, the test accuracy is only
    used for the purpose of reporting / plotting final results.

    In the context of evolutionary algorithms, a model is often referred to as
    an "individual".

    Attributes:
      arch: the architecture as an int representing a bit-string of length `DIM`.
          As a result, the integers are required to be less than `2**DIM`. They
          can be visualized as strings of 0s and 1s by calling `print(model)`,
          where `model` is an instance of this class.
      accuracy:  the simulated validation accuracy. This is the sum of the
          bits in the bit-string, divided by DIM to produce a value in the
          interval [0.0, 1.0]. After that, a small amount of Gaussian noise is
          added with mean 0.0 and standard deviation `NOISE_STDEV`. The resulting
          number is clipped to within [0.0, 1.0] to produce the final validation
          accuracy of the model. A given model will have a fixed validation
          accuracy but two models that have the same architecture will generally
          have different validation accuracies due to this noise. In the context
          of evolutionary algorithms, this is often known as the "fitness".
    """

    def __init__(self):
        self.arch = None
        self.accuracy = None
        self.age = 0
        self.life = 0

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)


def train_and_eval(arch):
    """Simulates training and evaluation.

    Computes the simulated validation accuracy of the given architecture. See
    the `accuracy` attribute in `Model` class for details.

    Args:
      arch: the architecture as an int representing a bit-string.
    """
    accuracy = float(_sum_bits(arch)) / float(DIM)
    accuracy += random.gauss(mu=0.0, sigma=NOISE_STDEV)
    accuracy = 0.0 if accuracy < 0.0 else accuracy
    accuracy = 1.0 if accuracy > 1.0 else accuracy
    return accuracy


def _sum_bits(arch):
    """Returns the number of 1s in the bit string.

    Args:
      arch: an int representing the bit string.
    """
    total = 0
    for _ in range(DIM):
        total += arch & 1
        arch = (arch >> 1)
    return total


class NASModel(object):
    def __init__(self):
        self.normal_arch = None
        self.reduction_arch = None
        self.accuracy = None
        self.age = 0
        self.life = 0

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return '{0:b}'.format(self.arch)

    def train_NAS(self):
        trainloader, testloader, classes = train_cnn.load_dataset(train_cnn.DATASET_PATH)

        arch = {};

        for i in self.normal_arch:
            arch[i] = []
            for j in self.normal_arch[i]:
                if j[1] != 0:
                    arch[i].append((j[0], j[1]))

        cell_config_list = {'normal_cell': arch}
        print(cell_config_list)

        model = CNN.CNN(cell_config_list, class_num=len(classes))
        train_cnn.train(model, trainloader)
        return train_cnn.evaluate(model, testloader)
