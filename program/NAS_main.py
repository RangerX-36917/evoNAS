import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import AgingEvolution
import NonAgingEvolution
import NewAgingEvolution
import pickle
import Model

# history = AgingEvolution.regularized_evolution(cycles=10000, population_size=10000, sample_size=50)
sns.set_style('white')
CYCLES = 100
POPULATION_SIZE = 10
SAMPLE_SIZE = 3
# history = AgingEvolution.NAS_evolution(cycles=10000, population_size=500, sample_size=50)
# xvalues = range(len(history))
# yvalues = [i.accuracy for i in history]
# ax = plt.gca()
# ax.scatter(
#     xvalues, yvalues, marker='.', facecolor=(0.0, 0.0, 0.0),
#     edgecolor=(0.0, 0.0, 0.0), linewidth=1, s=1)
#
# history = NonAgingEvolution.NAS_evolution(cycles=10000, population_size=500, sample_size=50)
# xvalues = range(len(history))
# yvalues = [i.accuracy for i in history]
# ax = plt.gca()
# ax.scatter(
#     xvalues, yvalues, marker='.', facecolor=(0.0, 0.0, 0.0),
#     edgecolor='r', linewidth=1, s=1)
dir = "6_1/"
# try:
population = []

model = Model.NASModel()
# set the init architecture below
model.normal_arch = {2: [(0, 1), (1, 7)],
                     3: [(0, 0), (1, 1), (2, 7)],
                     4: [(0, 0), (1, 0), (2, 1), (3, 7)],
                     5: [(0, 0), (1, 0), (2, 0), (3, 1), (4, 7)],
                     6: [(0, 0), (1, 0), (2, 0), (3, 0), (4, 1), (5, 7)]}

# model.reduction_arch=
model.accuracy = model.train_NAS()
model.age = POPULATION_SIZE - 1
model.life = POPULATION_SIZE
population.append(model)

history = NewAgingEvolution.NAS_evolution(pop=population, cycles=CYCLES, population_size=POPULATION_SIZE,
                                          sample_size=SAMPLE_SIZE, dir=dir)
# except Exception, e:
#     print(e.message)


with open(dir + 'history', "wb") as f:
    pickle.dump(history, f)

# f = open(dir+'history','wb')  
# det_str = pickle.dumps(history)

# data = {'k1':'python','k2':'java'}
# f.write(pickle.dumps(det_str))  
# f.close()

xvalues = range(len(history))
yvalues = [i.accuracy for i in history]
ax = plt.gca()
ax.scatter(
    xvalues, yvalues, marker='.', facecolor=(0.0, 0.0, 0.0),
    edgecolor='b', linewidth=1, s=1)

ax.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
ax.yaxis.set_major_locator(ticker.LinearLocator(numticks=2))
ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

fig = plt.gcf()
fig.set_size_inches(8, 6)
fig.tight_layout()
ax.tick_params(
    axis='x', which='both', bottom='on', top='off', labelbottom='on',
    labeltop='off', labelsize=14, pad=10)
ax.tick_params(
    axis='y', which='both', left='on', right='off', labelleft='on',
    labelright='off', labelsize=14, pad=5)
plt.xlabel('Number of Models Evaluated', labelpad=-16, fontsize=16)
plt.ylabel('Accuracy', labelpad=-30, fontsize=16)
plt.xlim(0, 10000)

sns.despine()
# plt.show()
