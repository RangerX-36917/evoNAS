import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import AgingEvolution
import NonAgingEvolution
import NewAgingEvolution
import pickle

# history = AgingEvolution.regularized_evolution(cycles=10000, population_size=10000, sample_size=50)
sns.set_style('white')
CYCLES = 100
POPULATION_SIZE = 10
SAMPLE_SIZE = 5
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
# try:
history = NewAgingEvolution.NAS_evolution(cycles=CYCLES, population_size=POPULATION_SIZE, sample_size=SAMPLE_SIZE)
# except Exception, e:
    # print(e.message)

det_str = pickle.dumps(history)


f = open('history','wb')  
data = {'k1':'python','k2':'java'}
f.write(pickle.dumps(data))  
f.close()

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
#plt.show()
