import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x, axis=1):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=axis, keepdims=True)

MIN = -8
MAX = 8
NUM_SAMPLES = 100
sample_range = np.linspace(MIN, MAX, NUM_SAMPLES)
x, y = np.meshgrid(sample_range, sample_range)
xy = np.stack((x, y), axis=2)
probs = softmax(xy, axis=2)


fig = plt.figure()
ax = fig.add_subplot(111)
ax.pcolormesh(x, y, probs[...,0])
ax.spines['left'].set_position(('axes', 0.5))
ax.spines['bottom'].set_position(('axes', 0.5))

#ax = sns.heatmap(probs[...,0], xticklabels=sample_range, yticklabels=sample_range)
#ax.invert_yaxis()

plt.tight_layout()
plt.show()
#plt.savefig("softmax.pdf")



"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_surface(x, y, probs[...,0], cmap=cm.coolwarm,
                       linewidth=0, antialiased=False, proj_type='ortho')

# Customize the z axis.
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""

