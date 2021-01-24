import numpy as np
import matplotlib.pyplot as plt
from torch import tensor
import torch
import os

base = torch.stack([tensor([6.7640e-04, 1.1142e-02, 9.8913e-05, 2.2454e-05]),
                   tensor([7.6587e-04, 1.0868e-02, 7.0056e-05, 2.3997e-05]),
                   tensor([5.8964e-04, 1.1734e-02, 1.0515e-04, 2.3244e-05])])

gp = torch.stack([tensor([4.9825e-04, 7.2052e-03, 5.5085e-05, 2.9091e-06]),
                  tensor([5.0081e-04, 7.3089e-03, 5.5514e-05, 2.7657e-06]),
                  tensor([5.2291e-04, 7.5681e-03, 4.7569e-05, 2.6857e-06])]) 

plt.figure(figsize = (4,3.5))
labels = ['$m$', '$b$', '$\sigma$', '$f_r$']
base_mean = base.mean(axis = 0)
base_std = base.std(axis = 0)

gp_mean = gp.mean(axis = 0)
gp_std = gp.std(axis = 0)

# fig, ax = plt.subplots()
# ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize = (4.5,3.5))
rects1 = ax.bar(x - width/2, base_mean/base_mean, width, yerr = base_std/(base_mean*np.sqrt(3)), capsize=5)
rects2 = ax.bar(x + width/2, gp_mean/base_mean, width, yerr = gp_std/(base_mean*np.sqrt(3)), capsize=5)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Test squared error loss (scaled)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(['No regularization', 'Gradient regularization'], fontsize=12, frameon=True)




fig.tight_layout()
destination = os.path.join('./figures', 'sg_parameter_comp')
plt.savefig(destination, dpi = 150, bbox_inches="tight")
plt.show()