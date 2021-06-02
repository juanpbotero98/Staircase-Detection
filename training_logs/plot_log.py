import pandas
import os 
import numpy as np
import matplotlib.pyplot as plt 
here = os.path.dirname(os.path.abspath(__file__))
csv_pth = os.path.join(here,'training_logs.csv')
df = pandas.read_csv(csv_pth)

total_loss = df['loss'].to_numpy()
loss_classifier = df['loss_classifier'].to_numpy()
loss_box_reg = df['loss_box_reg'].to_numpy()

steps = np.linspace(0,18950,380)

fig, axs = plt.subplots(1, 3)
axs[0].plot(steps, total_loss)
axs[0].set_xlabel('steps')
axs[0].set_ylabel('Total loss')
axs[0].xticks(np.arange(0,18951,1000))
fig.suptitle('Training losses', fontsize=16)

axs[1].plot(steps,loss_classifier)
axs[1].set_xlabel('steps')
axs[1].set_ylabel('Classifier loss')
axs[1].xticks(np.arange(0,18951,1000))

axs[2].plot(steps,loss_box_reg)
axs[2].set_xlabel('steps')
axs[2].set_ylabel('Classifier box regressor')
axs[2].xticks(np.arange(0,18951,1000))

fig.tight_layout()

plt.savefig(os.path.join(here,'training losses'))