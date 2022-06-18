
import pandas as pd
import matplotlib.pyplot as plt
import os
def plot_learning_curves(history,dir, name):
    pic = os.path.join(dir, name)
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.savefig(pic, format='png')
    # plt.show()