import matplotlib.pyplot as plt
import pandas as pd

def plot_results(rl_vals, ac_vals):
    df = pd.DataFrame({"RL": rl_vals, "AC": ac_vals})

    df.hist(bins=50, figsize=(10,5))
    plt.tight_layout()
    plt.show()

    print(df.describe())
    print("RL mean:", df["RL"].mean())
    print("AC mean:", df["AC"].mean())
