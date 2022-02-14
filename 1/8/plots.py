from main import main

import seaborn as sns
import numpy as np
import pandas as pd

pop, log = main(10)

fitness = np.array(log.chapters["fitness"].select("min"))
mean_size, max_size, min_size = log.chapters["size"].select(
    "mean", "max", "min"
)

df = (
    pd.DataFrame({
        "fitness": fitness[:, 0],
        "size": fitness[:, 1],
        "mean_size": mean_size,
        "max_size": max_size,
        "min_size": min_size

    }).reset_index()
    .rename(columns={"index": "gen"})
    .assign(
        # we minimized the positive value of the fitness, but in the
        # exercise they defined it as a negative value
        fitness=lambda x: -x["fitness"]
    )
)


sns.set_theme()

g = sns.relplot(
    data = df, kind="line",
    x = "gen", y = "fitness"
)
g.set_axis_labels("Generation", "Fitness")
g.ax.set_title("Fitness of best individual")
g.savefig("score_best.png")

g = sns.relplot(
    data=df, kind="line",
    x="gen", y="size"
)
g.set_axis_labels("Generation", "Size of best individual")
g.ax.set_title("Size of best individual")
g.savefig("size_best.png")

g = sns.relplot(
    data=df, kind="line",
    x="gen", y="mean_size",
)

g.ax.fill_between(df['gen'].to_numpy(), df['max_size'], df['min_size'], alpha=0.2)
g.set_axis_labels("Generation", "Mean size")
g.ax.set_title("Mean size of all individuals")
g.savefig("mean_size.png")
