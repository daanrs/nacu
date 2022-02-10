from main import main

import seaborn as sns
import numpy as np
import pandas as pd

pop, log = main(10)

fitness = np.array(log.chapters["fitness"].select("min"))
mean_size, max_size, min_size = log.chapters["size"].select(
    "mean", "max", "min"
)

df = pd.DataFrame({
    "best": fitness[:, 0],
    "size": fitness[:, 1],
    "mean_size": mean_size,
    "max_size": max_size,
    "min_size": min_size

}).reset_index().rename(columns={"index": "gen"})

sns.relplot(
    data = df, kind="line",
    x = "gen", y = "best"
).savefig("score_best.png")

sns.relplot(
    data=df, kind="line",
    x="gen", y="size"
).savefig("size_best.png")

#TODO: add max and min to it
sns.relplot(
    data=df, kind="line",
    x="gen", y="mean_size"
).savefig("mean_size.png")
