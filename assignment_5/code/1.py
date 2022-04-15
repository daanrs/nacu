import pandas as pd
import numpy as np

df = pd.read_csv("1.csv", header=[0, 1])

dfv = (
    df.stack(level=0)
    .groupby(axis=0, level=0)
    .agg([np.mean, np.max, np.min, np.prod])
    .swaplevel(0, 1, axis=1)
    .reindex(["mean", "amax", "amin", "prod"], level=0, axis=1)
)

dfv.to_csv("1_filled.csv", index=False)
