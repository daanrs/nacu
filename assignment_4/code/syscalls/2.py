import pandas as pd

from pathlib import Path
# from sklearn.metrics import roc_auc_curve


def main(n):
    p = Path("snd-cert")

    df = (
        pd.read_csv(p / "snd-cert.train", header=None)
        .rename(columns={0: "syscall"})
    )

    df["syscall"] = df["syscall"].apply(lambda s: partition_string(s, n))
    df = df.explode("syscall")
    df = df.reset_index()
    df = df.rename(columns={"index": "id"})
    return df


def partition_string(s, n):
    """partition string into chunks of size n"""
    ss = []
    for i in range(0, len(s) - n, n):
        ss.append(s[i:i+n])
    return ss
