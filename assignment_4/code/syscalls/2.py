import numpy as np
import pandas as pd
import subprocess as sp

from pathlib import Path
from sklearn.metrics import roc_auc_score

def main(k, n, r, p):
    df_train = read_data(list(p.glob("*.train"))[0])
    df_train = explode_syscall(df_train, k)

    train_file = p / "train.csv"
    df_train["syscall"].to_csv(train_file, index=False, header=False)

    df_test = read_test(p)
    df_test = explode_syscall(df_test, k)
    df_test = df_test.assign(syscall=lambda f: f["syscall"].astype(str)).dropna()
    # test_file = p / "test.csv"
    # df_test["syscall"].to_csv(test_file, index=False, header=False)

    alphabet = list(p.glob("*.alpha"))[0]

    q = [
        "java", "-jar", "../negsel2.jar", "-alphabet",
        f"file://{alphabet}", "-self", str(train_file), "-n", str(n),
        "-r", str(r), "-c", "-l",
    ]

    scores = sp.run(
        q,
        text=True,
        input="\n".join(df_test["syscall"]),
        capture_output=True
    ).stdout.split("\n")

    dfs = pd.DataFrame(scores[:-1]).astype(float)

    df = pd.concat((dfs, df_test), axis=1).rename(columns={0: "score"})
    df = df.groupby("index").agg({"score": np.mean, "truth": np.any})
    score = roc_auc_score(y_true=df["truth"], y_score=df["score"])
    print(f"file={p.name}, score={score}, n={n}, r={r}, chunksize={k}")
    return df

def read_test(p):
    df_t = pd.concat([
        read_data(t)
        for t in p.glob("*.test")
    ]).reset_index(drop=True)

    df_l = pd.concat([
        read_label(l)
        for l in p.glob("*.labels")
    ]).reset_index(drop=True)

    df = pd.concat((df_t, df_l), axis=1)
    return df

def read_data(p):
    return (
        pd.read_csv(p, header=None)
        .rename(columns={0: "syscall"})
    )

def read_label(p):
    return (
        pd.read_csv(p, header=None)
        .rename(columns={0: "truth"})
        .assign(truth= lambda frame: frame["truth"].astype(bool))
    )

def explode_syscall(df, k):
    df = df.assign(syscall=lambda f: f["syscall"].apply(lambda s: partition_string(s, k)))
    df = df[df["syscall"].apply(len) > 0].reset_index(drop=True)
    df = df.explode("syscall").reset_index()
    return df

def partition_string(s, k):
    """partition string into chunks of size k"""
    return [ s[i:i+k] for i in range(0, len(s) - k + 1, k) ]
    # if (2 * k) < len(s):
        # return [s[:k], s[k:(2* k)]]
    # else:
        # return []


if __name__ == "__main__":
    p = Path("snd-cert")
    main(10, 10, 4, p)
    p = Path("snd-unm")
    main(10, 10, 4, p)
