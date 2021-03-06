import numpy as np
import pandas as pd
import subprocess as sp

from pathlib import Path
from sklearn.metrics import roc_auc_score

def main(k, n, r, p, fast, write, i):
    """
    Compute AUC for the files in folder p.

    Parameters
    ----------
    k: int
        the size of the chunks to split the strings into
    n: int
        the size of the strings to generate
    r: int
        the size of substrings to match
    p: path
        the folder with relevant data
    fast: bool
        whether to only keep the first 5 chunks in a string, this
        reduces performance somewhat but makes it (much) faster
    write: bool
        whether to write the dataframe with scores to a file
    """
    # read and split training data into chunks
    df_train = read_data(list(p.glob("*.train"))[0])
    df_train = explode_syscall(df_train, k, fast)

    # save the training data to a file, to give to the jar as input
    train_file = p / "train.csv"
    df_train["syscall"].to_csv(train_file, index=False, header=False)

    # read and split test data into chunks
    df_test = read_test(i, p)
    df_test = explode_syscall(df_test, k, fast)
    df_test = df_test.assign(syscall=lambda f: f["syscall"].astype(str))

    alphabet = list(p.glob("*.alpha"))[0]

    # run the java jar
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
    ).stdout.split()

    # create scores dataframe
    dfs = pd.DataFrame(scores).astype(float)

    # concat scores to truth
    df = pd.concat((dfs, df_test), axis=1).rename(columns={0: "score"})

    # combine the scores for the different chunks of an entry
    df = df.groupby("index").agg({"score": np.mean, "truth": np.any})

    # compute AUC
    score = roc_auc_score(y_true=df["truth"], y_score=df["score"])
    print(f"file={p.name}, score={score}, n={n}, r={r}, testset={i}")

    if write:
        df.to_csv(p / "result.csv", index=False)
    return score

def read_test(i, p):
    """
    Read test files in folder p
    """
    df_t = pd.concat([
        read_data(t)
        for t in p.glob(f"*{i}.test")
    ]).reset_index(drop=True)

    df_l = pd.concat([
        read_label(l)
        for l in p.glob(f"*{i}.labels")
    ]).reset_index(drop=True)

    df = pd.concat((df_t, df_l), axis=1)
    return df

def read_data(p):
    """
    Read a file containing syscall data
    """
    return (
        pd.read_csv(p, header=None)
        .rename(columns={0: "syscall"})
    )

def read_label(p):
    """
    Read file containing labels for syscall data
    """
    return (
        pd.read_csv(p, header=None)
        .rename(columns={0: "truth"})
        .assign(truth= lambda frame: frame["truth"].astype(bool))
    )

def explode_syscall(df, k, fast):
    """
    Explode the syscall data into seperate chunks
    """
    df = df.assign(
        syscall=lambda f: f["syscall"].apply(
            lambda s: partition_string(s, k, fast)
        )
    )
    df = df[df["syscall"].apply(len) > 0].reset_index(drop=True)
    df = df.explode("syscall").reset_index()
    return df

def partition_string(s, k, fast):
    """
    Partition string into chunks of size k.
    """
    ss = [ s[i:i+k] for i in range(0, len(s) - k + 1, k) ]
    if fast:
        return ss[:5]
    else:
        return [ s[i:i+k] for i in range(0, len(s) - k + 1, k) ]

if __name__ == "__main__":
    p = Path("snd-cert")

    # TODO: change these paramaters to get a decent score, then set
    # fast=False and write=True to run it on the complete data, and to
    # save the dataframe
    # main(10, 10, 4, p, fast=True, write=False)

    scores_cert = pd.DataFrame(
        {
            "auc": main(n, n, r, p, fast=True, write=False, i=i),
            "n": n,
            "r": r,
            "i": i
        }
        for n in range(8, 13)
        for r in range(2, 7)
        for i in range(1, 4)
    ).round(4)
    scores_cert.to_csv("scores_cert.csv", index=False)

    # TODO: change these paramaters to get a decent score, then set
    # fast=False and write=True to run it on the complete data, and to
    # save the dataframe
    p = Path("snd-unm")
    # main(10, 10, 4, p, fast=True, write=False)

    scores_unm = pd.DataFrame(
        {
            "auc": main(n, n, r, p, fast=True, write=False, i=i),
            "n": n,
            "r": r,
            "i": i
        }
        for n in range(8, 13)
        for r in range(2, 7)
        for i in range(1, 4)
    ).round(4)

    scores_unm.to_csv("scores_unm.csv", index=False)

    # OTHER NOTES:
        # we use nonoverlapping substrings of a fixed length k

        # the composite anomaly score is computed by taking the mean of the
        # chunks anomaly score
