import pandas as pd

from sklearn.metrics import roc_auc_score
from pathlib import Path


def main():

    scores_path = Path("lang1_scores")
    for i in [1, 4, 9]:
        eng_p = scores_path / f"{i}_english.txt"
        tag_p = scores_path / f"{i}_tagalog.txt"

        df_eng = english(eng_p)
        df_tag = other(tag_p)
        df = pd.concat((df_tag, df_eng))
        score = roc_auc_score(y_true=df["truth"], y_score=df["score"])
        print(f"r={i}, score={score}")

    scores_path = Path("lang2_scores")
    df_english = english("lang1_scores/4_english.txt")
    for p in scores_path.iterdir():
        df_lang = other(p)
        df = pd.concat((df_lang, df_english))
        score = roc_auc_score(y_true=df["truth"], y_score=df["score"])
        print(f"file={p.name}, score={score}")


def english(filename):
    return pd.read_csv(
        filename,
        header=None
    ).rename(columns={0: "score"}).assign(truth=False)


def other(filename):
    return pd.read_csv(
        filename,
        header=None,
    ).rename(columns={0: "score"}).assign(truth=True)

if __name__ == "__main__":
    main()
