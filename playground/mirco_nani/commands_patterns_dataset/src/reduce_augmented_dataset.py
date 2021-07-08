import pandas as pd
import sys

def reduce_augmented_dataset(source, dest):
    df=pd.read_csv(source)
    df_reduced=pd.concat([df[df["pattern_id"]==pattern_id].sample(100, random_state=42) for pattern_id in df["pattern_id"].unique()])
    df_reduced.to_csv(dest)


if __name__ == "__main__":
    reduce_augmented_dataset(sys.argv[1], sys.argv[2])
