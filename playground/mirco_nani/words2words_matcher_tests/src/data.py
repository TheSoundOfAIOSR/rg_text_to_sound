import pandas as pd
import numpy as np


class SlidersDataset:
    def __init__(self, 
        data_path,
        words_pairs = [
            ("Bright", "Dark"),
            ("Full",   "Hollow"),
            ("Smooth", "Rough"),
            ("Warm",   "Metallic"),
            ("Clear",  "Muddy"),
            ("Thin",   "Thick"),
            ("Pure",   "Noisy"),
            ("Rich",   "Sparse"),
            ("Soft",   "Hard")
    ]):
        self.df = pd.read_csv(data_path)
        self.words_pairs = words_pairs

    def get_gt_at_threshold(self, uncertainty_threshold):
        sliders = [f"{l}_vs_{r}".lower() for l,r in self.words_pairs]
        df_gt = self.df[["description"]]
        for s, (l,r) in zip(sliders, self.words_pairs):
            df_gt[s]=self.df[s].apply(lambda x: l if x < 50 - uncertainty_threshold else (r if x > 50 + uncertainty_threshold else None) )
        df_res = df_gt[["description"]]
        df_res["gt"]=df_gt[sliders].apply(lambda x : x.values[x.values != np.array(None)], axis=1)
        return df_res

    