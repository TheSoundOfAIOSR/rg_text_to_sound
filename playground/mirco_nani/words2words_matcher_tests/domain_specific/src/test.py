from data import SlidersDataset
import pandas as pd


def test_model_on_gt(model, df_gt):
    df = df_gt[["gt"]]
    df["pred"] = df_gt["description"].apply(lambda x: model.predict([w.strip() for w in x.split(",")]))
    df["TP"] = df[["pred","gt"]].apply(lambda x: len(set(x["gt"]).intersection(set(x["pred"]))), axis=1)
    df["FP"] = df["pred"].map(len) - df["TP"]
    df["FN"] = df["gt"].map(len) - df["TP"]
    return df[["TP","FP","FN"]].sum().to_dict()

def compute_metrics(df_res):
    df = df_res.copy()
    df["precision"] = df["TP"]/(df["TP"]+df["FP"])
    df["recall"] = df["TP"]/(df["TP"]+df["FN"])
    df["accuracy"] = df["TP"]/(df["TP"]+df["FP"]+df["FN"])
    df["F1_score"] = 2/(1/df["precision"]+1/df["recall"])
    return df


def test_model(sliders_dataset_path, model):
    ds = SlidersDataset(sliders_dataset_path)
    results = []
    for uncertainty_threshold in [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]:
        df_gt = ds.get_gt_at_threshold(uncertainty_threshold)
        res = test_model_on_gt(model, df_gt)
        res["uncertainty_threshold"]=uncertainty_threshold
        results.append(res)
    df_res = compute_metrics(pd.DataFrame(results))
    return df_res


def test_model_and_save_results(sliders_dataset_path, model, results_dest):
    df_res = test_model(sliders_dataset_path, model)
    df_res.to_csv(results_dest, index=False)