from data import WordsProximityDataset
import pandas as pd


def test_model_on_gt(model, pair_index, left, right, gt_left, gt_right):
    result = {
        "pair": (left, right),
        "slider": f"{left}_vs_{right}",
        "left_correct" : 0,
        "left_incorrect" : 0,
        "right_correct" : 0,
        "right_incorrect" : 0,
    }
    for word in gt_left:
        pred = model.predict_closest_word_in_pair(word, pair_index)
        result["left_correct"] += int(pred == left)
        result["left_incorrect"] += int(pred != left)

    for word in gt_right:
        pred = model.predict_closest_word_in_pair(word, pair_index)
        result["right_correct"] += int(pred == right)
        result["right_incorrect"] += int(pred != right)

    return result


def compute_metrics_by_pair(df_results):
    df_metrics = df_results[["slider", "left_correct", "left_incorrect", "right_correct", "right_incorrect"]]
    df_metrics["accuracy"] = (df_results["left_correct"] + df_results["right_correct"]) / (df_results["left_correct"] + df_results["right_correct"] + df_results["left_incorrect"] + df_results["right_incorrect"])
    df_metrics["left_support"] = df_results["left_correct"] + df_results["left_incorrect"]
    df_metrics["right_support"] = df_results["right_correct"] + df_results["right_incorrect"]
    return df_metrics


def test_model(proximity_dataset_path, model):
    ds = WordsProximityDataset(proximity_dataset_path)
    results = []

    for pair_index, (left, right), (gt_left, gt_right) in ds.gt_by_pairs():
        result = test_model_on_gt(model, pair_index, left, right, gt_left, gt_right)
        results.append(result)

    df_results = pd.DataFrame(results)
    df_metrics = compute_metrics_by_pair(df_results)
    return df_metrics


def test_model_and_save_results(proximity_dataset_path, model, results_dest):
    df_res = test_model(proximity_dataset_path, model)
    df_res.to_csv(results_dest, index=False)
