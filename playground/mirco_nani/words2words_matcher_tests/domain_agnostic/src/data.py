import json

class WordsProximityDataset:
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
        self.data = json.load(open(data_path, "r"))
        self.words_pairs = words_pairs
    

    def gt_by_pairs(self):
        for pair_index, (left, right) in enumerate(self.words_pairs):
            yield pair_index, (left, right), (self.data[left.lower()], self.data[right.lower()])