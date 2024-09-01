from datasets import load_dataset
import numpy as np

DATASET_PATH=""

ds = load_dataset(DATASET_PATH)

print("---TEST---")

gold_choices = [c[l] for c, l in zip(ds["test"]["choices"], ds["test"]["label"])]
distractor_choices = []

for c, l in zip(ds["test"]["choices"], ds["test"]["label"]):
    distractor_choices += [e for i, e in enumerate(c) if i != l]

print(len(gold_choices))
print(len(distractor_choices))

print(np.mean([len(a) for a in gold_choices]))
print(np.std([len(a) for a in gold_choices]))
print(np.mean([len(a) for a in distractor_choices]))
print(np.std([len(a) for a in distractor_choices]))

print("---TRAIN---")

gold_choices = [c[l] for c, l in zip(ds["train"]["choices"], ds["train"]["label"])]
distractor_choices = []

for c, l in zip(ds["train"]["choices"], ds["train"]["label"]):
    distractor_choices += [e for i, e in enumerate(c) if i != l]

print(len(gold_choices))
print(len(distractor_choices))

print(np.mean([len(a) for a in gold_choices]))
print(np.std([len(a) for a in gold_choices]))
print(np.mean([len(a) for a in distractor_choices]))
print(np.std([len(a) for a in distractor_choices]))

