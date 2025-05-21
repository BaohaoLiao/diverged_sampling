import json
import random
import argparse
import numpy as np

from utils import set_seed, majority_voting


def main(args):
    # Load data
    with open(args.data_path, "r") as f:
        samples = json.load(f)

    ks = [int(k) for k in args.ks.split(",")]
    n = len(samples[0]["response"])
    assert max(ks) <= n
    print(f"#sample: {len(samples)}, n: {n}")

    results = {}
    for k in ks:
        avg_maj_scores = []
        for _ in range(50): # Sample 50 times
            inds = random.sample(range(n), k)
            inds.sort()
            maj_scores = []
            for sample in samples:
                preds = [sample["prediction"][i] for i in inds]
                scores = [sample["score"][i] for i in inds]
                _, maj_score = majority_voting(preds, scores)
                maj_scores.append(maj_score)
            avg_maj_scores.append(np.mean(maj_scores))

        results[f"maj@{k}"] = np.mean(avg_maj_scores)

    for k, v in results.items():
        print(k, ":", v)
            

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default=None, type=str, required=True)
    parser.add_argument("--ks", default="1,2,4,8,16,32", type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    main(args)