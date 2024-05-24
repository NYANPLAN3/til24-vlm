"""Taken from https://github.com/TIL-24/til-24-base/blob/main/test_vlm.py."""

import base64
import itertools
import json
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from eval.vlm_eval import vlm_eval

load_dotenv()

TEAM_NAME = os.getenv("TEAM_NAME")
TEAM_TRACK = os.getenv("TEAM_TRACK")
HOME = os.getenv("HOME")


# J-H: the original eval code takes too long to load so I multiprocess'ed it.
def load_image(input_dir, entry):
    out = []
    with open(input_dir / "images" / entry["image"], "rb") as file:
        image_bytes = file.read()
    for a in entry["annotations"]:
        out.append(
            (
                {
                    "caption": a["caption"],
                    "b64": base64.b64encode(image_bytes).decode("ascii"),
                },
                {"caption": a["caption"], "bbox": a["bbox"]},
            )
        )
    return out


def main():
    input_dir = Path(f"{HOME}/{TEAM_TRACK}")
    results_dir = Path("results")

    results_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    with open(input_dir / "vlm.jsonl", "r") as f:
        for line in f:
            l = line.strip()
            if l == "":
                continue
            entries.append((input_dir, json.loads(l)))

    # Don't do all
    random.seed(42)
    entries = random.sample(entries, 100)

    with mp.Pool(processes=os.cpu_count()) as pool:
        loaded = itertools.chain.from_iterable(
            pool.starmap(load_image, entries, chunksize=len(entries) // os.cpu_count())
        )

    instances = []
    truths = []

    for i, (instance, truth) in enumerate(loaded):
        instance["key"] = i
        truth["key"] = i
        instances.append(instance)
        truths.append(truth)

    assert len(truths) == len(instances)
    results = run_batched(instances)
    df = pd.DataFrame(results)
    assert len(truths) == len(results)
    df.to_csv(results_dir / "vlm_results.csv", index=False)
    # calculate eval
    eval_result = vlm_eval(
        [truth["bbox"] for truth in truths],
        [result["bbox"] for result in results],
    )
    print(f"IoU@0.5: {eval_result}")


def run_batched(
    instances: List[Dict[str, str | int]], batch_size: int = 4
) -> List[Dict[str, str | int]]:
    # split into batches
    results = []
    for index in tqdm(range(0, len(instances), batch_size)):
        _instances = instances[index : index + batch_size]
        response = requests.post(
            "http://localhost:5004/identify",
            # "http://172.17.0.1:5004/identify",
            data=json.dumps(
                {
                    "instances": [
                        {field: _instance[field] for field in ("key", "caption", "b64")}
                        for _instance in _instances
                    ]
                }
            ),
        )
        _results = response.json()["predictions"]
        results.extend(
            [
                {
                    "key": _instances[i]["key"],
                    "bbox": _results[i],
                }
                for i in range(len(_instances))
            ]
        )
    return results


if __name__ == "__main__":
    main()
