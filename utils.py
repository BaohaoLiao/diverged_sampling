import os
import json
from pathlib import Path
from typing import Iterable, Union, Any, Dict


def parse_question(example, data_name):
    question = ""
    if data_name in ["mmlu_stem", "gpqa"]:
        options = example["choices"]
        assert len(options) == 4
        for i, (label, option) in enumerate(zip("ABCD", options)):
            options[i] = f"{label}. {str(option).strip()}\n"
        options = " ".join(options).strip()
        question = f"{example['question'].strip()}\n\n {options}"
    else:
        for key in ["question", "problem", "Question", "input"]:
            if key in example:
                question = example[key]
                break
    return question.strip()


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()


def load_data(data_name, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/test.jsonl"
    assert os.path.exists(data_file)
    examples = list(load_jsonl(data_file))

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples