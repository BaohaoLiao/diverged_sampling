import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Any
import torch
from math_verify import parse, LatexExtractionConfig, StringExtractionConfig, verify


def set_seed(seed: int = 42) -> None:
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    print(f"Random seed set as {seed}")


def prepare_prompt(question, tokenizer, data_name):
    if data_name in ["gpqa"]:
        prefix = (
            "Answer the following multiple choice question. "
            "The last line of your response should be of the following format: "
            "'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. "
            "Think step by step before answering.\n\n"
        )
        message = [
            {"role": "user", "content": prefix + "Question: " + question},
        ]
    else:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Question: " + question},
        ]
    prompt = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
    )
    assert "<think>" not in prompt, "The chat template should not inlcude <think> into the prompt."
    prompt = prompt + "<think>\n"
    return prompt


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


def eval(answer, solutions, data_name):
    if data_name in ["gpqa"]:
        abcd = "ABCD"
        answer = "$" + abcd[answer] + "$"
    else:
        answer = "$" + str(answer) + "$"
    parsed_ans = parse(answer)

    parsed_preds = []
    scores = []
    for solution in solutions:
        if data_name in ["gpqa"]:
            parsed_pred = parse(
                solution,
                extraction_config=[StringExtractionConfig(lowercase=False)],
            )
        else:
            parsed_pred = parse(
                solution, 
                extraction_config=[
                    LatexExtractionConfig(
                        boxed_match_priority=0, 
                        try_extract_without_anchor=True,
                    ),
                ]
            )
        scores.append(verify(parsed_ans, parsed_pred))
        if parsed_pred:
            parsed_preds.append(str(parsed_pred[0]))
        else:
            parsed_preds.append(None)
    return parsed_preds, scores


def majority_voting(preds, scores):
    if len(preds) != len(scores):
        raise ValueError("The lists 'preds' and 'scores' must have the same length")
    
    # Filter out None predictions and gather the valid predictions and scores
    valid_entries = [(pred, score) for pred, score in zip(preds, scores) if pred is not None]
    if not valid_entries:
        return None, 0.0
    
    # Count occurrences of each prediction
    prediction_counts = {}
    prediction_scores = {}
    
    for pred, score in valid_entries:
        if pred not in prediction_counts:
            prediction_counts[pred] = 0
            prediction_scores[pred] = score
        prediction_counts[pred] += 1
    
    # Find the most common prediction
    max_count = 0
    majority_pred = None
    
    for pred, count in prediction_counts.items():
        if count > max_count:
            max_count = count
            majority_pred = pred
    
    return majority_pred, prediction_scores[majority_pred]