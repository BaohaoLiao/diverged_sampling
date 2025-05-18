import os
import json
import time
import random
import argparse
import numpy as np

import torch
import transformers
from vllm import LLM, SamplingParams
from math_verify import parse, LatexExtractionConfig, verify

from utils import load_data, parse_question


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
    system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    system_prompt_choice = (
        "Please reason step by step, and put your final answer within \\boxed{}.\n"
        "The last line of your response should be of the following format: "
        "'\\boxed{LETTER}' (without quotes) where LETTER is one of ABCD."
    )
    if data_name in ["gpqa"]:
        message = [
            {"role": "system", "content": system_prompt_choice},
            {"role": "user", "content": "Question: " + question},
        ]
    else:
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


def eval(answer, solutions):
    answer = "$" + str(answer) + "$"
    parsed_ans = parse(answer)

    parsed_preds = []
    scores = []
    for solution in solutions:
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


def main(args):
    # Out file
    model_name = args.model_name_or_path.split("/")[-1]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file_prefix = f"{output_dir}/{args.data_name}/{model_name}_" \
                  f"num{args.num_test_sample}_n{args.n_sampling}_totalTokens{args.max_tokens_per_call}"
    os.makedirs(f"{output_dir}/{args.data_name}", exist_ok=True)

    # Load and prepare data
    if "math500_level" in args.data_name:
        level = int(args.data_name.strip()[-1])
        examples = load_data("math500", args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(args.data_name, args.data_dir)

    if args.num_test_sample != -1:
        examples = examples[:args.num_test_sample]

    print("=" * 50)
    print(f"{args.data_name} || #samples: {len(examples)}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    samples = []
    for i, example in enumerate(examples):
        question = parse_question(example, args.data_name)
        prompt = prepare_prompt(question, tokenizer, args.data_name)
        samples.append({
            "idx": example["idx"],
            "question": question,
            "answer": example["answer"],
            "prompt": prompt,
        })
        if i == 0:
            print(prompt)

    # Load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        seed=args.seed,
        enable_prefix_caching=False,  # For efficiency
    )
    
    # Inference
    start_time = time.time()
    prompts = [sample["prompt"] for sample in samples]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens_per_call,
        #min_tokens=2,
        n=args.n_sampling,
        skip_special_tokens=False,
        seed=args.seed,
    )
    llm_outputs = llm.generate(
        prompts,
        sampling_params
    )
    llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
    assert len(samples) == len(llm_outputs)
    end_time = time.time()

    # Eval
    total_num_gen_tokens = 0
    for sample, llm_output in zip(samples, llm_outputs):
        total_num_gen_tokens += sum([len(o.token_ids) for o in llm_output.outputs])

        responses_per_question = [o.text for o in llm_output]
        solutions_per_question = []
        for response in responses_per_question:
            if "</think>" in response:
                solutions_per_question.append(response.split("</think>")[-1])
            else:
                solutions_per_question.append(response)


        preds, scores = eval(sample["answer"], solutions_per_question)
        maj_pred, maj_score = majority_voting(preds, scores)
        sample.update({
            "maj_pred": maj_pred,
            "maj_score": maj_score,
            "prediction": preds,
            "score": scores,
            "acc": float(np.mean(scores)),
            "response": responses_per_question,
        })

    acc = np.mean([sample["acc"] for sample in samples])
    maj_acc = np.mean([sample["maj_score"] for sample in samples])

    result_json = {
        "num_samples": len(samples),
        "acc": acc,
        "maj_acc": maj_acc,
        "num_total_tokens": total_num_gen_tokens,
        "time_use_in_min": (end_time - start_time) / 60,
    }
    print(result_json)

    # Save
    print(f"Saving {args.data_name} to {out_file_prefix}.json")
    json.dump(samples, open(f"{out_file_prefix}.json", "w",), indent=4)
    with open(f"{out_file_prefix}_metric.json", "w") as f:
        json.dump(result_json, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--data_dir", default="./datas", type=str)
    parser.add_argument("--data_name", default="aime24", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data

    # Model
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument('--max_model_len', type=int, default=40000)

    # Sampling
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--min_p", default=0., type=float)
    parser.add_argument("--top_k", default=-1, type=int)
    parser.add_argument("--max_tokens_per_call", default=32768, type=int)
    parser.add_argument("--n_sampling", default=4, type=int)

    # Save
    parser.add_argument("--output_dir", default="./output", type=str)

    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    # top_p must be 1 when using greedy sampling (vllm)
    args.top_p = 1 if args.temperature == 0 else args.top_p
    return args


if __name__ == "__main__":
    args = parse_args()
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print()

    set_seed(args.seed)
    main(args)