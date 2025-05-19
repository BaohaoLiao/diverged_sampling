import os
import json
import time
import random
import argparse
import numpy as np

import torch
import transformers
from vllm import LLM, SamplingParams
from math_verify import parse, LatexExtractionConfig, StringExtractionConfig, verify

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
    # system_prompt_choice = (
    #     "Please reason step by step, and put your final answer within \\boxed{}.\n"
    #     "The last line of your response should be of the following format: "
    #     "'\\boxed{LETTER}' (without quotes) where LETTER is one of ABCD."
    # )
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


def single_step_generation(llm, sampling_params, prompts_per_question, final_step=False):
    num_prompts_per_question = [len(prompts) for prompts in prompts_per_question]
    flattened_prompts = [prompt for prompts in prompts_per_question for prompt in prompts]
    
    finished_prompt_thinkings = [[] for _ in num_prompts_per_question]
    continued_prompt_thinkings = [[] for _ in num_prompts_per_question]
    num_gen_tokens = 0
    
    if len(flattened_prompts) == 0:
        return finished_prompt_thinkings, continued_prompt_thinkings, num_gen_tokens, True        

    llm_outputs = llm.generate(
        flattened_prompts,
        sampling_params
    )
    llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
    assert sum(num_prompts_per_question) == len(llm_outputs)

    global_idx = 0
    for i, num in enumerate(num_prompts_per_question):
        for j in range(num):
            for o in llm_outputs[global_idx].outputs:
                num_gen_tokens += len(o.token_ids)
                if final_step:
                    finished_prompt_thinkings[i].append(llm_outputs[global_idx].prompt + o.text)
                else:
                    if o.finish_reason == "length":
                        continued_prompt_thinkings[i].append(llm_outputs[global_idx].prompt + o.text)
                    else:
                        finished_prompt_thinkings[i].append(llm_outputs[global_idx].prompt + o.text)

            global_idx += 1

    return finished_prompt_thinkings, continued_prompt_thinkings, num_gen_tokens, False


def solution_generation(llm, sampling_params, prompts_per_question):
    num_prompts_per_question = [len(prompts) for prompts in prompts_per_question]
    flattened_prompts = [prompt + "</think>" for prompts in prompts_per_question for prompt in prompts]

    llm_outputs = llm.generate(
        flattened_prompts,
        sampling_params
    )
    llm_outputs = sorted(llm_outputs, key=lambda x: int(x.request_id))
    assert sum(num_prompts_per_question) == len(llm_outputs)

    solutions_per_question = [[] for _ in num_prompts_per_question]
    global_idx = 0
    num_gen_tokens = 0
    for i, num in enumerate(num_prompts_per_question):
        for j in range(num):
            solutions_per_question[i].append(llm_outputs[global_idx].outputs[0].text)
            num_gen_tokens += len(llm_outputs[global_idx].outputs[0].token_ids)
            global_idx += 1

    return solutions_per_question, num_gen_tokens


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


def main(args):
    # Out file
    model_name = args.model_name_or_path.split("/")[-1]
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file_prefix = f"{output_dir}/{args.data_name}/{model_name}_" \
                  f"num{args.num_test_sample}_step{args.num_diverged_steps}_" \
                  f"init{args.init_n_sampling}_totalThinkingTokens{args.total_thinking_tokens}_stepTokens{args.max_tokens_per_step}"
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
    ## Sample thinking
    total_num_gen_tokens = 0
    finished_prompt_thinkings = [[] for _ in samples]
    continued_prompt_thinkings = [[sample["prompt"]] for sample in samples]
    for step in range(args.num_diverged_steps):
        print(f"Sampling thinking for step {step} ...")
        if step == args.num_diverged_steps - 1:
            max_tokens = args.total_thinking_tokens - step * args.max_tokens_per_step
        else:
            max_tokens = args.max_tokens_per_step

        if step == 0:
            n_sampling = args.init_n_sampling
        else:
            n_sampling = 2

        step_sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            top_k=args.top_k,
            max_tokens=max_tokens,
            #min_tokens=2,
            n=n_sampling,
            skip_special_tokens=False,
            seed=args.seed,
            stop=["</think>"],
        )

        step_finished_prompt_thinkings, continued_prompt_thinkings, step_num_gen_tokens, stop = single_step_generation(
            llm, step_sampling_params, continued_prompt_thinkings, final_step=(step==args.num_diverged_steps-1)
        )

        if stop:
            break

        total_num_gen_tokens += step_num_gen_tokens
        for i, step_finished_prompt_thinking in enumerate(step_finished_prompt_thinkings):
            finished_prompt_thinkings[i] += step_finished_prompt_thinking  

    ## Sample solution
    print(f"Sampling solution ...")
    solution_sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        min_p=args.min_p,
        top_k=args.top_k,
        max_tokens=args.answer_tokens,
        #min_tokens=2,
        n=1,
        skip_special_tokens=False,
        seed=args.seed,
    )
    solutions, solution_num_gen_tokens = solution_generation(llm, solution_sampling_params, finished_prompt_thinkings)
    assert len(samples) == len(solutions) == len(finished_prompt_thinkings)
    end_time = time.time()

    # Eval
    for sample, thinkings_per_question, solutions_per_question in zip(samples, finished_prompt_thinkings, solutions):
        preds, scores = eval(sample["answer"], solutions_per_question, args.data_name)
        maj_pred, maj_score = majority_voting(preds, scores)
        sample.update({
            "maj_pred": maj_pred,
            "maj_score": maj_score,
            "prediction": preds,
            "score": scores,
            "acc": float(np.mean(scores)),
            "prompt_and_thinking": thinkings_per_question,
            "solution": solutions_per_question,
        })

    acc = np.mean([sample["acc"] for sample in samples])
    maj_acc = np.mean([sample["maj_score"] for sample in samples])

    result_json = {
        "num_samples": len(samples),
        "acc": acc,
        "maj_acc": maj_acc,
        "num_thinking_tokens": total_num_gen_tokens,
        "num_solution_tokens": solution_num_gen_tokens,
        "num_total_tokens": total_num_gen_tokens + solution_num_gen_tokens,
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
    parser.add_argument("--total_thinking_tokens", default=16384, type=int)
    parser.add_argument("--max_tokens_per_step", default=2048, type=int)
    parser.add_argument("--answer_tokens", default=3072, type=int)
    parser.add_argument("--num_diverged_steps", default=7, type=int)
    parser.add_argument("--init_n_sampling", default=4, type=int)

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