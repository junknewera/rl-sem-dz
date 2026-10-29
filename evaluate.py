"""Greedy evaluation of the base model and LoRA checkpoints on one split.

    python evaluate.py --split calib                                  # CPU: base model, HF generate
    python evaluate.py --split test --vllm --adapters runs/*/final    # GPU: base + every adapter via vLLM
Appends one row per (model, task) to results/<split>.jsonl and prints accuracy and outcome shares.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import countdown as cd

MAX_TOKENS = 256
BATCH = 16  # HF generate batch on CPU


def generate_hf(adapter, prompts, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        cd.MODEL, dtype=torch.float32
    )  # default bf16 is very slow on CPU
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    tokenizer.padding_side = "left"
    out = []
    for i in range(0, len(prompts), BATCH):
        enc = tokenizer(
            prompts[i : i + BATCH],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        with torch.inference_mode():
            ids = model.generate(**enc, max_new_tokens=MAX_TOKENS, do_sample=False)
        out += tokenizer.batch_decode(
            ids[:, enc["input_ids"].shape[1] :], skip_special_tokens=True
        )
        print(f"{i + len(prompts[i : i + BATCH])}/{len(prompts)}", flush=True)
    return out


def generate_vllm(adapters, prompts):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=cd.MODEL,
        enable_lora=bool(adapters),
        max_lora_rank=32,  # LORA_R in train.py
        gpu_memory_utilization=0.3,
    )  # shared GPU: the default 0.9 would take memory from the other user
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    for i, adapter in enumerate([None] + adapters):
        lora = LoRARequest(adapter, i, adapter) if adapter else None
        yield (
            adapter,
            [
                o.outputs[0].text
                for o in llm.generate(prompts, params, lora_request=lora)
            ],
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--limit", type=int)
    p.add_argument("--vllm", action="store_true")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    tasks = cd.load(f"data/{args.split}.jsonl")[: args.limit]
    tokenizer = AutoTokenizer.from_pretrained(cd.MODEL)
    prompts = [cd.prompt_text(tokenizer, t) for t in tasks]
    if args.vllm:
        runs = generate_vllm(args.adapters, prompts)
    else:
        runs = ((a, generate_hf(a, prompts, tokenizer)) for a in [None] + args.adapters)

    Path(args.out).mkdir(exist_ok=True)
    with open(Path(args.out) / f"{args.split}.jsonl", "a") as f:
        for adapter, completions in runs:
            name = adapter or "base"
            outcomes = Counter()
            for i, (t, c) in enumerate(zip(tasks, completions)):
                outcome, value, _ = cd.check(c, t["numbers"], t["target"])
                outcomes[outcome] += 1
                f.write(
                    json.dumps(
                        {
                            "model": name,
                            "i": i,
                            "numbers": t["numbers"],
                            "target": t["target"],
                            "outcome": outcome,
                            "value": None if value is None else str(value),
                            "completion": c,
                        }
                    )
                    + "\n"
                )
            shares = {k: round(v / len(tasks), 3) for k, v in outcomes.most_common()}
            print(
                f"{name}: accuracy {outcomes['correct'] / len(tasks):.3f} on {len(tasks)} {args.split} {shares}"
            )
