#!/usr/bin/env python3
"""
BoN (Best-of-N) sanity-check for a math actor + reward model.

- Actor (policy): any HF causal LM (e.g., meta-llama/Llama-3.2-3B-Instruct)
- Reward model: any HF sequence-classification or value-head model that returns a scalar reward per (prompt, response)
- Task: MATH-style problems; default loader expects a JSONL with {"problem", "answer"} or will fall back to hendrycks/competition_math and sample 500 items ("MATH500").

Outputs:
- metrics.csv: BoN curves for N in {1,2,4,8,16,32} (configurable)
- bon_curve.png: plot of RM@N, Oracle@N, Random@N, LogP@N
- per_prompt_results.jsonl: raw candidates, scores, choices for debugging

This script performs *no training*. It only samples from the actor and re-ranks with the RM.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import pandas as pd

import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# Optional: datasets is used to pull hendrycks/competition_math when a local file isn't provided.
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False


# -------------------------
# Utility: set seeds
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Math answer verification
# -------------------------
from math_verify import parse, verify


def equal_answers(pred: str, gold: str) -> bool:
    """
    Compare predicted answer with gold answer using math_verify library.
    Returns True if answers are mathematically equivalent.
    """
    try:
        gold_parsed = parse(gold)
        pred_parsed = parse(pred)
        # Order matters: verify(gold, answer)
        return verify(gold_parsed, pred_parsed)
    except Exception:
        # Fallback to simple string comparison if parsing fails
        return pred.strip().lower() == gold.strip().lower()


# -------------------------
# Data loading
# -------------------------

def load_test_data(
    dataset_path: str = None, 
    seed: int = 0, 
    limit: int = 500, 
    split: str = 'test',
    problem_column: str = 'problem',
    answer_column: str = 'answer'
) -> List[Dict[str, str]]:
    """
    Return a list of {problem, answer}. Load from HuggingFace dataset repo or local JSONL file.
    
    Args:
        dataset_path: HuggingFace dataset repo (e.g., 'HuggingFaceH4/MATH-500') or local JSONL file path.
                     If None, defaults to 'hendrycks/competition_math'.
        seed: Random seed for shuffling
        limit: Maximum number of examples to return
        split: Dataset split ('train' or 'test') for HuggingFace datasets
        problem_column: Name of the column containing the problem/prompt
        answer_column: Name of the column containing the answer/solution
    """
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not installed.")
    
    # Default to hendrycks/competition_math if no path provided
    if not dataset_path:
        dataset_path = 'hendrycks/competition_math'
    
    # Try to load as HuggingFace dataset first
    try:
        # Check if it's a local file (exists and is a file)
        if os.path.exists(dataset_path) and os.path.isfile(dataset_path):
            # Load from local JSONL file
            rows = []
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    if problem_column in obj and answer_column in obj:
                        rows.append({'problem': obj[problem_column], 'answer': obj[answer_column]})
            if limit:
                rows = rows[:limit]
            return rows
        else:
            # Load as HuggingFace dataset repo
            # Special handling for hendrycks/competition_math which has 'all' config
            if 'competition_math' in dataset_path.lower():
                ds = load_dataset(dataset_path, 'all', split=split)
            # Special handling for openai/gsm8k which requires 'main' or 'socratic' config
            elif 'gsm8k' in dataset_path.lower():
                ds = load_dataset(dataset_path, 'main', split=split)
            else:
                ds = load_dataset(dataset_path, split=split)
            
            data = []
            for ex in ds:
                prob = ex.get(problem_column) or ex.get('problem') or ex.get('question') or ''
                ans = ex.get(answer_column) or ex.get('solution') or ex.get('answer') or ''
                if prob:
                    data.append({'problem': prob, 'answer': ans})
            rng = random.Random(seed)
            rng.shuffle(data)
            return data[:limit]
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset from '{dataset_path}': {e}")


# -------------------------
# Prompting utilities
# -------------------------

def build_chat_prompt(tokenizer, question: str, force_boxed: bool = True) -> Dict[str, Any]:
    system = (
        "You are a careful math tutor. Solve the problem step by step, and finish with a single line of the form "
        + ("FINAL_ANSWER: \\boxed{<answer>}" if force_boxed else "FINAL_ANSWER: <answer>")
        + "."
    )
    user = (
        "Problem:\n" + question.strip() + "\n\n"
        "Provide your reasoning briefly, then the final answer line exactly as requested."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        prompt = (
            f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT] "
        )
    return {"text": prompt}

def build_plain_prompt(question: str) -> Dict[str, Any]:
    """
    Plain (non-chat) prompt format for base causal LMs.
    Keep it minimal and avoid chat role markers.
    """
    prompt = (
        "Solve the following math problem.\n\n"
        "Problem:\n"
        f"{question.strip()}\n\n"
        "Answer:\n"
    )
    return {"text": prompt}

def build_prompt(tokenizer, question: str, prompt_format: str = "chat", force_boxed: bool = True) -> Dict[str, Any]:
    if prompt_format == "plain":
        return build_plain_prompt(question)
    return build_chat_prompt(tokenizer, question, force_boxed=force_boxed)


# -------------------------
# Scoring with RM
# -------------------------
@dataclass
class RMConfig:
    repo: str
    logits_index: int = 0  # if model returns >1 logits
    trust_remote_code: bool = True
    device_map: str = "auto"
    dtype: str = "bfloat16"


class RewardScorer:
    def __init__(self, cfg: RMConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.repo, use_fast=True, trust_remote_code=cfg.trust_remote_code)
        
        # Set padding token if not already defined (common issue with LLaMA-based models)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load config first to ensure custom attributes are loaded
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cfg.repo, trust_remote_code=cfg.trust_remote_code)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.repo,
            config=config,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            torch_dtype=getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else None,
        )
        
        # Update model's pad_token_id to match tokenizer
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        self.cfg = cfg

    @torch.no_grad()
    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        # pairs: list of (prompt, response)
        texts = []
        
        # Check if tokenizer has a chat template
        has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None
        
        if has_chat_template:
            # Use chat template for models that require it (e.g., ArmoRM)
            for prompt, response in pairs:
                conversation = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}
                ]
                formatted = self.tokenizer.apply_chat_template(
                    conversation, 
                    tokenize=False,
                    add_generation_prompt=False
                )
                texts.append(formatted)
        else:
            # Fallback: simple concatenation for models without chat template
            texts = [p + "\n\nAssistant: " + r for (p, r) in pairs]
        
        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=4096)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits.squeeze(-1)
        if logits.dim() == 1:
            scores = logits.detach().float().cpu().tolist()
        else:
            # choose one logit index if model has multi-logit head
            scores = logits[..., self.cfg.logits_index].detach().float().cpu().tolist()
        return scores


# -------------------------
# Actor sampling + logprob scoring
# -------------------------
@dataclass
class ActorConfig:
    repo: str
    trust_remote_code: bool = True
    device_map: str = "auto"
    dtype: str = "bfloat16"
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    eos_token: str = None


class Actor:
    def __init__(self, cfg: ActorConfig):
        self.tok = AutoTokenizer.from_pretrained(cfg.repo, use_fast=True, trust_remote_code=cfg.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.repo,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            torch_dtype=getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else None,
        )
        if self.tok.pad_token is None:
            # pad with eos to keep generate() happy
            self.tok.pad_token = self.tok.eos_token
        self.cfg = cfg

    @torch.no_grad()
    def generate_n(self, prompt: str, n: int) -> List[str]:
        inputs = self.tok([prompt], return_tensors='pt').to(self.model.device)
        out = self.model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=n,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_new_tokens=self.cfg.max_new_tokens,
            repetition_penalty=self.cfg.repetition_penalty,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        texts = self.tok.batch_decode(out, skip_special_tokens=True)
        # The decoded strings include the prompt; we need only the continuation part.
        prompt_len = inputs['input_ids'].shape[1]
        cont_ids = out[:, prompt_len:]
        cont_texts = self.tok.batch_decode(cont_ids, skip_special_tokens=True)
        return cont_texts

    @torch.no_grad()
    def avg_logprob(self, prompt: str, responses: List[str]) -> List[float]:
        # Compute per-token average logprob of the response under the actor, conditioning on prompt.
        scores = []
        for resp in responses:
            text = prompt + resp
            enc = self.tok(text, return_tensors='pt').to(self.model.device)
            # labels equal to input_ids but mask the prompt tokens
            labels = enc['input_ids'].clone()
            # Mask prompt part
            with self.tok.as_target_tokenizer():
                prompt_ids = self.tok(prompt, return_tensors='pt')['input_ids'][0]
            Lp = prompt_ids.shape[0]
            labels[0, :Lp] = -100
            out = self.model(**enc, labels=labels)
            # out.loss is averaged over unmasked tokens; recover token-wise logprobs
            # We compute explicitly to avoid any averaging surprises
            logits = out.logits[0, :-1, :]
            target = enc['input_ids'][0, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            token_lp = logprobs[torch.arange(target.shape[0]), target]
            # Only completion tokens contribute (prompt masked already)
            comp_lp = token_lp[Lp-1:]  # shift by one due to labels alignment
            if comp_lp.numel() == 0:
                scores.append(-1e9)
            else:
                scores.append(float(comp_lp.mean().item()))
        return scores


# -------------------------
# Utility: extract RM name from repo path
# -------------------------
def get_rm_name(repo: str) -> str:
    """Extract RM name from repo path, e.g., 'Skywork/Skywork-Reward-V2-Llama-3.1-8B' -> 'Skywork-Reward-V2-Llama-3.1-8B'"""
    return repo.split('/')[-1]


# -------------------------
# BoN evaluation
# -------------------------

def bon_eval(
    data: List[Dict[str, str]],
    actor: Actor,
    rm: RewardScorer,
    rm_repo: str,
    n_list: List[int],
    max_prompts: int,
    out_dir: str,
    beta_list: List[float] = (0.0, 0.5, 1.0),
    seed: int = 0,
    save_per_prompt: bool = True,
    prompt_format: str = "chat",
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    max_n = max(n_list)
    csv_path = os.path.join(out_dir, 'metrics.csv')
    raw_path = os.path.join(out_dir, 'per_prompt_results.jsonl')
    rm_name = get_rm_name(rm_repo)
    
    # Check if this is an incremental run (metrics.csv exists)
    is_incremental = os.path.exists(csv_path) and os.path.exists(raw_path)
    
    if is_incremental:
        print(f"Found existing results. Running incremental evaluation for RM: {rm_name}")
        return bon_eval_incremental(data, actor, rm, rm_name, n_list, max_prompts, out_dir, seed)
    
    # First run: generate candidates and compute all baselines
    print(f"First run. Generating candidates and computing baselines + RM: {rm_name}")
    fw = None
    if save_per_prompt:
        fw = open(raw_path, 'w', encoding='utf-8')

    # Stats by N
    acc_rm = {N: [] for N in n_list}
    acc_oracle = {N: [] for N in n_list}
    acc_random = {N: [] for N in n_list}
    acc_logp = {N: [] for N in n_list}
    
    results_csv = []

    for idx, ex in enumerate(tqdm(data[:max_prompts], desc="Prompts")):
        q = ex['problem']
        gold = ex['answer']
        prompt_obj = build_prompt(actor.tok, q, prompt_format=prompt_format)
        prompt = prompt_obj['text']
        # Generate candidates
        cand_texts = actor.generate_n(prompt, max_n)
        # Score with RM
        rm_scores = rm.score_batch([(prompt, c) for c in cand_texts])
        # LogP scores (avg)
        logp_scores = actor.avg_logprob(prompt, cand_texts)

        # Precompute correctness of each candidate for Oracle and later analysis
        correct_flags = [equal_answers(c, gold) for c in cand_texts]

        # For Random baseline reproducibility
        rnd_idx = list(range(max_n))
        random.shuffle(rnd_idx)

        # Persist raw (if enabled)
        if fw is not None:
            fw.write(json.dumps({
                'i': idx,
                'question': q,
                'prompt_text': prompt,
                'prompt_format': prompt_format,
                'gold': gold,
                'candidates': cand_texts,
                'rm_scores': rm_scores,
                'logp_scores': logp_scores,
                'correct': correct_flags,
            }) + "\n")

        # For each N, compute choices + correctness
        for N in n_list:
            # RM pick
            top_rm = int(np.argmax(rm_scores[:N]))
            acc_rm[N].append(1 if correct_flags[top_rm] else 0)
            # Oracle pick (if any correct among first N)
            acc_oracle[N].append(1 if any(correct_flags[:N]) else 0)
            # Random pick
            rand_pick = random.randrange(N)
            acc_random[N].append(1 if correct_flags[rand_pick] else 0)
            # LogP pick
            top_lp = int(np.argmax(logp_scores[:N]))
            acc_logp[N].append(1 if correct_flags[top_lp] else 0)

    if fw is not None:
        fw.close()

    # Aggregate + save CSV
    df_data = {'N': sorted(n_list)}
    df_data[rm_name] = [float(np.mean(acc_rm[N])) if acc_rm[N] else 0.0 for N in sorted(n_list)]
    df_data['Oracle@N'] = [float(np.mean(acc_oracle[N])) if acc_oracle[N] else 0.0 for N in sorted(n_list)]
    df_data['Random@N'] = [float(np.mean(acc_random[N])) if acc_random[N] else 0.0 for N in sorted(n_list)]
    df_data['LogP@N'] = [float(np.mean(acc_logp[N])) if acc_logp[N] else 0.0 for N in sorted(n_list)]
    
    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)

    # Plot
    plot_bon_curves(csv_path, out_dir)
    
    print(f"Saved metrics to {csv_path} and plot to bon_curve.png")
    if save_per_prompt:
        print(f"Saved per-prompt results to {raw_path}")
    else:
        print("Per-prompt results were not saved (use --save_per_prompt to enable)")


# -------------------------
# Incremental evaluation for additional RMs
# -------------------------

def bon_eval_incremental(
    data: List[Dict[str, str]],
    actor: Actor,
    rm: RewardScorer,
    rm_name: str,
    n_list: List[int],
    max_prompts: int,
    out_dir: str,
    seed: int = 0,
    prompt_format: str = "chat",
):
    """
    Incremental run: load cached candidates from per_prompt_results.jsonl,
    score with new RM, and update metrics.csv.
    """
    set_seed(seed)
    raw_path = os.path.join(out_dir, 'per_prompt_results.jsonl')
    csv_path = os.path.join(out_dir, 'metrics.csv')
    
    # Load existing CSV
    df_existing = pd.read_csv(csv_path)
    
    # Check if this RM already exists
    if rm_name in df_existing.columns:
        print(f"Warning: RM '{rm_name}' already exists in metrics.csv. Overwriting...")
    
    # Load cached results
    cached_results = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            cached_results.append(json.loads(line))
    
    if max_prompts is not None and len(cached_results) > max_prompts:
        cached_results = cached_results[:max_prompts]
    
    print(f"Loaded {len(cached_results)} cached results. Re-scoring with {rm_name}...")
    
    # Re-score with new RM
    acc_rm = {N: [] for N in n_list}
    
    for res in tqdm(cached_results, desc="Scoring with new RM"):
        # Prefer using the exact cached prompt text (critical for base-model/plain prompts).
        prompt_text = res.get('prompt_text')
        if not prompt_text:
            # Backward compat with older cache files
            q = res.get('question') or res.get('prompt') or ''
            prompt_text = build_prompt(actor.tok, q, prompt_format=prompt_format)['text']
        candidates = res['candidates']
        correct_flags = res['correct']
        
        # Score with new RM
        rm_scores = rm.score_batch([(prompt_text, c) for c in candidates])
        
        # For each N, compute RM accuracy
        for N in n_list:
            top_rm = int(np.argmax(rm_scores[:N]))
            acc_rm[N].append(1 if correct_flags[top_rm] else 0)
    
    # Add new RM column to dataframe
    rm_accuracies = [float(np.mean(acc_rm[N])) if acc_rm[N] else 0.0 for N in sorted(n_list)]
    df_existing[rm_name] = rm_accuracies
    
    # Save updated CSV
    df_existing.to_csv(csv_path, index=False)
    print(f"Updated {csv_path} with {rm_name}")
    
    # Replot
    plot_bon_curves(csv_path, out_dir)
    print(f"Updated plot: bon_curve.png")


# -------------------------
# Plotting function
# -------------------------

def plot_bon_curves(csv_path: str, out_dir: str):
    """
    Plot BoN curves from metrics.csv.
    Baselines (Oracle@N, Random@N, LogP@N) are plotted with dashed lines.
    RMs are plotted with solid lines.
    """
    df = pd.read_csv(csv_path)
    n_values = df['N'].values
    
    plt.figure(figsize=(10, 6))
    
    # Plot baselines with dashed lines
    baselines = ['Oracle@N', 'Random@N', 'LogP@N']
    baseline_colors = {'Oracle@N': 'black', 'Random@N': 'gray', 'LogP@N': 'brown'}
    
    for col in baselines:
        if col in df.columns:
            plt.plot(n_values, df[col].values, marker='o', linestyle='--', 
                    label=col, color=baseline_colors.get(col, None), alpha=0.7)
    
    # Plot RMs with solid lines
    rm_cols = [col for col in df.columns if col not in ['N'] + baselines]
    for col in rm_cols:
        plt.plot(n_values, df[col].values, marker='o', linestyle='-', label=col, linewidth=2)
    
    plt.xlabel('N (candidates)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Best-of-N Curves (Math)', fontsize=14)
    #plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'bon_curve.png'), bbox_inches='tight', dpi=150)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor_repo', type=str, required=False, default='meta-llama/Llama-3.2-3B-Instruct', help='HF repo for the actor')
    p.add_argument('--rm_repo', type=str, required=True, help='HF repo for reward model(s). Can be comma-separated list for multiple RMs.')
    p.add_argument('--rm_logits_index', type=int, default=0, help='If RM has multi-logit head, which index to use')
    p.add_argument('--dataset_path', type=str, default='', help='HuggingFace dataset repo (e.g., HuggingFaceH4/MATH-500) or local JSONL file path; defaults to hendrycks/competition_math')
    p.add_argument('--split', type=str, default='test', choices=['train', 'test'], help='Dataset split to use when loading from HuggingFace (train or test)')
    p.add_argument('--problem_column', type=str, default='problem', help='Name of the column containing the problem/prompt in your dataset')
    p.add_argument('--answer_column', type=str, default='solution', help='Name of the column containing the answer/solution in your dataset')
    p.add_argument('--max_prompts', type=int, default=None, help='Number of prompts to evaluate (defaults to all available prompts)')
    p.add_argument('--n_list', type=str, default='1,2,4,8,16,32', help='Comma-separated list of N values')
    p.add_argument('--out_dir', type=str, default='bon_out')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--max_new_tokens', type=int, default=512)
    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float16','bfloat16','float32'])
    p.add_argument('--no_save_per_prompt', action='store_true', help='Skip saving per-prompt results to JSONL file')
    p.add_argument('--prompt_format', type=str, default='chat', choices=['chat', 'plain'],
                   help="Prompt format for the actor: 'chat' for instruct/chat models, 'plain' for base models.")
    args = p.parse_args()

    set_seed(args.seed)

    # Parse RM repos (can be comma-separated)
    rm_repos = [r.strip() for r in args.rm_repo.split(',') if r.strip()]
    print(f"Testing {len(rm_repos)} reward model(s): {rm_repos}")

    actor_cfg = ActorConfig(
        repo=args.actor_repo,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )

    # Load data
    data = load_test_data(
        dataset_path=args.dataset_path or None, 
        seed=args.seed, 
        limit=500, 
        split=args.split,
        problem_column=args.problem_column,
        answer_column=args.answer_column
    )

    # Init actor (only once)
    actor = Actor(actor_cfg)

    # N list
    n_list = [int(x) for x in args.n_list.split(',') if x.strip()]

    # Process each RM
    for idx, rm_repo in enumerate(rm_repos):
        print(f"\n{'='*80}")
        print(f"Processing RM {idx+1}/{len(rm_repos)}: {rm_repo}")
        print(f"{'='*80}\n")
        
        # Init RM
        rm_cfg = RMConfig(repo=rm_repo, logits_index=args.rm_logits_index, dtype=args.dtype)
        rm = RewardScorer(rm_cfg)
        
        # Run evaluation
        bon_eval(
            data=data,
            actor=actor,
            rm=rm,
            rm_repo=rm_repo,
            n_list=n_list,
            max_prompts=args.max_prompts,
            out_dir=args.out_dir,
            seed=args.seed,
            save_per_prompt=not args.no_save_per_prompt,
            prompt_format=args.prompt_format,
        )
        
        # Clean up RM to free memory
        del rm
        torch.cuda.empty_cache()
        
    print(f"\n{'='*80}")
    print(f"Completed evaluation of all {len(rm_repos)} RMs!")
    print(f"Results saved to {args.out_dir}/metrics.csv and {args.out_dir}/bon_curve.png")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
