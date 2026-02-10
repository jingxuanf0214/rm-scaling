#!/usr/bin/env python3
"""
BoN (Best-of-N) sanity-check for a math actor + reward model
+ optional IFEval mode matching lm-eval-harness prompt/gen/scoring.

When --ifeval is set:
- Dataset: google/IFEval (split=train, matching harness task yaml)
- Prompt: doc["prompt"] exactly
- Scoring: exactly via lm-eval-harness's lm_eval.tasks.ifeval.utils.process_results
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

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

# Optional: datasets is used to pull HF datasets
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
        return verify(gold_parsed, pred_parsed)
    except Exception:
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
    """
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not installed.")

    if not dataset_path:
        dataset_path = 'hendrycks/competition_math'

    try:
        if os.path.exists(dataset_path) and os.path.isfile(dataset_path):
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
            if 'competition_math' in dataset_path.lower():
                ds = load_dataset(dataset_path, 'all', split=split)
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


def load_ifeval_data(seed: int = 0, limit: Optional[int] = None, split: str = "train") -> List[Dict[str, Any]]:
    """
    Load google/IFEval exactly as lm-eval-harness task yaml uses:
    dataset_path: google/IFEval
    test_split: train
    doc_to_text: prompt
    """
    if not HAS_DATASETS:
        raise RuntimeError("datasets library not installed.")
    ds = load_dataset("google/IFEval", split=split)
    rows = [dict(ex) for ex in ds]
    # Harness does not shuffle by default; but your script previously shuffled.
    # To keep your BoN script behavior reproducible while not changing prompt/scoring,
    # we shuffle here under your seed.
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit is not None:
        rows = rows[:limit]
    return rows


# -------------------------
# Prompting utilities (math mode)
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
        prompt = f"<s>[SYSTEM]\n{system}\n[/SYSTEM]\n[USER]\n{user}\n[/USER]\n[ASSISTANT] "
    return {"text": prompt}


def build_plain_prompt(question: str) -> Dict[str, Any]:
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
# IFEval scoring (exact via lm-eval-harness)
# -------------------------
def import_ifeval_utils_exact():
    """
    Import local IFEval utils from the copied ifeval folder.
    """
    try:
        # Import from local ifeval folder
        from ifeval import utils as ifeval_utils  # type: ignore
        return ifeval_utils
    except Exception as e:
        raise RuntimeError(
            "Could not import local IFEval utils from ifeval folder. "
            "Make sure the ifeval folder is in the same directory as this script.\n"
            f"Import error: {e}"
        )


def ifeval_process_results(ifeval_utils, doc: Dict[str, Any], gen_text: str) -> Dict[str, Any]:
    """
    Call harness process_results with best-effort signature compatibility.
    In harness YAML: process_results: !function utils.process_results
    For generate_until tasks, harness passes `results` as a list of generations.
    """
    # Most harness task process_results signatures are (doc, results)
    # where results is a list with one string for generate_until.
    try:
        return ifeval_utils.process_results(doc, [gen_text])
    except TypeError:
        # Some variants may pass the raw string
        return ifeval_utils.process_results(doc, gen_text)


def inst_flags_to_counts(flags: Any) -> Tuple[int, int]:
    """
    Convert harness per-doc inst_level_* output into (num_followed, num_total).
    Harness typically returns a list[bool/int] per doc for instruction-level metrics.
    """
    if flags is None:
        return (0, 0)
    if isinstance(flags, (int, float, bool)):
        # Unexpected, but handle gracefully
        v = int(bool(flags))
        return (v, 1)
    if isinstance(flags, (list, tuple)):
        vals = [int(bool(x)) for x in flags]
        return (sum(vals), len(vals))
    # Unknown type
    return (0, 0)


# -------------------------
# Scoring with RM
# -------------------------
@dataclass
class RMConfig:
    repo: str
    logits_index: int = 0
    trust_remote_code: bool = True
    device_map: str = "auto"
    dtype: str = "bfloat16"


class RewardScorer:
    def __init__(self, cfg: RMConfig):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.repo, use_fast=True, trust_remote_code=cfg.trust_remote_code)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(cfg.repo, trust_remote_code=cfg.trust_remote_code)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.repo,
            config=config,
            trust_remote_code=cfg.trust_remote_code,
            device_map=cfg.device_map,
            torch_dtype=getattr(torch, cfg.dtype) if hasattr(torch, cfg.dtype) else None,
        )

        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id

        self.cfg = cfg

    @torch.no_grad()
    def score_batch(self, pairs: List[Tuple[str, str]]) -> List[float]:
        texts = []
        has_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None

        if has_chat_template:
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
            texts = [p + "\n\nAssistant: " + r for (p, r) in pairs]

        enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=4096)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        out = self.model(**enc)
        logits = out.logits.squeeze(-1)
        if logits.dim() == 1:
            return logits.detach().float().cpu().tolist()
        return logits[..., self.cfg.logits_index].detach().float().cpu().tolist()


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
            self.tok.pad_token = self.tok.eos_token
        self.cfg = cfg

    @torch.no_grad()
    def generate_n(
        self,
        prompt: str,
        n: int,
        *,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.0,
    ) -> List[str]:
        """
        Generate N continuations.

        Note: If do_sample=False (greedy), transformers typically only supports 1 return sequence.
        In that case we generate once and replicate N times (or loop N times).
        """
        inputs = self.tok([prompt], return_tensors='pt').to(self.model.device)

        if not do_sample:
            out = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            prompt_len = inputs['input_ids'].shape[1]
            cont_ids = out[:, prompt_len:]
            cont_text = self.tok.batch_decode(cont_ids, skip_special_tokens=True)[0]
            return [cont_text for _ in range(n)]

        out = self.model.generate(
            **inputs,
            do_sample=True,
            num_return_sequences=n,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        prompt_len = inputs['input_ids'].shape[1]
        cont_ids = out[:, prompt_len:]
        cont_texts = self.tok.batch_decode(cont_ids, skip_special_tokens=True)
        return cont_texts

    @torch.no_grad()
    def avg_logprob(self, prompt: str, responses: List[str]) -> List[float]:
        scores = []
        for resp in responses:
            text = prompt + resp
            enc = self.tok(text, return_tensors='pt').to(self.model.device)
            labels = enc['input_ids'].clone()

            prompt_ids = self.tok(prompt, return_tensors='pt')['input_ids'][0]
            Lp = prompt_ids.shape[0]
            labels[0, :Lp] = -100

            out = self.model(**enc, labels=labels)
            logits = out.logits[0, :-1, :]
            target = enc['input_ids'][0, 1:]
            logprobs = torch.log_softmax(logits, dim=-1)
            token_lp = logprobs[torch.arange(target.shape[0]), target]

            comp_lp = token_lp[Lp-1:]
            if comp_lp.numel() == 0:
                scores.append(-1e9)
            else:
                scores.append(float(comp_lp.mean().item()))
        return scores


# -------------------------
# Utility: extract RM name from repo path
# -------------------------
def get_rm_name(repo: str) -> str:
    return repo.split('/')[-1]


# -------------------------
# Plotting
# -------------------------
def plot_bon_curves_generic(csv_path: str, out_dir: str, title: str, y_label: str, png_name: str):
    df = pd.read_csv(csv_path)
    n_values = df['N'].values

    plt.figure(figsize=(10, 6))

    # Baselines first if present
    baseline_cols = [c for c in df.columns if c.startswith("Oracle") or c.startswith("Random") or c.startswith("LogP")]
    for col in baseline_cols:
        plt.plot(n_values, df[col].values, marker='o', linestyle='--', label=col, alpha=0.7)

    rm_cols = [c for c in df.columns if c not in ['N'] + baseline_cols]
    for col in rm_cols:
        plt.plot(n_values, df[col].values, marker='o', linestyle='-', label=col, linewidth=2)

    plt.xlabel('N (candidates)', fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, png_name), bbox_inches='tight', dpi=150)
    plt.close()


# -------------------------
# BoN eval (math mode)
# -------------------------
def bon_eval_math(
    data: List[Dict[str, str]],
    actor: Actor,
    rm: RewardScorer,
    rm_repo: str,
    n_list: List[int],
    max_prompts: Optional[int],
    out_dir: str,
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

    is_incremental = os.path.exists(csv_path) and os.path.exists(raw_path)
    if is_incremental:
        print(f"Found existing results. Running incremental evaluation for RM: {rm_name}")
        return bon_eval_math_incremental(data, actor, rm, rm_name, n_list, max_prompts, out_dir, seed, prompt_format)

    print(f"First run. Generating candidates and computing baselines + RM: {rm_name}")
    fw = open(raw_path, 'w', encoding='utf-8') if save_per_prompt else None

    acc_rm = {N: [] for N in n_list}
    acc_oracle = {N: [] for N in n_list}
    acc_random = {N: [] for N in n_list}
    acc_logp = {N: [] for N in n_list}

    for idx, ex in enumerate(tqdm(data[:max_prompts] if max_prompts else data, desc="Prompts")):
        q = ex['problem']
        gold = ex['answer']
        prompt = build_prompt(actor.tok, q, prompt_format=prompt_format)['text']

        cand_texts = actor.generate_n(
            prompt, max_n,
            do_sample=True,
            temperature=actor.cfg.temperature,
            top_p=actor.cfg.top_p,
            max_new_tokens=actor.cfg.max_new_tokens,
            repetition_penalty=actor.cfg.repetition_penalty,
        )

        rm_scores = rm.score_batch([(prompt, c) for c in cand_texts])
        logp_scores = actor.avg_logprob(prompt, cand_texts)
        correct_flags = [equal_answers(c, gold) for c in cand_texts]

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

        for N in n_list:
            top_rm = int(np.argmax(rm_scores[:N]))
            acc_rm[N].append(1 if correct_flags[top_rm] else 0)

            acc_oracle[N].append(1 if any(correct_flags[:N]) else 0)

            rand_pick = random.randrange(N)
            acc_random[N].append(1 if correct_flags[rand_pick] else 0)

            top_lp = int(np.argmax(logp_scores[:N]))
            acc_logp[N].append(1 if correct_flags[top_lp] else 0)

    if fw is not None:
        fw.close()

    df_data = {'N': sorted(n_list)}
    df_data[rm_name] = [float(np.mean(acc_rm[N])) if acc_rm[N] else 0.0 for N in sorted(n_list)]
    df_data['Oracle@N'] = [float(np.mean(acc_oracle[N])) if acc_oracle[N] else 0.0 for N in sorted(n_list)]
    df_data['Random@N'] = [float(np.mean(acc_random[N])) if acc_random[N] else 0.0 for N in sorted(n_list)]
    df_data['LogP@N'] = [float(np.mean(acc_logp[N])) if acc_logp[N] else 0.0 for N in sorted(n_list)]

    df = pd.DataFrame(df_data)
    df.to_csv(csv_path, index=False)

    # existing plot name
    plot_bon_curves_generic(csv_path, out_dir, "Best-of-N Curves (Math)", "Accuracy", "bon_curve.png")

    print(f"Saved metrics to {csv_path} and plot to bon_curve.png")
    if save_per_prompt:
        print(f"Saved per-prompt results to {raw_path}")


def bon_eval_math_incremental(
    data: List[Dict[str, str]],
    actor: Actor,
    rm: RewardScorer,
    rm_name: str,
    n_list: List[int],
    max_prompts: Optional[int],
    out_dir: str,
    seed: int = 0,
    prompt_format: str = "chat",
):
    set_seed(seed)
    raw_path = os.path.join(out_dir, 'per_prompt_results.jsonl')
    csv_path = os.path.join(out_dir, 'metrics.csv')

    df_existing = pd.read_csv(csv_path)
    if rm_name in df_existing.columns:
        print(f"Warning: RM '{rm_name}' already exists in metrics.csv. Overwriting...")

    cached_results = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            cached_results.append(json.loads(line))

    if max_prompts is not None and len(cached_results) > max_prompts:
        cached_results = cached_results[:max_prompts]

    print(f"Loaded {len(cached_results)} cached results. Re-scoring with {rm_name}...")

    acc_rm = {N: [] for N in n_list}

    for res in tqdm(cached_results, desc="Scoring with new RM"):
        prompt_text = res.get('prompt_text')
        candidates = res['candidates']
        correct_flags = res['correct']

        rm_scores = rm.score_batch([(prompt_text, c) for c in candidates])

        for N in n_list:
            top_rm = int(np.argmax(rm_scores[:N]))
            acc_rm[N].append(1 if correct_flags[top_rm] else 0)

    rm_accuracies = [float(np.mean(acc_rm[N])) if acc_rm[N] else 0.0 for N in sorted(n_list)]
    df_existing[rm_name] = rm_accuracies
    df_existing.to_csv(csv_path, index=False)

    plot_bon_curves_generic(csv_path, out_dir, "Best-of-N Curves (Math)", "Accuracy", "bon_curve.png")
    print(f"Updated {csv_path} and plot bon_curve.png")


# -------------------------
# BoN eval (IFEval mode)
# -------------------------
def bon_eval_ifeval(
    data: List[Dict[str, Any]],
    actor: Actor,
    rm: RewardScorer,
    rm_repo: str,
    n_list: List[int],
    max_prompts: Optional[int],
    out_dir: str,
    seed: int = 0,
    save_per_prompt: bool = True,
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_new_tokens: int = 512,
):
    """
    IFEval mode:
    - prompt is doc["prompt"] exactly (doc_to_text: prompt)
    - generation uses sampling with configurable temperature, top_p, max_new_tokens
    - scoring matches harness via lm_eval.tasks.ifeval.utils.process_results
    """
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    ifeval_utils = import_ifeval_utils_exact()

    max_n = max(n_list)
    rm_name = get_rm_name(rm_repo)

    # Separate output files so you can run math and ifeval without collisions
    csv_path = os.path.join(out_dir, 'metrics_ifeval.csv')
    raw_path = os.path.join(out_dir, 'per_prompt_results_ifeval.jsonl')

    is_incremental = os.path.exists(csv_path) and os.path.exists(raw_path)
    if is_incremental:
        print(f"Found existing IFEval results. Running incremental evaluation for RM: {rm_name}")
        return bon_eval_ifeval_incremental(data, actor, rm, rm_name, n_list, max_prompts, out_dir, seed)

    print(f"First IFEval run. Generating candidates and computing baselines + RM: {rm_name}")
    fw = open(raw_path, 'w', encoding='utf-8') if save_per_prompt else None

    # For each N, we track prompt-level and instruction-level outcomes separately (strict/loose).
    # prompt-level: list of 0/1 per prompt
    pl_strict_rm = {N: [] for N in n_list}
    pl_loose_rm = {N: [] for N in n_list}
    pl_strict_oracle = {N: [] for N in n_list}
    pl_loose_oracle = {N: [] for N in n_list}
    pl_strict_random = {N: [] for N in n_list}
    pl_loose_random = {N: [] for N in n_list}
    pl_strict_logp = {N: [] for N in n_list}
    pl_loose_logp = {N: [] for N in n_list}

    # inst-level: accumulate counts (followed, total)
    il_strict_rm = {N: [0, 0] for N in n_list}
    il_loose_rm = {N: [0, 0] for N in n_list}
    il_strict_oracle = {N: [0, 0] for N in n_list}
    il_loose_oracle = {N: [0, 0] for N in n_list}
    il_strict_random = {N: [0, 0] for N in n_list}
    il_loose_random = {N: [0, 0] for N in n_list}
    il_strict_logp = {N: [0, 0] for N in n_list}
    il_loose_logp = {N: [0, 0] for N in n_list}

    iterable = data[:max_prompts] if max_prompts else data

    for idx, doc in enumerate(tqdm(iterable, desc="IFEval Prompts")):
        prompt = doc["prompt"]  # exact harness doc_to_text: prompt

        # Generation with sampling using configurable parameters
        cand_texts = actor.generate_n(
            prompt,
            max_n,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=actor.cfg.repetition_penalty,
        )

        # Score candidates with harness IFEval logic
        cand_scores = []
        for c in cand_texts:
            r = ifeval_process_results(ifeval_utils, doc, c)
            # Expected keys per harness task yaml
            # - prompt_level_strict_acc: 0/1
            # - prompt_level_loose_acc: 0/1
            # - inst_level_strict_acc: list[0/1] (or similar)
            # - inst_level_loose_acc: list[0/1]
            cand_scores.append(r)

        # RM + LogP scoring (for BoN selection)
        rm_scores = rm.score_batch([(prompt, c) for c in cand_texts])
        logp_scores = actor.avg_logprob(prompt, cand_texts)

        if fw is not None:
            fw.write(json.dumps({
                "i": idx,
                "key": doc.get("key"),
                "prompt_text": prompt,
                "instruction_id_list": doc.get("instruction_id_list"),
                "kwargs": doc.get("kwargs"),
                "candidates": cand_texts,
                "rm_scores": rm_scores,
                "logp_scores": logp_scores,
                "ifeval_scores": cand_scores,  # per-candidate dict
            }) + "\n")

        for N in n_list:
            # Candidate indices under each chooser
            top_rm = int(np.argmax(rm_scores[:N]))
            top_lp = int(np.argmax(logp_scores[:N]))
            rand_pick = random.randrange(N)

            # ORACLE: prompt-level = any candidate hits 1
            pl_strict_vals = [int(cand_scores[j].get("prompt_level_strict_acc", 0)) for j in range(N)]
            pl_loose_vals = [int(cand_scores[j].get("prompt_level_loose_acc", 0)) for j in range(N)]
            pl_strict_oracle[N].append(1 if any(pl_strict_vals) else 0)
            pl_loose_oracle[N].append(1 if any(pl_loose_vals) else 0)

            # ORACLE for inst-level: pick candidate with max fraction satisfied
            def inst_frac(j: int, key: str) -> float:
                f, t = inst_flags_to_counts(cand_scores[j].get(key))
                return (f / t) if t > 0 else 0.0

            best_inst_strict = max(range(N), key=lambda j: inst_frac(j, "inst_level_strict_acc"))
            best_inst_loose = max(range(N), key=lambda j: inst_frac(j, "inst_level_loose_acc"))

            # RM picks (prompt-level)
            pl_strict_rm[N].append(int(cand_scores[top_rm].get("prompt_level_strict_acc", 0)))
            pl_loose_rm[N].append(int(cand_scores[top_rm].get("prompt_level_loose_acc", 0)))

            # Random picks (prompt-level)
            pl_strict_random[N].append(int(cand_scores[rand_pick].get("prompt_level_strict_acc", 0)))
            pl_loose_random[N].append(int(cand_scores[rand_pick].get("prompt_level_loose_acc", 0)))

            # LogP picks (prompt-level)
            pl_strict_logp[N].append(int(cand_scores[top_lp].get("prompt_level_strict_acc", 0)))
            pl_loose_logp[N].append(int(cand_scores[top_lp].get("prompt_level_loose_acc", 0)))

            # Instruction-level accumulation for RM pick
            f, t = inst_flags_to_counts(cand_scores[top_rm].get("inst_level_strict_acc"))
            il_strict_rm[N][0] += f; il_strict_rm[N][1] += t
            f, t = inst_flags_to_counts(cand_scores[top_rm].get("inst_level_loose_acc"))
            il_loose_rm[N][0] += f; il_loose_rm[N][1] += t

            # Instruction-level accumulation for Random pick
            f, t = inst_flags_to_counts(cand_scores[rand_pick].get("inst_level_strict_acc"))
            il_strict_random[N][0] += f; il_strict_random[N][1] += t
            f, t = inst_flags_to_counts(cand_scores[rand_pick].get("inst_level_loose_acc"))
            il_loose_random[N][0] += f; il_loose_random[N][1] += t

            # Instruction-level accumulation for LogP pick
            f, t = inst_flags_to_counts(cand_scores[top_lp].get("inst_level_strict_acc"))
            il_strict_logp[N][0] += f; il_strict_logp[N][1] += t
            f, t = inst_flags_to_counts(cand_scores[top_lp].get("inst_level_loose_acc"))
            il_loose_logp[N][0] += f; il_loose_logp[N][1] += t

            # Instruction-level accumulation for Oracle pick (best fraction)
            f, t = inst_flags_to_counts(cand_scores[best_inst_strict].get("inst_level_strict_acc"))
            il_strict_oracle[N][0] += f; il_strict_oracle[N][1] += t
            f, t = inst_flags_to_counts(cand_scores[best_inst_loose].get("inst_level_loose_acc"))
            il_loose_oracle[N][0] += f; il_loose_oracle[N][1] += t

            # Prompt-level for Oracle already tracked above (pl_*_oracle)

    if fw is not None:
        fw.close()

    # Build metrics_ifeval.csv with 4 metrics × methods
    Ns = sorted(n_list)

    def safe_mean(xs: List[int]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def safe_inst_acc(pair: List[int]) -> float:
        f, t = pair
        return float(f) / float(t) if t > 0 else 0.0

    df = pd.DataFrame({
        "N": Ns,

        # Prompt-level strict
        f"{rm_name}:prompt_level_strict_acc": [safe_mean(pl_strict_rm[N]) for N in Ns],
        "Oracle:prompt_level_strict_acc": [safe_mean(pl_strict_oracle[N]) for N in Ns],
        "Random:prompt_level_strict_acc": [safe_mean(pl_strict_random[N]) for N in Ns],
        "LogP:prompt_level_strict_acc": [safe_mean(pl_strict_logp[N]) for N in Ns],

        # Prompt-level loose
        f"{rm_name}:prompt_level_loose_acc": [safe_mean(pl_loose_rm[N]) for N in Ns],
        "Oracle:prompt_level_loose_acc": [safe_mean(pl_loose_oracle[N]) for N in Ns],
        "Random:prompt_level_loose_acc": [safe_mean(pl_loose_random[N]) for N in Ns],
        "LogP:prompt_level_loose_acc": [safe_mean(pl_loose_logp[N]) for N in Ns],

        # Inst-level strict
        f"{rm_name}:inst_level_strict_acc": [safe_inst_acc(il_strict_rm[N]) for N in Ns],
        "Oracle:inst_level_strict_acc": [safe_inst_acc(il_strict_oracle[N]) for N in Ns],
        "Random:inst_level_strict_acc": [safe_inst_acc(il_strict_random[N]) for N in Ns],
        "LogP:inst_level_strict_acc": [safe_inst_acc(il_strict_logp[N]) for N in Ns],

        # Inst-level loose
        f"{rm_name}:inst_level_loose_acc": [safe_inst_acc(il_loose_rm[N]) for N in Ns],
        "Oracle:inst_level_loose_acc": [safe_inst_acc(il_loose_oracle[N]) for N in Ns],
        "Random:inst_level_loose_acc": [safe_inst_acc(il_loose_random[N]) for N in Ns],
        "LogP:inst_level_loose_acc": [safe_inst_acc(il_loose_logp[N]) for N in Ns],
    })

    df.to_csv(csv_path, index=False)

    # Produce 4 plots (one per harness metric)
    plot_cols = [
        ("prompt_level_strict_acc", "IFEval BoN (prompt_level_strict_acc)", "Accuracy", "bon_curve_ifeval_prompt_level_strict_acc.png"),
        ("prompt_level_loose_acc", "IFEval BoN (prompt_level_loose_acc)", "Accuracy", "bon_curve_ifeval_prompt_level_loose_acc.png"),
        ("inst_level_strict_acc", "IFEval BoN (inst_level_strict_acc)", "Accuracy", "bon_curve_ifeval_inst_level_strict_acc.png"),
        ("inst_level_loose_acc", "IFEval BoN (inst_level_loose_acc)", "Accuracy", "bon_curve_ifeval_inst_level_loose_acc.png"),
    ]

    for suffix, title, ylab, png in plot_cols:
        # create a temporary CSV view with columns relevant to this metric
        sub_cols = ["N"] + [c for c in df.columns if c.endswith(suffix)]
        tmp = df[sub_cols].copy()
        tmp_path = os.path.join(out_dir, f"_tmp_{suffix}.csv")
        tmp.to_csv(tmp_path, index=False)
        plot_bon_curves_generic(tmp_path, out_dir, title, ylab, png)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    print(f"Saved IFEval metrics to {csv_path}")
    print(f"Saved IFEval per-prompt results to {raw_path}" if save_per_prompt else "IFEval per-prompt results not saved")


def bon_eval_ifeval_incremental(
    data: List[Dict[str, Any]],
    actor: Actor,
    rm: RewardScorer,
    rm_name: str,
    n_list: List[int],
    max_prompts: Optional[int],
    out_dir: str,
    seed: int = 0,
):
    """
    Incremental IFEval:
    Load cached candidates + harness scores from per_prompt_results_ifeval.jsonl,
    rescore only RM, and overwrite RM columns in metrics_ifeval.csv.
    """
    set_seed(seed)
    csv_path = os.path.join(out_dir, 'metrics_ifeval.csv')
    raw_path = os.path.join(out_dir, 'per_prompt_results_ifeval.jsonl')

    df_existing = pd.read_csv(csv_path)

    # Identify which columns correspond to this RM (from previous run)
    rm_cols = [c for c in df_existing.columns if c.startswith(f"{rm_name}:")]
    if not rm_cols:
        print(f"No existing RM columns found for '{rm_name}' in {csv_path}. Will append new columns.")

    cached = []
    with open(raw_path, 'r', encoding='utf-8') as f:
        for line in f:
            cached.append(json.loads(line))

    if max_prompts is not None and len(cached) > max_prompts:
        cached = cached[:max_prompts]

    # We recompute RM@N selections but reuse cached harness ifeval_scores per candidate.
    Ns = sorted(n_list)

    pl_strict_rm = {N: [] for N in Ns}
    pl_loose_rm = {N: [] for N in Ns}
    il_strict_rm = {N: [0, 0] for N in Ns}
    il_loose_rm = {N: [0, 0] for N in Ns}

    for ex in tqdm(cached, desc="IFEval incremental scoring"):
        prompt = ex["prompt_text"]
        candidates = ex["candidates"]
        ifeval_scores = ex["ifeval_scores"]

        rm_scores = rm.score_batch([(prompt, c) for c in candidates])

        for N in Ns:
            top_rm = int(np.argmax(rm_scores[:N]))
            pl_strict_rm[N].append(int(ifeval_scores[top_rm].get("prompt_level_strict_acc", 0)))
            pl_loose_rm[N].append(int(ifeval_scores[top_rm].get("prompt_level_loose_acc", 0)))

            f, t = inst_flags_to_counts(ifeval_scores[top_rm].get("inst_level_strict_acc"))
            il_strict_rm[N][0] += f; il_strict_rm[N][1] += t

            f, t = inst_flags_to_counts(ifeval_scores[top_rm].get("inst_level_loose_acc"))
            il_loose_rm[N][0] += f; il_loose_rm[N][1] += t

    def safe_mean(xs: List[int]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    def safe_inst_acc(pair: List[int]) -> float:
        f, t = pair
        return float(f) / float(t) if t > 0 else 0.0

    # Overwrite or add RM columns
    df_existing[f"{rm_name}:prompt_level_strict_acc"] = [safe_mean(pl_strict_rm[N]) for N in Ns]
    df_existing[f"{rm_name}:prompt_level_loose_acc"] = [safe_mean(pl_loose_rm[N]) for N in Ns]
    df_existing[f"{rm_name}:inst_level_strict_acc"] = [safe_inst_acc(il_strict_rm[N]) for N in Ns]
    df_existing[f"{rm_name}:inst_level_loose_acc"] = [safe_inst_acc(il_loose_rm[N]) for N in Ns]

    df_existing.to_csv(csv_path, index=False)

    # Replot 4 metrics
    plot_cols = [
        ("prompt_level_strict_acc", "IFEval BoN (prompt_level_strict_acc)", "Accuracy", "bon_curve_ifeval_prompt_level_strict_acc.png"),
        ("prompt_level_loose_acc", "IFEval BoN (prompt_level_loose_acc)", "Accuracy", "bon_curve_ifeval_prompt_level_loose_acc.png"),
        ("inst_level_strict_acc", "IFEval BoN (inst_level_strict_acc)", "Accuracy", "bon_curve_ifeval_inst_level_strict_acc.png"),
        ("inst_level_loose_acc", "IFEval BoN (inst_level_loose_acc)", "Accuracy", "bon_curve_ifeval_inst_level_loose_acc.png"),
    ]
    df = df_existing
    for suffix, title, ylab, png in plot_cols:
        sub_cols = ["N"] + [c for c in df.columns if c.endswith(suffix)]
        tmp = df[sub_cols].copy()
        tmp_path = os.path.join(out_dir, f"_tmp_{suffix}.csv")
        tmp.to_csv(tmp_path, index=False)
        plot_bon_curves_generic(tmp_path, out_dir, title, ylab, png)
        try:
            os.remove(tmp_path)
        except Exception:
            pass

    print(f"Updated {csv_path} with RM {rm_name} and refreshed plots.")


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--actor_repo', type=str, required=False, default='meta-llama/Llama-3.2-3B-Instruct',
                   help='HF repo for the actor')
    p.add_argument('--rm_repo', type=str, required=True,
                   help='HF repo for reward model(s). Can be comma-separated list for multiple RMs.')
    p.add_argument('--rm_logits_index', type=int, default=0,
                   help='If RM has multi-logit head, which index to use')

    # Math dataset args (ignored if --ifeval)
    p.add_argument('--dataset_path', type=str, default='',
                   help='HF dataset repo or local JSONL; defaults to hendrycks/competition_math')
    p.add_argument('--split', type=str, default='test', choices=['train', 'test'],
                   help='Dataset split for math datasets')
    p.add_argument('--problem_column', type=str, default='problem',
                   help='Column name for problem')
    p.add_argument('--answer_column', type=str, default='solution',
                   help='Column name for answer/solution')

    # IFEval switch
    p.add_argument('--ifeval', action='store_true',
                   help='Run IFEval mode: google/IFEval train split, exact lm-eval-harness prompt/gen/scoring')

    p.add_argument('--max_prompts', type=int, default=None,
                   help='Number of prompts to evaluate (defaults to all)')
    p.add_argument('--n_list', type=str, default='1,2,4,8,16,32',
                   help='Comma-separated list of N values')
    p.add_argument('--out_dir', type=str, default='bon_out')
    p.add_argument('--seed', type=int, default=123)

    # Actor sampling params (used in both math mode and ifeval generation)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.95)
    p.add_argument('--max_new_tokens', type=int, default=512)

    p.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
    p.add_argument('--no_save_per_prompt', action='store_true',
                   help='Skip saving per-prompt results to JSONL file')
    p.add_argument('--prompt_format', type=str, default='chat', choices=['chat', 'plain'],
                   help="Prompt format for math mode only.")

    args = p.parse_args()
    set_seed(args.seed)

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
    if args.ifeval:
        data = load_ifeval_data(seed=args.seed, limit=args.max_prompts, split="train")
        print(f"Loaded IFEval: {len(data)} examples (google/IFEval, split=train)")
    else:
        data = load_test_data(
            dataset_path=args.dataset_path or None,
            seed=args.seed,
            limit=500,
            split=args.split,
            problem_column=args.problem_column,
            answer_column=args.answer_column
        )
        if args.max_prompts is not None:
            data = data[:args.max_prompts]
        print(f"Loaded math data: {len(data)} examples")

    actor = Actor(actor_cfg)
    n_list = [int(x) for x in args.n_list.split(',') if x.strip()]

    for idx, rm_repo in enumerate(rm_repos):
        print(f"\n{'='*80}")
        print(f"Processing RM {idx+1}/{len(rm_repos)}: {rm_repo}")
        print(f"{'='*80}\n")

        rm_cfg = RMConfig(repo=rm_repo, logits_index=args.rm_logits_index, dtype=args.dtype)
        rm = RewardScorer(rm_cfg)

        if args.ifeval:
            bon_eval_ifeval(
                data=data,
                actor=actor,
                rm=rm,
                rm_repo=rm_repo,
                n_list=n_list,
                max_prompts=args.max_prompts,
                out_dir=args.out_dir,
                seed=args.seed,
                save_per_prompt=not args.no_save_per_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
            )
        else:
            bon_eval_math(
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

        del rm
        torch.cuda.empty_cache()

    print(f"\n{'='*80}")
    print(f"Completed evaluation of all {len(rm_repos)} RMs!")
    print(f"Results saved to {args.out_dir}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
