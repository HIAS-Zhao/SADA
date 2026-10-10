# -*- coding: utf-8 -*-
"""
Metrics utilities for caption and VQA evaluation.
Supports: BLEU, ROUGE-L, METEOR, CIDEr, accuracy, GPT scoring.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass
class GPTScoringConfig:
    """
    Parameters
    ----------
    model : str
        OpenAI model name used as an LLM judge (e.g., "gpt-4", "gpt-4.1", "gpt-5.2-2025-12-11").
    temperature : float
        Lower is more deterministic. CLAIR logic typically uses greedy decoding (0.0).
    max_output_tokens : int
        Max tokens for the judge's output. Keep it small because we only need JSON.
    retries : int
        Retry count for transient API failures.
    retry_sleep_sec : float
        Base sleep time between retries (exponential backoff).
    """
    model: str = "gpt-5.2-2025-12-11"
    temperature: float = 0.01
    max_output_tokens: int = 128
    retries: int = 2
    retry_sleep_sec: float = 1.0



def accuracy(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """
    Exact-match accuracy.

    Parameters
    ----------
    y_true : List[str]
        Ground-truth labels (already normalized if needed).
    y_pred : List[str]
        Predicted labels.

    Returns
    -------
    Dict[str, Any]
        {"n": int, "correct": int, "accuracy": float}
    """
    assert len(y_true) == len(y_pred)
    n = len(y_true)
    correct = sum(int(a == b) for a, b in zip(y_true, y_pred))
    return {"n": n, "correct": correct, "accuracy": (correct / n if n else 0.0)}


def tokenize(text: str) -> List[str]:
    """Lowercase, alphanumeric tokenizer."""
    text = (text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()


def bleu_scores(pred: str, refs: List[str]) -> Dict[str, float]:
    """BLEU-1..4 scores using sacrebleu."""
    try:
        import sacrebleu
        score = sacrebleu.sentence_bleu(pred, refs, smooth_method="exp")
        prec = score.precisions
        return {
            "bleu1": float(prec[0]),
            "bleu2": float(prec[1]),
            "bleu3": float(prec[2]),
            "bleu4": float(prec[3]),
        }
    except Exception as e:
        warnings.warn(
            f"sacrebleu unavailable ({e}); falling back to unigram precision. "
            "Install 'sacrebleu' for BLEU-1..4.",
            UserWarning,
        )
        # Fallback: unigram precision against the first reference.
        hyp = tokenize(pred)
        ref = tokenize(refs[0]) if refs else []
        if not hyp:
            return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}
        ref_set = set(ref)
        inter = sum(1 for t in hyp if t in ref_set)
        p1 = inter / len(hyp)
        return {"bleu1": float(p1), "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}





def rouge_l(pred: str, refs: List[str]) -> float:
    """ROUGE-L F1 score (best over references), returned as percentage 0-100."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        best = 0.0
        for ref in refs:
            score = scorer.score(ref, pred)
            f1 = score["rougeL"].fmeasure
            best = max(best, f1)
        return float(best) * 100.0
    except Exception as e:
        warnings.warn(
            f"rouge_score unavailable ({e}); using simple LCS fallback. "
            "Install 'rouge-score' for ROUGE-L.",
            UserWarning,
        )
        def lcs_len(a: List[str], b: List[str]) -> int:
            """Longest common subsequence length."""
            dp = [0] * (len(b) + 1)
            for i in range(1, len(a) + 1):
                prev = 0
                for j in range(1, len(b) + 1):
                    tmp = dp[j]
                    if a[i - 1] == b[j - 1]:
                        dp[j] = prev + 1
                    else:
                        dp[j] = max(dp[j], dp[j - 1])
                    prev = tmp
            return dp[-1]

        hyp = tokenize(pred)
        if not hyp:
            return 0.0
        best = 0.0
        for r in refs:
            ref = tokenize(r)
            if not ref:
                continue
            lcs = lcs_len(ref, hyp)
            if lcs == 0:
                continue
            p = lcs / len(hyp)
            rr = lcs / len(ref)
            f1 = (2 * p * rr / (p + rr)) if (p + rr) else 0.0
            best = max(best, f1)
        return float(best) * 100.0


def meteor(pred: str, refs: List[str]) -> float:
    """METEOR score using NLTK, returned as percentage 0-100."""
    from nltk.translate.meteor_score import meteor_score
    pred_toks = tokenize(pred)
    refs_toks = [tokenize(ref) for ref in refs]
    return float(meteor_score(refs_toks, pred_toks)) * 100.0


def cider_simple(pred: str, refs: List[str], n: int = 4) -> float:
    """CIDEr score using pycocoevalcap (or TF n-gram fallback)."""
    try:
        from pycocoevalcap.cider.cider import Cider
        # CIDEr expects format: {image_id: [hyp_caption]} and {image_id: [ref1, ref2, ...]}
        cider_scorer = Cider(n=n)
        # Use a dummy image_id since we're scoring one caption pair
        hypo = {0: [pred]}
        refs_dict = {0: refs}
        score, scores = cider_scorer.compute_score(refs_dict, hypo)
        return float(score)
    except Exception as e:
        warnings.warn(
            f"pycocoevalcap unavailable ({e}); using lightweight CIDEr approximation. "
            "Install 'pycocoevalcap' for official COCO CIDEr.",
            UserWarning,
        )
        # Lightweight fallback: TF n-gram cosine similarity
        hyp = tokenize(pred)
        ref_toks = [tokenize(r) for r in refs]
        if not hyp or not ref_toks:
            return 0.0

        def tf(tokens: List[str], k: int) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for i in range(len(tokens) - k + 1):
                ng = " ".join(tokens[i:i + k])
                out[ng] = out.get(ng, 0.0) + 1.0
            total = sum(out.values()) or 1.0
            for key in list(out.keys()):
                out[key] /= total
            return out

        def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
            if not a or not b:
                return 0.0
            dot = 0.0
            for key, v in a.items():
                dot += v * b.get(key, 0.0)
            na = sum(v * v for v in a.values()) ** 0.5
            nb = sum(v * v for v in b.values()) ** 0.5
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

        score = 0.0
        for k in range(1, n + 1):
            h = tf(hyp, k)
            best = 0.0
            for rt in ref_toks:
                best = max(best, cosine(h, tf(rt, k)))
            score += best
        return float(score * (10.0 / n))


# ---------------- OpenAI helpers ----------------
def _extract_json_object(text: str) -> Dict[str, Any]:
    """Extract JSON object from text."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Cannot find JSON in: {text[:200]}")
    return json.loads(m.group(0))


def _openai_response_text(prompt: str, config: GPTScoringConfig, api_key: Optional[str]) -> str:
    """Call OpenAI Responses API and return text."""
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("Missing dependency: openai. Install via `pip install openai`.") from e

    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (and api_key not provided).")

    # IMPORTANT: route requests to your gateway
    base_url = os.getenv("OPENAI_BASE_URL", "https://aigc.x-see.cn").rstrip("/")
    client = OpenAI(api_key=key, base_url=base_url)

    last_err: Optional[Exception] = None
    for attempt in range(int(config.retries) + 1):
        try:
            resp = client.responses.create(
                model=config.model,
                input=prompt,
                temperature=float(config.temperature),
                max_output_tokens=int(config.max_output_tokens),
            )

            # Preferred: SDK convenience field
            text = (getattr(resp, "output_text", None) or "").strip()
            if text:
                return text

            # Fallback: assemble from resp.output[].content[]
            chunks = []
            for item in (getattr(resp, "output", None) or []):
                for part in (getattr(item, "content", None) or []):
                    if getattr(part, "type", None) in ("output_text", "text"):
                        chunks.append(getattr(part, "text", "") or "")
            return "".join(chunks).strip()

        except Exception as e:
            last_err = e
            if attempt >= int(config.retries):
                break
            time.sleep(float(config.retry_sleep_sec) * (2 ** attempt))

    raise RuntimeError(f"OpenAI call failed after retries: {last_err}") from last_err


def _stable_prompt_hash(prompt: str) -> str:
    """Stable SHA256 hash for caching."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


# ---------------- Task10: VQA semantic match ----------------
def vqa_gpt_match(
    question: str,
    ground_truth: str,
    predicted: str,
    config: Optional[GPTScoringConfig] = None,
    api_key: Optional[str] = None,
) -> int:
    """GPT semantic match for VQA (returns 1 or 0)."""
    cfg = config or GPTScoringConfig(model="gpt-5.2-2025-12-11", max_output_tokens=16, temperature=0.01)
    prompt = (
        f"Question: {question}\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Predicted Answer: {predicted}\n"
        "Does the predicted answer match the ground truth? "
        "Answer 1 for match and 0 for not match. "
        "Use semantic meaning not exact match. "
        "Synonyms are also treated as a match.\n"
        "Return ONLY JSON: {\"match\": 1 or 0}."
    )
    out = _openai_response_text(prompt, cfg, api_key)
    obj = _extract_json_object(out)
    try:
        return 1 if int(obj.get("match", 0)) == 1 else 0
    except Exception:
        return 0


# ---------------- Task2: Disaster caption GPT scoring ----------------
def gpt_score_disaster_caption(
    ground_truth: str,
    predicted: str,
    config: Optional[GPTScoringConfig] = None,
    api_key: Optional[str] = None,
) -> Dict[str, float]:
    """Score disaster caption with GPT (DAP/DDR/FC: 0-5 scale)."""
    cfg = config or GPTScoringConfig(model="gpt-5.2-2025-12-11", max_output_tokens=256, temperature=0.01)
    prompt = (
        "You are grading a long disaster caption.\n"
        "Score each dimension on a 0-5 scale (0=very poor, 5=excellent).\n"
        "- DAP (damage assessment precision): Are mentioned damages correct and not hallucinated?\n"
        "- DDR (damage detail recall): Are important damage details from GT covered?\n"
        "- FC (factual correctness): Overall factual consistency with the GT.\n\n"
        f"Ground Truth:\n{ground_truth}\n\n"
        f"Predicted:\n{predicted}\n\n"
        "Return ONLY JSON: {\"DAP\": <0-5>, \"DDR\": <0-5>, \"FC\": <0-5>}."
    )
    out = _openai_response_text(prompt, cfg, api_key)
    obj = _extract_json_object(out)

    def _clip_0_5(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, min(5.0, v))

    dap = _clip_0_5(obj.get("DAP", 0.0))
    ddr = _clip_0_5(obj.get("DDR", 0.0))
    fc = _clip_0_5(obj.get("FC", 0.0))
    return {"DAP": dap, "DDR": ddr, "FC": fc, "avg": (dap + ddr + fc) / 3.0}


# ---------------- Task9: CLAIR (official logic) ----------------
def _build_clair_official_prompt(candidates: List[str], references: List[str]) -> str:
    # Format as bullet lists to emphasize "sets".
    cand_block = "\n".join(f"- {c}" for c in candidates) if candidates else "- "
    ref_block = "\n".join(f"- {r}" for r in references) if references else "- "

    return (
        "You are trying to tell if a candidate set of captions is describing the same image as a reference set of captions.\n"
        f"Candidate set:\n{cand_block}\n"
        f"Reference set:\n{ref_block}\n"
        "On a precise scale from 0 to 100, how likely is it that the candidate set is describing the same image as the reference set?\n"
        "(JSON format, with a key \"score\", value between 0 and 100, no other words.)"
    )


def _clair_score_0_100_from_prompt(prompt: str, cfg: GPTScoringConfig, api_key: Optional[str]) -> float:
    out = _openai_response_text(prompt, cfg, api_key)
    obj = _extract_json_object(out)
    try:
        score = float(obj.get("score", 0.0))
    except Exception:
        score = 0.0
    return max(0.0, min(100.0, score))

def clair_official_score_0_100(
    candidates: List[str],
    references: List[str],
    config: Optional[GPTScoringConfig] = None,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """CLAIR: candidate vs reference set similarity (0-100)."""
    cfg = config or GPTScoringConfig(model="gpt-5.2-2025-12-11", max_output_tokens=64, temperature=0.01)

    # Format as bullet lists to emphasize "sets".
    cfg = config or GPTScoringConfig(model="gpt-5.2-2025-12-11", max_output_tokens=64, temperature=0.01)

    prompt = _build_clair_official_prompt(candidates, references)
    score = _clair_score_0_100_from_prompt(prompt, cfg, api_key)

    return {"score": score, "prompt": prompt}



def task01_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Task01: accuracy."""
    return accuracy(y_true, y_pred)


def task03_08_metrics(y_true: List[str], y_pred: List[str]) -> Dict[str, Any]:
    """Tasks03-08: accuracy."""
    return accuracy(y_true, y_pred)


def task02_metrics(
    gt_pred_pairs: Sequence[Tuple[str, str]],
    *,
    api_key: Optional[str] = None,
    gpt_model: str = "gpt-5.2-2025-12-11",
    gpt_cache_get=None,
    gpt_cache_set=None,
) -> Dict[str, Any]:
    """
    Task02: compute average DAP/DDR/FC over dataset.

    Parameters
    ----------
    gt_pred_pairs : Sequence[(gt, pred)]
        Each element is (ground_truth_caption, predicted_caption).
    api_key : Optional[str]
        OpenAI API key.
    gpt_model : str
        Judge model name.
    gpt_cache_get / gpt_cache_set : callable
        Optional cache hooks:
          - get(key)->Any|None
          - set(key,value)->None

    Returns
    -------
    Dict[str, Any]
        {"n": int, "DAP": float, "DDR": float, "FC": float, "avg": float}
    """
    cfg = GPTScoringConfig(model=gpt_model, max_output_tokens=256, temperature=0.01)

    totals = {"DAP": 0.0, "DDR": 0.0, "FC": 0.0, "avg": 0.0}
    n = 0

    for gt, pred in gt_pred_pairs:
        # Build a stable cache key based on the exact prompt content.
        prompt = (
            "You are grading a long disaster caption.\n"
            "Score each dimension on a 0-5 scale (0=very poor, 5=excellent).\n"
            "- DAP (damage assessment precision): Are mentioned damages correct and not hallucinated?\n"
            "- DDR (damage detail recall): Are important damage details from GT covered?\n"
            "- FC (factual correctness): Overall factual consistency with the GT.\n\n"
            f"Ground Truth:\n{gt}\n\n"
            f"Predicted:\n{pred}\n\n"
            "Return ONLY JSON: {\"DAP\": <0-5>, \"DDR\": <0-5>, \"FC\": <0-5>}."
        )
        cache_key = f"{gpt_model}::task02::{_stable_prompt_hash(prompt)}" if (gpt_cache_get and gpt_cache_set) else None

        cached = gpt_cache_get(cache_key) if cache_key else None
        if cached is None:
            scores = gpt_score_disaster_caption(gt, pred, config=cfg, api_key=api_key)
            if cache_key:
                gpt_cache_set(cache_key, scores)
        else:
            scores = cached

        totals["DAP"] += float(scores.get("DAP", 0.0))
        totals["DDR"] += float(scores.get("DDR", 0.0))
        totals["FC"] += float(scores.get("FC", 0.0))
        totals["avg"] += float(scores.get("avg", 0.0))
        n += 1

    if n == 0:
        return {"n": 0}

    out = {k: v / n for k, v in totals.items()}
    out["n"] = n
    return out


def task09_metrics(
    gt_pred_pairs: Sequence[Tuple[str, str]],
    *,
    api_key: Optional[str] = None,
    gpt_model: str = "gpt-5.2-2025-12-11",
    gpt_cache_get=None,
    gpt_cache_set=None,
) -> Dict[str, Any]:
    """
    Task09: classic metrics + CLAIR (official logic).

    Parameters
    ----------
    gt_pred_pairs : Sequence[(gt, pred)]
        Each element is (ground_truth_caption, predicted_caption).
        If your dataset provides only one GT caption per image (as in your JSON),
        just use that single caption as gt.
    api_key : Optional[str]
        OpenAI API key.
    gpt_model : str
        Judge model name for CLAIR.
    gpt_cache_get / gpt_cache_set : callable
        Optional cache hooks:
          - get(key)->Any|None
          - set(key,value)->None

    Returns
    -------
    Dict[str, Any]
        Dataset-level averages:
          bleu1..4, rougeL, meteor, cider, pred_len,
          clair_0_100, clair_0_1,
          (optional) clair_reason_sample
    """
    totals = {
        "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
        "rougeL": 0.0, "meteor": 0.0, "cider": 0.0,
        "pred_len": 0.0,
        "clair_0_100": 0.0,
    }
    n = 0

    cfg = GPTScoringConfig(model=gpt_model, max_output_tokens=64, temperature=0.01)

    for gt, pred in gt_pred_pairs:
        refs = [gt]

        b = bleu_scores(pred, refs)
        r = rouge_l(pred, refs)
        m = meteor(pred, refs)
        c = cider_simple(pred, refs)
        length = len(tokenize(pred))

        totals["bleu1"] += b["bleu1"]
        totals["bleu2"] += b["bleu2"]
        totals["bleu3"] += b["bleu3"]
        totals["bleu4"] += b["bleu4"]
        totals["rougeL"] += r
        totals["meteor"] += m
        totals["cider"] += c
        totals["pred_len"] += float(length)

        # --- CLAIR official logic ---
        # Candidate set vs Reference set (even if each has only one caption).
        candidates = [pred]
        references = [gt]

        # Build the prompt (shared with clair_official_score_0_100), so cache matches.
        clair_prompt = _build_clair_official_prompt(candidates, references)

        cache_key = (
            f"{gpt_model}::task09_clair::{_stable_prompt_hash(clair_prompt)}"
            if (gpt_cache_get and gpt_cache_set)
            else None
        )
        cached = gpt_cache_get(cache_key) if cache_key else None

        if cached is None:
            score100 = _clair_score_0_100_from_prompt(clair_prompt, cfg, api_key)

            if cache_key:
                gpt_cache_set(cache_key, {"score": score100})
        else:
            try:
                score100 = float(cached.get("score", 0.0))
            except Exception:
                score100 = 0.0
            score100 = max(0.0, min(100.0, score100))

        totals["clair_0_100"] += score100

        n += 1

    if n == 0:
        return {"n": 0}

    avg = {k: (v / n) for k, v in totals.items()}
    avg["n"] = n
    avg["clair_0_1"] = avg["clair_0_100"] / 100.0
    return avg


def task09_metrics_from_items(
    items: Sequence[Dict[str, Any]],
    uid_to_pred: Dict[Any, str],
    *,
    uid_key: str = "id",
    gt_key: str = "caption",
    api_key: Optional[str] = None,
    gpt_model: str = "gpt-5.2-2025-12-11",
    gpt_cache_get=None,
    gpt_cache_set=None,
) -> Dict[str, Any]:
    """
    Convenience wrapper for your dataset JSON structure:
      item example has fields {"id":..., "caption":...}.

    Parameters
    ----------
    items : Sequence[Dict[str, Any]]
        Dataset items. Each should have `uid_key` and `gt_key`.
    uid_to_pred : Dict[Any, str]
        Mapping from uid (item[uid_key]) to predicted caption text.
    uid_key : str
        Field name used to look up the uid in each item (default "id").
    gt_key : str
        Field name used to read GT caption in each item (default "caption").
    ... (other args)
        Passed to task09_metrics.

    Returns
    -------
    Dict[str, Any]
        Same as task09_metrics.
    """
    pairs: List[Tuple[str, str]] = []
    for it in items:
        uid = it.get(uid_key)
        if uid not in uid_to_pred:
            continue
        gt = str(it.get(gt_key, ""))
        pred = str(uid_to_pred.get(uid, ""))
        pairs.append((gt, pred))
    return task09_metrics(
        pairs,
        api_key=api_key,
        gpt_model=gpt_model,
        gpt_cache_get=gpt_cache_get,
        gpt_cache_set=gpt_cache_set,
    )


def task10_metrics(
    items: Sequence[Dict[str, Any]],
    uid_to_pred: Dict[str, str],
    *,
    api_key: Optional[str] = None,
    gpt_model: str = "gpt-5.2-2025-12-11",
    gpt_cache_get=None,
    gpt_cache_set=None,
) -> Dict[str, Any]:
    """
    Task10: overall + typewise accuracy using GPT semantic match.

    Parameters
    ----------
    items : Sequence[Dict[str, Any]]
        Adapted samples. Expected each has:
          - item["uid"] (or similar) used to index uid_to_pred
          - item["meta"]["question"]
          - item["meta"]["type"]
          - item["answer"] or item["meta"]["answer"] as ground truth (depends on your adapter)
        This function follows the original design of your code: it expects
        question/type under `item["meta"]`.
    uid_to_pred : Dict[str, str]
        Predicted answers mapping by uid.
    api_key : Optional[str]
        OpenAI API key.
    gpt_model : str
        Judge model name.
    gpt_cache_get / gpt_cache_set : callable
        Optional cache hooks.

    Returns
    -------
    Dict[str, Any]
        {
          "overall": {"n":int,"correct":int,"accuracy":float},
          "by_type": {type: {"n":int,"correct":int,"accuracy":float}}
        }
    """
    cfg = GPTScoringConfig(model=gpt_model, max_output_tokens=16, temperature=0.01)

    total_n = 0
    total_correct = 0
    by_type: Dict[str, Dict[str, int]] = {}

    for it in items:
        uid = str(it.get("uid", it.get("id", "")))
        if uid not in uid_to_pred:
            continue

        meta = it.get("meta", {}) or {}
        q = str(meta.get("question", it.get("question", "")))
        t = str(meta.get("type", it.get("type", "unknown")))

        # Try a few common GT fields without breaking your existing adapter.
        gt = meta.get("ground_truth", None)
        if gt is None:
            gt = it.get("ground_truth", None)
        if gt is None:
            gt = it.get("answer", None)
        if gt is None:
            gt = meta.get("answer", "")

        gt = str(gt)
        pred = str(uid_to_pred[uid])

        prompt = (
            f"Question: {q}\n"
            f"Ground Truth Answer: {gt}\n"
            f"Predicted Answer: {pred}\n"
            "Does the predicted answer match the ground truth? "
            "Answer 1 for match and 0 for not match. "
            "Use semantic meaning not exact match. "
            "Synonyms are also treated as a match.\n"
            "Return ONLY JSON: {\"match\": 1 or 0}."
        )

        cache_key = (
            f"{gpt_model}::task10::{_stable_prompt_hash(prompt)}"
            if (gpt_cache_get and gpt_cache_set)
            else None
        )
        cached = gpt_cache_get(cache_key) if cache_key else None

        if cached is None:
            out = _openai_response_text(prompt, cfg, api_key)
            obj = _extract_json_object(out)
            try:
                match = 1 if int(obj.get("match", 0)) == 1 else 0
            except Exception:
                match = 0
            if cache_key:
                gpt_cache_set(cache_key, {"match": match})
        else:
            try:
                match = 1 if int(cached.get("match", 0)) == 1 else 0
            except Exception:
                match = 0

        total_n += 1
        total_correct += match

        if t not in by_type:
            by_type[t] = {"n": 0, "correct": 0}
        by_type[t]["n"] += 1
        by_type[t]["correct"] += int(match == 1)

    overall = {
        "n": total_n,
        "correct": total_correct,
        "accuracy": (total_correct / total_n if total_n else 0.0),
    }
    by_type_out = {
        t: {
            "n": s["n"],
            "correct": s["correct"],
            "accuracy": (s["correct"] / s["n"] if s["n"] else 0.0),
        }
        for t, s in by_type.items()
    }
    return {"overall": overall, "by_type": by_type_out}
