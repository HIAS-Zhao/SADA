# -*- coding: utf-8 -*-
import argparse
import os
import re
import json
from typing import Any, Dict, List, Tuple

import metrics_common as mc

from io_utils import ensure_dir, load_jsonl, save_json
from data_adapters import load_and_adapt
from gpt_cache import GPTCache
from robustness_utils import (
    SafeDataLoader,
    CheckpointManager,
    EvaluationResult,
    EvaluationLogger,
    generate_evaluation_report,
)


_YES = {"yes", "y", "true", "1"}
_NO = {"no", "n", "false", "0"}


def extract_yes_no(text: str) -> str:
    if not text:
        return ""
    s = text.strip().lower()
    m = re.search(r"\b(yes|no|true|false|y|n)\b", s)
    if m:
        tok = m.group(1)
        if tok in _YES:
            return "Yes"
        if tok in _NO:
            return "No"
    if s.startswith("yes"):
        return "Yes"
    if s.startswith("no"):
        return "No"
    return ""


def extract_option_letter(text: str, valid_letters: str) -> str:
    if not text:
        return ""
    s = text.strip().upper()
    m = re.search(rf"(ANSWER|FINAL|OPTION)\s*[:\-]?\s*([{valid_letters}])\b", s)
    if m:
        return m.group(2)
    m = re.search(rf"\b([{valid_letters}])\b", s)
    if m:
        return m.group(1)
    m = re.search(rf"([{valid_letters}])", s)
    return m.group(1) if m else ""


def extract_option_letters(value: Any, valid_letters: str) -> str:
    if value is None:
        return ""
    allowed = set(valid_letters)
    letters: List[str] = []
    if isinstance(value, (list, tuple, set)):
        chunks = [str(v) for v in value]
    else:
        chunks = [str(value)]
    for chunk in chunks:
        for ch in chunk.upper():
            if ch in allowed and ch not in letters:
                letters.append(ch)
    return ",".join(letters)


def is_binary_yesno_options(options: Dict[str, Any]) -> bool:
    if not isinstance(options, dict) or not options:
        return False
    vals = [str(v).strip().lower() for v in options.values()]
    return set(vals).issubset({"yes", "no", "true", "false"})


def _load_preds_map(pred_path: str, data_loader: SafeDataLoader = None, logger: EvaluationLogger = None) -> Dict[str, str]:
    """安全加载预测文件，支持错误恢复"""
    if data_loader is None:
        rows = load_jsonl(pred_path)
        return {str(r["uid"]): str(r.get("pred", "")) for r in rows if "uid" in r}

    if logger:
        logger.debug(f"Loading predictions from {pred_path}")

    if pred_path.endswith(".jsonl"):
        rows, errors = data_loader.load_jsonl(pred_path)
        preds_map = {row["uid"]: row.get("pred", "") for row in rows if "uid" in row}

        if errors and logger:
            logger.warning(
                f"Loaded {len(rows)} predictions with {len(errors)} errors"
            )
            for err in errors[:5]:
                logger.debug(f"  {err}")
    else:
        data, errors = data_loader.load_json(pred_path)
        if errors:
            if logger:
                logger.error(f"Failed to load {pred_path}: {errors[0]}")
            return {}
        preds_map = data if isinstance(data, dict) else {}

    if logger:
        logger.info(f"Loaded {len(preds_map)} predictions")
    return preds_map


def eval_task01(samples: List[Dict[str, Any]], preds: Dict[str, str]) -> Dict[str, Any]:
    """
    Task01 混合评分逻辑，支持三种格式：
    1. 选择题(文本选项)：比较选项字母（A-G）
    2. Yes/No 问题：比较是否/否
    3. 选择题(图片选项)：比较选项字母（A-D），预测通常是图片路径，需要根据meta中的options映射

    根据样本元数据自动判断类型
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    missing = 0

    for s in samples:
        uid = s["uid"]
        if uid not in preds:
            missing += 1
            continue

        gt_value = s["gt"]
        gt_raw = str(gt_value)
        pred_raw = str(preds[uid])

        meta = s.get("meta", {}) or {}
        options = meta.get("options", {}) or {}
        sample_type = meta.get("type", "")
        type_id = meta.get("type_id")

        # 图片选项格式：options 的值是路径，需要反向映射
        if sample_type == "image_options" and isinstance(options, dict):
            # 尝试找预测的图片对应的字母选项
            pred_letter = None
            for letter, img_path in options.items():
                if pred_raw.endswith(img_path) or pred_raw == img_path:
                    pred_letter = letter
                    break

            # 如果没找到，尝试直接提取字母
            if not pred_letter:
                pred_letter = extract_option_letter(pred_raw, "ABCD") or pred_raw.strip().upper()

            gt = extract_option_letter(gt_raw, "ABCD") or gt_raw.strip().upper()
            pr = pred_letter
        # 文本选项或Yes/No格式
        elif is_binary_yesno_options(options) or gt_raw.strip().lower() in ("yes", "no", "true", "false"):
            # Yes/No 题
            gt = extract_yes_no(gt_raw) or gt_raw.strip()
            pr = extract_yes_no(pred_raw) or pred_raw.strip()
        elif type_id == 3 or sample_type == "multi_mcq_opt_text":
            keys = sorted([str(k).strip().upper() for k in options.keys()]) if isinstance(options, dict) else []
            valid = "".join([k for k in keys if re.fullmatch(r"[A-Z]", k)]) or "ABCDEFG"
            gt = extract_option_letters(gt_value, valid) or gt_raw.strip().upper()
            pr = extract_option_letters(pred_raw, valid) or pred_raw.strip().upper()
        else:
            # 单选题：提取选项字母
            keys = sorted([str(k).strip().upper() for k in options.keys()]) if isinstance(options, dict) else []
            valid = "".join([k for k in keys if re.fullmatch(r"[A-Z]", k)]) or "ABCDEFG"
            gt = extract_option_letter(gt_raw, valid) or gt_raw.strip().upper()
            pr = extract_option_letter(pred_raw, valid) or pred_raw.strip().upper()

        y_true.append(gt)
        y_pred.append(pr)

    out = mc.accuracy(y_true, y_pred)
    out["missing_preds"] = missing
    return out


def eval_task_03_08(samples: List[Dict[str, Any]], preds: Dict[str, str]) -> Dict[str, Any]:
    """
    统一评分逻辑：Task 03-08 都是单选题
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    missing = 0
    for s in samples:
        uid = s["uid"]
        if uid not in preds:
            missing += 1
            continue
        # 提取正确答案和预测答案
        gt_raw = str(s["gt"]).strip().upper()
        pred_raw = str(preds[uid])

        # 尝试从预测中提取选项字母（A-E）
        pred_extracted = extract_option_letter(pred_raw, "ABCDE") or pred_raw.strip().upper()

        y_true.append(gt_raw)
        y_pred.append(pred_extracted)

    out = mc.accuracy(y_true, y_pred)
    out["missing_preds"] = missing
    return out


def eval_task02(samples: List[Dict[str, Any]], preds: Dict[str, str], judge_model: str, cache: GPTCache) -> Dict[str, Any]:
    pairs: List[Tuple[str, str]] = []
    missing = 0
    for s in samples:
        uid = s["uid"]
        if uid not in preds:
            missing += 1
            continue
        pairs.append((str(s["gt"]), str(preds[uid])))

    out = mc.task02_metrics(
        pairs,
        gpt_model=judge_model,
        gpt_cache_get=cache.get,
        gpt_cache_set=cache.set,
    )
    out["missing_preds"] = missing
    return out


def eval_task09(samples: List[Dict[str, Any]], preds: Dict[str, str], judge_model: str, cache: GPTCache) -> Dict[str, Any]:
    pairs: List[Tuple[str, str]] = []
    missing = 0
    for s in samples:
        uid = s["uid"]
        if uid not in preds:
            missing += 1
            continue
        pairs.append((str(s["gt"]), str(preds[uid])))

    out = mc.task09_metrics(
        pairs,
        gpt_model=judge_model,
        gpt_cache_get=cache.get,
        gpt_cache_set=cache.set,
    )
    out["missing_preds"] = missing
    return out


def eval_task10(samples: List[Dict[str, Any]], preds: Dict[str, str], judge_model: str, cache: GPTCache) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for s in samples:
        meta = s.get("meta", {}) or {}
        items.append({
            "uid": s["uid"],
            "meta": {
                "question": meta.get("question", ""),
                "type": meta.get("type", "vqa")  # Task10默认为"vqa"（不是"unknown"）
            },
            "answer": s["gt"],
        })

    out = mc.task10_metrics(
        items,
        preds,
        gpt_model=judge_model,
        gpt_cache_get=cache.get,
        gpt_cache_set=cache.set,
    )
    return out

def eval_task11(samples: List[Dict[str, Any]], preds: Dict[str, str]) -> Dict[str, Any]:
    """
    Task11 yes/no 题评分逻辑
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    missing = 0

    for s in samples:
        uid = s["uid"]
        if uid not in preds:
            missing += 1
            continue

        gt_raw = str(s["gt"])
        pr_raw = str(preds[uid])

        # 提取yes/no答案
        gt = extract_yes_no(gt_raw) or gt_raw.strip()
        pr = extract_yes_no(pr_raw) or pr_raw.strip()

        y_true.append(gt)
        y_pred.append(pr)

    out = mc.accuracy(y_true, y_pred)
    out["missing_preds"] = missing
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", type=str, default="../dataset")
    ap.add_argument("--dataset_json_dir", type=str, default="../dataset/json_test_taskall")
    ap.add_argument("--run_dir", type=str, default="../runs/default_run")
    ap.add_argument("--tasks", type=str, default="1,2,3,4,5,6,7,8,9,10,11")
    ap.add_argument("--judge_model", type=str, default="gpt-5.2-2025-12-11")
    ap.add_argument("--cache_db", type=str, default=None)
    ap.add_argument("--enable_checkpoints", action="store_true", help="启用断点续推")
    ap.add_argument("--clear_checkpoints", action="store_true", help="清空检查点重新开始")
    args = ap.parse_args()

    tasks = [int(x.strip()) for x in args.tasks.split(",") if x.strip()]

    pred_dir = os.path.join(args.run_dir, "preds")
    metrics_dir = os.path.join(args.run_dir, "metrics")
    ensure_dir(metrics_dir)

    # 初始化改进工具
    data_loader = SafeDataLoader(skip_errors=True)
    checkpoint_manager = CheckpointManager(args.run_dir, auto_cleanup=False) if args.enable_checkpoints else None
    logger = EvaluationLogger(os.path.join(args.run_dir, "logs"))

    logger.info(f"Evaluation session started with tasks: {tasks}")

    # 清空检查点（如需要）
    if args.clear_checkpoints and checkpoint_manager:
        checkpoint_manager.cleanup()
        logger.info("Checkpoints cleared")

    cache_path = args.cache_db or os.path.join(args.run_dir, "gpt_cache.sqlite")
    cache = GPTCache(cache_path)

    summary: Dict[str, Any] = {
        "judge_model": args.judge_model,
        "tasks": {},
        "eval_mode": {
            "checkpoints": args.enable_checkpoints,
        }
    }

    for t in tasks:
        # 检查是否已完成
        if checkpoint_manager and checkpoint_manager.is_completed(t):
            logger.info(f"Task {t:02d} already completed, loading from checkpoint")
            cp = checkpoint_manager.load(t)
            summary["tasks"][f"task{t:02d}"] = cp.get("results", {})
            continue

        try:
            logger.info(f"Starting evaluation for task {t:02d}")

            samples = load_and_adapt(args.dataset_root, args.dataset_json_dir, t)
            pred_path = os.path.join(pred_dir, f"task{t:02d}.jsonl")

            # 使用安全加载
            preds = _load_preds_map(pred_path, data_loader, logger)

            if t == 1:
                m = eval_task01(samples, preds)
            elif t in [3, 4, 5, 6, 7, 8]:
                m = eval_task_03_08(samples, preds)
            elif t == 2:
                m = eval_task02(samples, preds, args.judge_model, cache)
            elif t == 9:
                m = eval_task09(samples, preds, args.judge_model, cache)
            elif t == 10:
                m = eval_task10(samples, preds, args.judge_model, cache)
            elif t == 11:
                m = eval_task11(samples, preds)
            else:
                raise ValueError(f"Unsupported task_id={t}")

            out_path = os.path.join(metrics_dir, f"task{t:02d}.json")
            save_json(out_path, m)
            summary["tasks"][f"task{t:02d}"] = m
            logger.info(f"Task {t:02d} completed: wrote metrics -> {out_path}")

            # 保存检查点
            if checkpoint_manager:
                checkpoint_manager.save(t, m, status="completed")

        except KeyboardInterrupt:
            logger.error("Evaluation interrupted by user")
            logger.info("Use checkpoint to resume: run again with same arguments")
            break
        except Exception as e:
            logger.error(f"Task {t:02d}: Unexpected error: {str(e)}")
            summary["tasks"][f"task{t:02d}"] = {"error": str(e)}
            if checkpoint_manager:
                checkpoint_manager.save(t, {"error": str(e)}, status="failed")

    save_json(os.path.join(metrics_dir, "summary.json"), summary)
    cache.close()

    # 生成评估报告
    report = generate_evaluation_report(args.run_dir, summary["tasks"])
    report_path = os.path.join(metrics_dir, "evaluation_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Summary saved -> {os.path.join(metrics_dir, 'summary.json')}")
    logger.info(f"Report saved -> {report_path}")
    logger.info(f"Cache saved -> {cache_path}")

    # 打印建议
    if report.get("recommendations"):
        print("\n" + "=" * 60)
        print("RECOMMENDATIONS")
        print("=" * 60)
        for rec in report["recommendations"]:
            print(f"  {rec}")
        print("=" * 60 + "\n")

    print(f"[eval] All tasks completed. Results saved to {metrics_dir}")


if __name__ == "__main__":
    main()
