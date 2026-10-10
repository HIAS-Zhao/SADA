# -*- coding: utf-8 -*-
"""
通用推理脚本 - 支持任意视觉语言模型

工作流：
1. 加载原始数据 (ID + OOD 子集)
2. 调用用户指定的推理函数进行预测
3. 使用eval_all.py的评估函数进行评估
4. 生成结果报告

使用方式：
  python generic_inference.py --model_type {your_model} --model_path {path}
"""

import json
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Callable
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from io_utils import ensure_dir, save_json, load_json, write_jsonl
from eval_all import (
    eval_task01, eval_task_03_08, eval_task02, eval_task09,
    eval_task10, eval_task11, _load_preds_map
)
from data_adapters import load_and_adapt
from robustness_utils import (
    SafeDataLoader, EvaluationLogger,
    generate_evaluation_report
)
from gpt_cache import GPTCache


# ============================================================================
# Prompt辅助函数
# ============================================================================
def _option_labels(options: Dict[str, Any] = None) -> List[str]:
    if isinstance(options, dict) and options:
        return [str(k).upper() for k in sorted(options)]
    return list("ABCD")


def _label_examples(options: Dict[str, Any] = None) -> str:
    labels = _option_labels(options)
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _ordinal(index: int) -> str:
    ordinals = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
    return ordinals[index] if index < len(ordinals) else f"#{index + 1}"


# ============================================================================
# 日志配置
# ============================================================================
def setup_logger(run_dir: str, name: str = "generic_inference") -> logging.Logger:
    """配置日志记录"""
    ensure_dir(run_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    log_file = os.path.join(run_dir, "inference.log")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


# ============================================================================
# Prompt 构造（通用）
# ============================================================================
def build_prompt(task_id: int, question: str, options: Dict[str, Any] = None,
                 sample_type: str = None, type_id: int = None) -> str:
    """
    构造模型输入prompt - 适用于所有后端

    这个函数将prompt逻辑集中在公共代码中，避免每个实验都要重复实现。
    不同后端可以直接调用此函数而不需要自己实现。

    Args:
        task_id: 任务ID (1-11)
        question: 问题/指令文本
        options: 选项字典 {A: "选项A", B: "选项B", ...}
        sample_type: 样本类型 (image_options/yesno/text_options)

    Returns:
        格式化的prompt字符串

    说明：
        Task1: 图像分类选择（需要根据sample_type调整提示方式）
        Task2: 灾难变化检测（Before-After对比）
        Task9: 图像字幕生成
        Task10: 视觉问答 (需要LLM评分)
        其他: 通用推理
    """

    # Task1特殊处理
    if task_id == 1:
        if type_id == 1 or sample_type == "image_options":
            # 4张图 + 单选
            labels = _option_labels(options)
            image_options = "\n".join(
                f"- {label}: image {index + 1} ({_ordinal(index)} attached image)"
                for index, label in enumerate(labels)
            )
            return (
                f"{question}\n\n"
                f"You are given {len(labels)} attached candidate images in order. "
                "Each candidate image corresponds to exactly one option letter:\n"
                f"{image_options}\n\n"
                "Choose the option whose image best answers the question. "
                "Do not answer with the crop name, pest name, image filename, or option text.\n"
                f"Respond with ONLY one capital letter: {_label_examples(options)}.\n"
                "No explanation, punctuation, or extra text."
            )
        elif type_id == 4:
            # Yes/No问题（单图）
            return f"{question}\n\nAnswer only with 'Yes' or 'No'."
        elif type_id == 3:
            # 1张图 + 多选（文本选项）
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            return (
                f"{question}\n\n"
                f"Options:\n{opts_str}\n\n"
                "Based on the provided image, respond with ONLY the capital letters of all correct options as a comma-separated list (e.g., B,C). Do not include any explanation."
            )
        elif type_id == 2 or sample_type == "text_options":
            # 1张图 + 单选（文本选项）
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            return (
                f"{question}\n\n"
                f"Options:\n{opts_str}\n\n"
                f"Based on the provided image, please respond with ONLY the capital letter of the correct option (e.g., {_label_examples(options)}). Do not include any explanation."
            )
        else:
            # 默认：要求单字母答案或Yes/No
            return (
                f"{question}\n\n"
                "Answer with ONLY a single capital letter (A, B, C, D, etc.) or 'Yes'/'No'. No explanation."
            )

    # Task2: 灾难变化检测 + Anomaly Reasoning (Before-After对比)
    elif task_id == 2:
        return (
            "Your TASK is to perform a comprehensive anomaly detection and disaster impact analysis on the provided pair of pre-disaster and post-disaster remote sensing images.\n\n"
            "You will act as a remote sensing analyst to:\n"
            "1. Identify anomalies (changes) between the pre-disaster and post-disaster states\n"
            "2. Classify the type of disaster\n"
            "3. Assess its impact on both built and natural environments across five specific categories\n\n"
            "pre-disaster image:\n"
            "<image>\n\n"
            "post-disaster image:\n"
            "<image>\n\n"
            "Anomaly Reasoning Process:\n"
            "- Compare pixel-level changes and spatial patterns between the two images\n"
            "- Identify areas with significant modifications (damage, destruction, or transformation)\n"
            "- Correlate observed anomalies with known disaster signatures\n"
            "- Assess the severity and extent of changes\n\n"
            "Your analysis must be formatted as follows:\n"
            "DISASTER: [the name of the disaster]\n"
            "BUILDING: [describe impacts on buildings with specific anomalies observed]\n"
            "ROAD: [describe impacts on road networks with location and extent]\n"
            "VEGETATION: [describe impacts on natural, unmanaged vegetation cover]\n"
            "WATER_BODY: [describe changes to water bodies including new formations or modifications]\n"
            "AGRICULTURE: [describe impacts on managed agricultural land]\n"
            "ANOMALY_SUMMARY: [describe the key anomalies detected and their spatial distribution]\n"
            "CONCLUSION: [provide a concise 1-2 sentence summary synthesizing the overall disaster impacts observed across the categories.]"
        )

    # Task3: 图像分析 - 选择最佳选项
    elif task_id == 3:
        if isinstance(options, dict) and options:
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            return (
                "Analyze the image to answer the following question. Choose the best option from the candidate options provided.\n\n"
                f"Question: {question}\n"
                f"Options:\n{opts_str}\n\n"
                "Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text."
            )
        else:
            return (
                "Analyze the image to answer the following question. Choose the best option from the candidate options provided.\n\n"
                f"Question: {question}\n\n"
                "Your task is to respond with ONLY the capital letter of the correct option (e.g., C). Do not include any explanation or other text."
            )

    # Task10: 图像字幕生成 - 遥感分析师视角
    elif task_id == 10:
        return (
            "You are a remote sensing analyst. Write ONE caption for the given aerial/remote-sensing image.\n\n"
            "Requirements:\n"
            "- 3–7 sentences.\n"
            "- Start with: \"The image, sourced from GoogleEarth, shows ...\" \n"
            "  If the source is unknown, start with: \"The aerial image shows ...\"\n"
            "- Mention whether the image appears high-resolution or not.\n"
            "- Describe the scene objectively: main area type (e.g., urban, rural, industrial facility, airport, harbor, water body).\n"
            "- Include counts of prominent objects when they are clearly visible (e.g., two airplanes, multiple storage tanks).\n"
            "- Use clear spatial descriptions (top/bottom/left/right/center; near edges/corners; clustered/isolated; surrounded by).\n"
            "- Highlight distinctive visible attributes (shape, layout, texture, relative size), but do NOT invent details not visible.\n"
            "- Do NOT mention day/night, weather, people, or vehicle motion.\n"
            "- For airports, mention tarmac/runways/terminals/boarding bridges only if clearly visible.\n"
            "- For roads, mention straight/curved segments and density if clear.\n\n"
            "Output:\n"
            "Return only the caption text, with no quotes, no bullet points, and no extra commentary."
        )

    # Task11: 视觉问答（原Task10）
    elif task_id == 11:
        return (
            f"{question}\n\n"
            "Answer the question using ONLY a single word or short phrase (max 3 words).\n"
            "Do not use punctuation or extra text."
        )

    # Task4: 二元判断 (是/否)（原Task11）
    elif task_id == 4:
        return (
            f"{question}\n\n"
            "Answer with ONLY 'yes' or 'no'. No other text, no explanation, no punctuation.\n"
            "Just one word: yes or no."
        )

    # 其他Task: 通用推理
    else:
        if isinstance(options, dict) and options:
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            if task_id == 6:
                return (
                    f"{question}\n\n"
                    f"Options:\n{opts_str}\n\n"
                    "Your task is to respond with ONLY the capital letter of the correct option "
                    f"(e.g., {_label_examples(options)}). Do not include any explanation, option text, "
                    "punctuation, or other words."
                )
            return f"{question}\n\nOptions:\n{opts_str}\n\nAnswer with just the letter."
        else:
            return f"{question}\n\nProvide a detailed answer based on the image."


# ============================================================================
# 图像处理工具函数（通用）
# ============================================================================
def get_image_paths(sample: Dict[str, Any]) -> List[str]:
    """
    从样本中提取所有图像路径 - 适用于所有任务

    支持：
    - Task2: 灾前灾后两张图 (pre_image_path, post_image_path)
    - Task1: 图片选项 (options 中的图片路径)
    - 其他: 主图 (image_path)

    Args:
        sample: 样本字典，包含 image_path, meta (pre_image_path, post_image_path, options)

    Returns:
        图像路径列表。如果没有找到任何图像，返回 ["dummy"] 作为占位符
    """
    paths = []
    meta = sample.get("meta", {})

    # Task2：灾前灾后两张图
    if "pre_image_path" in meta and meta["pre_image_path"]:
        paths.append(meta["pre_image_path"])
    if "post_image_path" in meta and meta["post_image_path"]:
        paths.append(meta["post_image_path"])

    # Task1图片选项：所有选项都是图片
    options = meta.get("options", {})
    if isinstance(options, dict):
        # 检查是否都是路径
        for k in sorted(options):
            v = options[k]
            if isinstance(v, str) and ("/" in v or v.endswith((".jpg", ".png", ".jpeg", ".webp"))):
                paths.append(v)

    # 主图
    if sample.get("image_path"):
        if not paths:  # 没有其他图就用主图
            paths.append(sample["image_path"])
        elif len(paths) == 1:  # 已有其他图就追加主图
            paths.insert(0, sample["image_path"])

    return paths if paths else ["dummy"]


def load_images(image_paths: List[str]) -> List[Any]:
    """
    加载图像文件 - 适用于所有后端

    使用 PIL 加载图像，转换为 RGB 格式。
    对于失败的图像，跳过它们，最后确保返回至少一个占位符图像。

    Args:
        image_paths: 图像文件路径列表

    Returns:
        PIL Image 对象列表。如果没有成功加载任何图像，返回一个 100x100 的占位符
    """
    from PIL import Image

    max_pixels_env = os.environ.get("VLM_MAX_IMAGE_PIXELS")
    max_pixels = int(max_pixels_env) if max_pixels_env else None
    resampling = getattr(Image, "Resampling", Image)

    def _downscale_if_needed(img: Any) -> Any:
        if max_pixels is None:
            return img
        width, height = img.size
        pixels = width * height
        if pixels <= max_pixels:
            return img

        scale = (max_pixels / float(pixels)) ** 0.5
        new_size = (
            max(1, int(width * scale)),
            max(1, int(height * scale)),
        )
        return img.resize(new_size, resampling.LANCZOS)

    images = []
    for path in image_paths:
        if path == "dummy":  # 跳过占位符路径
            continue
        try:
            img = Image.open(path).convert("RGB")
            img = _downscale_if_needed(img)
            images.append(img)
        except Exception:
            # 跳过无法加载的图像，不打印警告
            pass

    return images if images else [Image.new("RGB", (100, 100))]


# ============================================================================
# 推理接口
# ============================================================================
class InferenceBackend:
    """推理后端基类 - 用户需要继承并实现predict方法"""

    def __init__(self, model_path: str, device: str = "cuda", **kwargs):
        """
        初始化后端

        Args:
            model_path: 模型路径
            device: cuda / cpu
            **kwargs: 其他模型特定参数
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.processor = None

    def load_model(self):
        """加载模型 - 子类必须实现"""
        raise NotImplementedError("Subclass must implement load_model()")

    def predict(self, sample: Dict[str, Any]) -> str:
        """
        对单个样本进行推理

        Args:
            sample: 包含以下字段
                - image_path: 图像路径
                - meta: 元数据（instruction, options等）
                - task_id: 任务ID

        Returns:
            预测答案字符串
        """
        raise NotImplementedError("Subclass must implement predict()")

    def unload_model(self):
        """卸载模型"""
        self.model = None
        self.processor = None


# ============================================================================
# 推理管理器
# ============================================================================
class InferenceManager:
    """管理批量推理和结果存储"""

    def __init__(self, run_dir: str, backend: InferenceBackend, logger: logging.Logger):
        self.run_dir = run_dir
        self.backend = backend
        self.logger = logger

        self.preds_dir = os.path.join(run_dir, "preds")
        self.metrics_dir = os.path.join(run_dir, "metrics")
        ensure_dir(self.preds_dir)
        ensure_dir(self.metrics_dir)

    def run_inference(self, samples: List[Dict[str, Any]], batch_size: int = 1) -> Dict[str, str]:
        """
        运行推理，支持批处理，返回 {uid: prediction}

        Args:
            samples: 样本列表
            batch_size: 批处理大小（单个样本推理，但用于内存管理）
        """
        preds = {}

        self.logger.info(f"Running inference on {len(samples)} samples (batch_size={batch_size})")

        # 分批处理
        for batch_idx in tqdm(range(0, len(samples), batch_size), desc="Inference batches"):
            batch_samples = samples[batch_idx:batch_idx + batch_size]

            for sample in batch_samples:
                uid = sample["uid"]
                try:
                    prediction = self.backend.predict(sample)
                    preds[uid] = prediction
                except Exception as e:
                    self.logger.error(f"Error inferring {uid}: {e}")
                    # 为不同任务类型提供备选答案
                    task_id = sample.get("task_id", 1)
                    meta = sample.get("meta", {})
                    sample_type = meta.get("type", "")

                    # 对于image_options类型，返回默认选项A（无法推理时的保守猜测）
                    if sample_type == "image_options":
                        preds[uid] = "A"
                    # 对于yes/no类型，返回默认答案No（更保守的选择）
                    elif sample_type == "yesno" or task_id == 4:
                        preds[uid] = "No"
                    # 对于其他多选题，返回A
                    else:
                        preds[uid] = "A"

        return preds

    def save_predictions(self, task_id: int, preds: Dict[str, str]) -> str:
        """保存预测为JSONL格式"""
        pred_file = os.path.join(self.preds_dir, f"task{task_id:02d}.jsonl")

        rows = [{"uid": uid, "pred": pred} for uid, pred in preds.items()]
        write_jsonl(pred_file, rows)

        self.logger.info(f"Saved {len(rows)} predictions to {pred_file}")
        return pred_file

    def evaluate(self, task_id: int, samples: List[Dict[str, Any]],
                 preds: Dict[str, str], cache: GPTCache = None) -> Dict[str, Any]:
        """评估单个任务"""
        if task_id == 1:
            return eval_task01(samples, preds)
        elif task_id in [3, 5, 6, 7, 8, 9]:
            return eval_task_03_08(samples, preds)
        elif task_id == 2:
            if cache is None:
                self.logger.warning(f"Task 2: No GPT cache provided, returning empty metrics")
                return {}
            return eval_task02(samples, preds, "gpt-4", cache)
        elif task_id == 10:
            if cache is None:
                self.logger.warning(f"Task 10: No GPT cache provided, returning empty metrics")
                return {}
            return eval_task09(samples, preds, "gpt-4", cache)
        elif task_id == 11:
            if cache is None:
                self.logger.warning(f"Task 11: No GPT cache provided, returning empty metrics")
                return {}
            return eval_task10(samples, preds, "gpt-4", cache)
        elif task_id == 4:
            return eval_task11(samples, preds)
        else:
            return {"error": f"Unsupported task_id={task_id}"}

    def save_metrics(self, task_id: int, metrics: Dict[str, Any]) -> str:
        """保存评估指标"""
        metrics_file = os.path.join(self.metrics_dir, f"task{task_id:02d}.json")
        save_json(metrics_file, metrics)
        self.logger.info(f"Saved metrics to {metrics_file}")
        return metrics_file


# ============================================================================
# 主程序
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="通用推理+评估脚本"
    )

    parser.add_argument(
        '--infer_module',
        type=str,
        required=True,
        help='推理模块路径 (module.path:ClassName)，如 my_inference:QwenBackend'
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='模型路径'
    )

    parser.add_argument(
        '--dataset_root',
        type=str,
        default='../dataset',
        help='数据集根目录'
    )

    parser.add_argument(
        '--dataset_json_dir',
        type=str,
        default='../dataset/json_test_taskall',
        help='数据集JSON目录'
    )

    parser.add_argument(
        '--run_dir',
        type=str,
        default='./runs/default',
        help='结果保存目录'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='1,2,3,4,5,6,7,8,9,10,11',
        help='任务列表 (逗号分隔)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='推理设备'
    )

    parser.add_argument(
        '--skip_inference',
        action='store_true',
        help='跳过推理，仅评估'
    )

    parser.add_argument(
        '--judge_model',
        type=str,
        default='gpt-4',
        help='用于评估的GPT模型'
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logger(args.run_dir)
    logger.info("Starting Generic Inference and Evaluation")
    logger.info(f"Infer module: {args.infer_module}")
    logger.info(f"Model path: {args.model_path}")

    # 动态导入推理类
    try:
        module_path, class_name = args.infer_module.rsplit(':', 1)
        module = __import__(module_path, fromlist=[class_name])
        BackendClass = getattr(module, class_name)
        logger.info(f"Loaded backend: {BackendClass.__name__}")
    except Exception as e:
        logger.error(f"Failed to load inference module: {e}")
        sys.exit(1)

    # 解析任务列表
    tasks = [int(x.strip()) for x in args.tasks.split(",") if x.strip()]
    logger.info(f"Tasks: {tasks}")

    # 初始化推理后端和管理器
    backend = BackendClass(model_path=args.model_path, device=args.device)
    manager = InferenceManager(args.run_dir, backend, logger)

    # 初始化缓存（用于GPT评估）
    cache = GPTCache(os.path.join(args.run_dir, "gpt_cache.sqlite"))

    # 加载数据加载器
    data_loader = SafeDataLoader(skip_errors=True)

    # 执行推理和评估
    summary = {"tasks": {}, "config": vars(args)}

    for task_id in tasks:
        try:
            logger.info(f"=== Task {task_id:02d} ===")

            # 加载数据
            samples = load_and_adapt(args.dataset_root, args.dataset_json_dir, task_id)
            logger.info(f"Loaded {len(samples)} samples for task {task_id:02d}")

            # 推理
            if not args.skip_inference:
                if backend.model is None:
                    backend.load_model()

                preds = manager.run_inference(samples)
                manager.save_predictions(task_id, preds)
            else:
                pred_path = os.path.join(manager.preds_dir, f"task{task_id:02d}.jsonl")
                if os.path.exists(pred_path):
                    preds = _load_preds_map(pred_path, data_loader, logger)
                else:
                    logger.warning(f"Prediction file not found: {pred_path}")
                    continue

            # 评估
            metrics = manager.evaluate(task_id, samples, preds, cache)
            manager.save_metrics(task_id, metrics)

            summary["tasks"][f"task{task_id:02d}"] = metrics
            logger.info(f"Task {task_id:02d} accuracy: {metrics.get('accuracy', 'N/A')}")

        except Exception as e:
            logger.error(f"Task {task_id:02d} failed: {e}")
            summary["tasks"][f"task{task_id:02d}"] = {"error": str(e)}

    # 保存汇总
    summary_path = os.path.join(manager.metrics_dir, "summary.json")
    save_json(summary_path, summary)
    logger.info(f"Summary saved to {summary_path}")

    cache.close()
    logger.info("Done!")


if __name__ == '__main__':
    main()
