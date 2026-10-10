# -*- coding: utf-8 -*-
"""
数据适配器 - 将原始数据转换为统一格式

原始数据来自: /dataset/3_Evaluation/{ID_Test,OOD_Test}/data.json
数据格式: 统一的 JSON 数组，每条记录包含:
  - task_id: 任务ID
  - unique_id: 唯一标识
  - instruction: 问题文本
  - label: 正确答案
  - image_path: 图像路径（task2有两个：pre/post）
  - options: 选项（task1,3-8有；task2,11没有）
"""

import os
from typing import Any, Dict, List

from io_utils import load_json


def _norm_path(p: str, base: str = "") -> str:
    """规范化路径 - 转换相对路径为绝对路径"""
    if not p:
        return ""

    # 移除开头的 ./
    p = p.lstrip("./")

    # 如果提供了base路径，将相对路径转换为绝对路径
    if base:
        full_path = os.path.join(base, p)
        return os.path.normpath(full_path)

    return os.path.normpath(p)


def load_and_adapt(dataset_root: str, dataset_json_dir: str, task_id: int) -> List[Dict[str, Any]]:
    """
    加载并适配数据为统一格式

    输入数据格式 (来自 data.json):
    {
      "task_id": int,
      "uid": str,
      "question": str,
      "ground_truth": str,
      "image_path": str (or "pre_image_path"/"post_image_path" for task2),
      "options": dict or str (可选),
      ...
    }

    输出格式 (统一格式):
    {
      "uid": str,
      "task_id": int,
      "image_path": str,
      "gt": str,
      "meta": {
        "question": str,
        "options": dict or str (可选),
        ...
      }
    }
    """

    # 加载数据 - 从指定子集的data.json加载所有样本
    data_path = os.path.join(dataset_root, dataset_json_dir, "data.json")
    base_dir = os.path.join(dataset_root, dataset_json_dir)

    items = []

    if os.path.exists(data_path):
        all_data = load_json(data_path)
        # 过滤出指定task_id的样本
        for item in all_data:
            if item.get("task_id") == task_id:
                items.append(item)
    else:
        # 如果data.json不存在，尝试旧的结构（ID_Test/OOD_Test子目录）
        id_data_path = os.path.join(dataset_root, dataset_json_dir, "ID_Test", "data.json")
        ood_data_path = os.path.join(dataset_root, dataset_json_dir, "OOD_Test", "data.json")

        if os.path.exists(id_data_path):
            id_data = load_json(id_data_path)
            for item in id_data:
                if item.get("task_id") == task_id:
                    item["_base_dir"] = os.path.join(dataset_root, dataset_json_dir, "ID_Test")
                    items.append(item)

        if os.path.exists(ood_data_path):
            ood_data = load_json(ood_data_path)
            for item in ood_data:
                if item.get("task_id") == task_id:
                    item["_base_dir"] = os.path.join(dataset_root, dataset_json_dir, "OOD_Test")
                    items.append(item)

    # 适配为统一格式
    samples = []
    for item in items:
        item_base_dir = item.pop("_base_dir", base_dir)
        sample = _adapt_item(item, item_base_dir, task_id)
        if sample:
            samples.append(sample)

    return samples


def _adapt_item(item: Dict[str, Any], dataset_root: str, task_id: int) -> Dict[str, Any]:
    """
    将单个原始数据项适配为统一格式

    Task1 有三种格式:
    1. 选择题(文本选项)：有 image_path，有 options（A-G的文本）
    2. Yes/No 问题：有 image_path，无 options，label="Yes"/"No"
    3. 选择题(图片选项)：无 image_path，options 的值是图片路径（A-D），label="A"-"D"
    """

    # 基础字段 (所有task都有)
    uid = item.get("uid", "")
    # Task 2 uses 'prompts' field instead of 'question'
    if task_id == 2:
        prompts = item.get("prompts", [])
        instruction = prompts[0] if prompts else ""
    else:
        instruction = item.get("question", "")
    label = item.get("ground_truth", "")

    # Task2需要特殊处理（两张图）
    if task_id == 2:
        pre_img = item.get("pre_image_path", "")
        post_img = item.get("post_image_path", "")
        pre_img = _norm_path(pre_img, dataset_root)
        post_img = _norm_path(post_img, dataset_root)

        return {
            "uid": uid,
            "task_id": task_id,
            "image_path": post_img,  # 主要用post图
            "gt": label,
            "meta": {
                "question": instruction,  # instruction已从prompts[0]提取
                "pre_image_path": pre_img,
                "post_image_path": post_img,
            }
        }

    # Task1 特殊处理 (三种格式)
    if task_id == 1:
        options = item.get("options", {})
        image_path = item.get("image_path", "")

        # 检查是否是图片选项格式 (options 的值是路径)
        if options and isinstance(options, dict):
            # 检查第一个选项值是否看起来像路径
            first_val = next(iter(options.values()), "")
            is_image_options = isinstance(first_val, str) and (
                first_val.startswith("./images/") or
                first_val.endswith(".jpg") or
                first_val.endswith(".png") or
                "/" in first_val
            )

            if is_image_options:
                # 格式3：选择题(图片选项) - 需要处理所有图片路径
                full_options = {}
                for k, v in options.items():
                    full_path = _norm_path(v, dataset_root)
                    full_options[k] = full_path

                # 找一个主要图片作为 image_path (通常是第一张)
                main_img = next(iter(full_options.values()), "")

                return {
                    "uid": uid,
                    "task_id": task_id,
                    "image_path": main_img,  # 第一张作为主要图片
                    "gt": label,
                    "meta": {
                        "question": instruction,
                        "options": full_options,  # 包含所有图片路径
                        "type": "image_options",  # 标记类型
                        "type_id": item.get("type_id"),
                        "raw_type": item.get("type", ""),
                    }
                }
            else:
                # 格式1：选择题(文本选项) 或 格式2：Yes/No
                image_path = _norm_path(image_path, dataset_root)
                type_id = item.get("type_id")
                raw_type = item.get("type", "")
                if type_id == 3 or raw_type == "multi_mcq_opt_text":
                    sample_type = "multi_mcq_opt_text"
                else:
                    sample_type = "text_options" if options else "yesno"

                # 保留原始options
                return {
                    "uid": uid,
                    "task_id": task_id,
                    "image_path": image_path,
                    "gt": label,
                    "meta": {
                        "question": instruction,
                        "options": options,
                        "type": sample_type,
                        "type_id": type_id,
                        "raw_type": raw_type,
                    }
                }
        else:
            # 无options (格式2：Yes/No)
            image_path = _norm_path(image_path, dataset_root)

            return {
                "uid": uid,
                "task_id": task_id,
                "image_path": image_path,
                "gt": label,
                "meta": {
                    "question": instruction,
                    "type": "yesno",
                    "type_id": item.get("type_id"),
                    "raw_type": item.get("type", ""),
                }
            }

    # 其他task (单张图)
    image_path = item.get("image_path", "")
    image_path = _norm_path(image_path, dataset_root)

    # 适配 options (task3-8可能有)
    options = item.get("options", {})
    if isinstance(options, str):
        # 某些task的options是字符串格式，保持原样
        options = options
    else:
        # 转为字典
        options = options if isinstance(options, dict) else {}

    return {
        "uid": uid,
        "task_id": task_id,
        "image_path": image_path,
        "gt": label,
        "meta": {
            "question": instruction,
            "options": options,
        }
    }
