# -*- coding: utf-8 -*-
"""
Task-level Prompt Templates.

This module defines the task-level prompts used in the inference pipeline.
These prompts are prepended to each sample's text to provide task-specific instructions.
"""

from typing import Dict, Optional


def get_task_prompt(
    task_id: int,
    question: str = "",
    options: Optional[Dict[str, str]] = None,
    type_id: Optional[int] = None
) -> str:
    """
    Get task-level prompt for a given task.

    Args:
        task_id: Task ID (1-11)
        question: Question text (for tasks that need it)
        options: Options dictionary (for multiple choice tasks)
        type_id: Type ID for Task 1 (1-4)

    Returns:
        Task-level prompt string
    """
    # Task1: 多类型分类任务
    if task_id == 1:
        if type_id == 3:
            # 4张图 + 单选 (原生多图输入)
            return (
                f"{question}\n\n"
                "You are shown 4 images as separate inputs:\n"
                "- Option A: First image\n"
                "- Option B: Second image\n"
                "- Option C: Third image\n"
                "- Option D: Fourth image\n\n"
                "Carefully examine all four images and answer the question above.\n"
                "Respond with ONLY a single letter: A, B, C, or D\n"
                "No explanation, no other text. Just the letter."
            )
        elif type_id == 4:
            # Yes/No问题（单图）
            return f"{question}\n\nAnswer only with 'Yes' or 'No'."
        elif type_id == 1:
            # 1张图 + 单选（文本选项）
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())]) if options else ""
            return (
                f"{question}\n\n"
                f"Options:\n{opts_str}\n\n"
                "Based on the provided image, please respond with ONLY the capital letter of the correct option (e.g., A, B, C, or D). Do not include any explanation."
            )
        elif type_id == 2:
            # 1张图 + 多选（文本选项）
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())]) if options else ""
            return (
                f"{question}\n\n"
                f"Options:\n{opts_str}\n\n"
                "Based on the provided image, please respond with the capital letters of the correct options (e.g., A,C). Do not include any explanation."
            )
        else:
            # 默认：要求单字母答案或Yes/No
            return (
                f"{question}\n\n"
                "Answer with ONLY a single capital letter (A, B, C, D, etc.) or 'Yes'/'No'. No explanation."
            )

    # Task2: 灾难变化检测 + Anomaly Reasoning (原生双图输入)
    elif task_id == 2:
        return (
            "Your TASK is to perform a comprehensive anomaly detection and disaster impact analysis on the provided pair of pre-disaster and post-disaster remote sensing images.\n\n"
            "You will act as a remote sensing analyst to:\n"
            "1. Identify anomalies (changes) between the pre-disaster and post-disaster states\n"
            "2. Classify the type of disaster\n"
            "3. Assess its impact on both built and natural environments across five specific categories\n\n"
            "The first image is the pre-disaster state.\n"
            "The second image is the post-disaster state.\n\n"
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

    # Task4: 二元判断 (是/否)
    elif task_id == 4:
        return (
            f"{question}\n\n"
            "Answer with ONLY 'Yes' or 'No'. No other text, no explanation, no punctuation.\n"
            "Just one word: Yes or No."
        )

    # Task5-8: 通用分类任务
    elif task_id in [5, 6, 7, 8]:
        if isinstance(options, dict) and options:
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            return (
                f"{question}\n\n"
                f"Options:\n{opts_str}\n\n"
                "Based on the provided image, please respond with ONLY the capital letter of the correct option. Do not include any explanation."
            )
        else:
            return f"{question}\n\nAnswer with ONLY a single capital letter (A, B, C, D, etc.). No explanation."

    # Task9: 图像字幕生成 (原Task10)
    elif task_id == 9:
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

    # Task10: 视觉问答 (原Task11)
    elif task_id == 10:
        return (
            f"{question}\n\n"
            "Answer the question using ONLY a single word or short phrase (max 3 words).\n"
            "Do not use punctuation or extra text."
        )

    # Task11: POPE (Polling-based Object Probing Evaluation)
    elif task_id == 11:
        return (
            f"{question}\n\n"
            "Answer with ONLY 'Yes' or 'No'. No other text, no explanation, no punctuation.\n"
            "Just one word: Yes or No."
        )

    # 默认: 通用推理
    else:
        if isinstance(options, dict) and options:
            opts_str = "\n".join([f"{k}: {v}" for k, v in sorted(options.items())])
            return f"{question}\n\nOptions:\n{opts_str}\n\nAnswer with just the letter."
        else:
            return f"{question}\n\nProvide a detailed answer based on the image."


def get_simple_task_prompt(task_id: int) -> str:
    """
    Get a simplified task-level prompt without sample-specific information.
    Used for feature extraction where we want consistent task context.

    Args:
        task_id: Task ID (1-11)

    Returns:
        Simplified task-level prompt string
    """
    # Task1: 多类型分类任务
    if task_id == 1:
        return (
            "You are performing a visual classification task. "
            "Analyze the provided image(s) and select the correct option. "
            "Respond with ONLY the capital letter of your choice (A, B, C, or D)."
        )

    # Task2: 灾难变化检测
    elif task_id == 2:
        return (
            "Your TASK is to perform a comprehensive anomaly detection and disaster impact analysis "
            "on the provided pair of pre-disaster and post-disaster remote sensing images. "
            "You will act as a remote sensing analyst to identify anomalies, classify the disaster type, "
            "and assess its impact on both built and natural environments."
        )

    # Task3: 图像分析
    elif task_id == 3:
        return (
            "Analyze the image to answer the following question. "
            "Choose the best option from the candidate options provided. "
            "Respond with ONLY the capital letter of the correct option."
        )

    # Task4: 二元判断
    elif task_id == 4:
        return "Answer the question with ONLY 'Yes' or 'No'. No other text or explanation."

    # Task5-8: 通用分类任务
    elif task_id in [5, 6, 7, 8]:
        return (
            "Based on the provided image, select the correct option. "
            "Respond with ONLY the capital letter of your choice."
        )

    # Task9: 图像字幕生成
    elif task_id == 9:
        return (
            "You are a remote sensing analyst. Write ONE caption for the given aerial/remote-sensing image. "
            "Describe the scene objectively with 3-7 sentences, including spatial descriptions and visible attributes."
        )

    # Task10: 视觉问答
    elif task_id == 10:
        return "Answer the question using ONLY a single word or short phrase (max 3 words)."

    # Task11: POPE
    elif task_id == 11:
        return "Answer the question with ONLY 'Yes' or 'No'. No other text or explanation."

    # 默认
    else:
        return "Analyze the image and provide your answer based on the question."
