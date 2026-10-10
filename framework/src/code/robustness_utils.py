"""健壮性工具：数据验证、显存监控、检查点管理、输出标准化"""

import json
import os
import sys
import torch
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime


class SafeDataLoader:
    """安全数据加载器，支持错误恢复和验证"""

    def __init__(self, skip_errors: bool = True, max_errors_log: int = 100):
        self.skip_errors = skip_errors
        self.max_errors_log = max_errors_log

    def load_jsonl(self, path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        安全加载 JSONL，返回 (rows, errors)
        """
        rows = []
        errors = []

        if not os.path.exists(path):
            errors.append(f"File not found: {path}")
            return rows, errors

        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        err_msg = f"Line {line_no}: {str(e)[:80]}"
                        if self.skip_errors:
                            if len(errors) < self.max_errors_log:
                                errors.append(err_msg)
                        else:
                            raise ValueError(err_msg)
        except Exception as e:
            errors.append(f"Failed to read {path}: {str(e)}")

        return rows, errors

    def load_json(self, path: str) -> Tuple[Dict[str, Any] | List, List[str]]:
        """安全加载 JSON"""
        errors = []

        if not os.path.exists(path):
            errors.append(f"File not found: {path}")
            return {}, errors

        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), errors
        except Exception as e:
            errors.append(f"Failed to load {path}: {str(e)}")
            return {}, errors


class CheckpointManager:
    """检查点管理：保存和恢复任务进度"""

    def __init__(self, run_dir: str, auto_cleanup: bool = False):
        self.checkpoint_dir = os.path.join(run_dir, ".checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.auto_cleanup = auto_cleanup

    def get_checkpoint_path(self, task_id: int) -> str:
        """获取任务检查点路径"""
        return os.path.join(self.checkpoint_dir, f"task_{task_id:02d}.json")

    def save(self, task_id: int, results: Dict[str, Any], status: str = "completed"):
        """保存任务检查点"""
        path = self.get_checkpoint_path(task_id)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        checkpoint = {
            "timestamp": datetime.now().isoformat(),
            "task_id": task_id,
            "status": status,  # "completed", "failed", "partial"
            "results": results,
        }

        with open(path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)

    def load(self, task_id: int) -> Optional[Dict[str, Any]]:
        """加载任务检查点"""
        path = self.get_checkpoint_path(task_id)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return json.load(f)
            except:
                return None
        return None

    def exists(self, task_id: int) -> bool:
        """检查是否有检查点"""
        return os.path.exists(self.get_checkpoint_path(task_id))

    def is_completed(self, task_id: int) -> bool:
        """检查任务是否已完成"""
        cp = self.load(task_id)
        return cp is not None and cp.get("status") == "completed"

    def cleanup(self):
        """清理检查点目录"""
        import shutil
        if os.path.exists(self.checkpoint_dir):
            shutil.rmtree(self.checkpoint_dir)
        os.makedirs(self.checkpoint_dir, exist_ok=True)


class EvaluationResult:
    """标准化的评估结果容器"""

    def __init__(self, task_id: int, judge_model: Optional[str] = None):
        self.task_id = task_id
        self.timestamp = datetime.now().isoformat()
        self.judge_model = judge_model
        self.summary = {}  # 汇总指标
        self.per_sample_results = []  # 逐样本结果
        self.errors = []
        self.warnings = []
        self.metadata = {
            "backend_version": "2.0",
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "cuda_available": torch.cuda.is_available(),
        }

    def add_error(self, error: str):
        """添加错误记录"""
        if len(self.errors) < 200:  # 最多记录200个错误
            self.errors.append(error)

    def add_warning(self, warning: str):
        """添加警告记录"""
        if len(self.warnings) < 100:
            self.warnings.append(warning)

    def to_dict(self, include_details: bool = False) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "judge_model": self.judge_model,
            "summary": self.summary,
            "metadata": self.metadata,
            "statistics": {
                "total_samples": len(self.per_sample_results),
                "error_count": len(self.errors),
                "warning_count": len(self.warnings),
            },
        }

        # 计算成功率
        if self.per_sample_results:
            success_count = sum(
                1 for r in self.per_sample_results if r.get("error") is None
            )
            result["statistics"]["success_rate"] = success_count / len(self.per_sample_results)

        # 仅在需要时包含详细信息
        if include_details:
            result["errors"] = self.errors
            result["warnings"] = self.warnings
            result["per_sample_results"] = self.per_sample_results[:100]  # 样本限制

        return result

    def save(self, output_dir: str, include_details: bool = False):
        """保存评估结果"""
        os.makedirs(output_dir, exist_ok=True)

        # 主结果文件
        with open(os.path.join(output_dir, f"task{self.task_id:02d}.json"), "w") as f:
            json.dump(self.to_dict(include_details=False), f, indent=2, default=str)

        # 详细结果（如果有）
        if include_details and self.per_sample_results:
            details_path = os.path.join(output_dir, f"task{self.task_id:02d}_details.jsonl")
            with open(details_path, "w") as f:
                for result in self.per_sample_results:
                    f.write(json.dumps(result, default=str) + "\n")

        # 错误日志
        if self.errors:
            with open(os.path.join(output_dir, f"task{self.task_id:02d}_errors.log"), "w") as f:
                for i, err in enumerate(self.errors, 1):
                    f.write(f"{i}. {err}\n")


class EvaluationLogger:
    """评估过程日志记录"""

    def __init__(self, log_dir: str):
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, "evaluation.log")
        self.setup_logger()

    def setup_logger(self):
        """初始化日志器"""
        self.logger = logging.getLogger("evaluation")
        self.logger.setLevel(logging.DEBUG)

        # 文件处理
        fh = logging.FileHandler(self.log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)

        # 控制台处理
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # 格式
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def info(self, message: str):
        """信息级日志"""
        self.logger.info(message)

    def warning(self, message: str):
        """警告级日志"""
        self.logger.warning(message)

    def error(self, message: str):
        """错误级日志"""
        self.logger.error(message)

    def debug(self, message: str):
        """调试级日志"""
        self.logger.debug(message)


def generate_evaluation_report(run_dir: str, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """生成综合评估报告"""
    report = {
        "generated_at": datetime.now().isoformat(),
        "run_directory": run_dir,
        "task_summary": {},
        "overall_statistics": {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "total_samples": 0,
            "total_errors": 0,
        },
        "recommendations": [],
    }

    for task_name, metrics in results.items():
        report["task_summary"][task_name] = {
            "completed": "accuracy" in metrics or "n" in metrics or "bleu1" in metrics,
            "metrics": metrics,
        }

        report["overall_statistics"]["total_tasks"] += 1

        if report["task_summary"][task_name]["completed"]:
            report["overall_statistics"]["completed_tasks"] += 1
            if "n" in metrics:
                report["overall_statistics"]["total_samples"] += metrics["n"]
        else:
            report["overall_statistics"]["failed_tasks"] += 1

    # 生成建议
    completion_rate = (
        report["overall_statistics"]["completed_tasks"] /
        report["overall_statistics"]["total_tasks"]
    ) if report["overall_statistics"]["total_tasks"] > 0 else 0

    if completion_rate < 1.0:
        failed = report["overall_statistics"]["failed_tasks"]
        report["recommendations"].append(
            f"⚠ {failed} 个任务未完成，检查错误日志了解原因"
        )

    if report["overall_statistics"]["total_errors"] > 0:
        report["recommendations"].append(
            f"⚠ 发现 {report['overall_statistics']['total_errors']} 个评估错误，"
            "建议检查输入数据质量"
        )

    return report
