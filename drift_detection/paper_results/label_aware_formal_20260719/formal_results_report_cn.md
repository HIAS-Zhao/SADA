# Label-Aware 漂移检测实验报告

## 1. 实验记录与审计

联合审计状态为 `pass`，错误数为 0。

| 实验族 | 正式运行记录数 |
|---|---:|
| Closed-set 主实验 | 756 |
| Post-change budget | 126 |
| Task-macro | 315 |
| Ordering sensitivity | 147 |
| Heterogeneous formats | 1848 |

所有实验均使用 3 个随机种子 `42/43/44`。发布结果与完整实验工作区的图表生成
产物见第 10 节。

## 2. 输入与协议审计

### 2.1 严格正确性信号

Closed-set 实验对允许答案集合进行 task-aware option scoring，
并根据完整选项得分得到 `0/1` correctness。

| 数据池 | 样本数 | 错误率 | Correct-option NLL 均值 |
|---|---:|---:|---:|
| Calibration | 1364 | 25.66% | 0.609 |
| No drift | 1643 | 25.56% | 0.591 |
| Drift validation | 596 | 68.12% | 1.650 |
| Drift test | 2339 | 67.98% | 1.639 |

### 2.2 统一 NLL 预处理

三份 teacher-forced NLL 文件共 14825 条记录，错误行 0、重复 UID 0，并与
feature metadata 的 UID 内容和顺序完全一致：

| Split | 样本数 |
|---|---:|
| Calibration | 3515 |
| No drift | 3664 |
| Natural drift | 7646 |

NLL 计算配置为：

- `max_image_pixels=1003520`
- natural-drift manifest
- Qwen2.5-VL-3B-Instruct
- teacher-forced mean answer NLL

### 2.3 统一检测协议

- Window size: 200
- Main windows per ratio: 1000
- Calibration windows: 300
- SADA calibration windows: 10000
- Validation windows: 200
- Seeds: 42, 43, 44
- Calibration target: `FPR <= 5%`
- Selection rule: 在满足 calibration FPR 约束的配置中最大化独立 drift-validation power
- SADA: `V+VT`（代码特征组 `A+E_text`）, PCA 128, `max_norm`, quantile 0.99
- DDM、EDDM、ADWIN、HDDM-W、Page-Hinkley、KSWIN 使用 River 实现
- OPTWIN 实现为 `local OPTWIN-style reproduction`

若某个方法没有任何配置满足 calibration FPR 约束，则结果明确记为
`not_calibratable`，不会静默删除。

## 3. Closed-Set Tasks with Explicit Accuracy

Closed-set 结果使用 natural weighting 和 binary-error stream。FPR 为独立 no-drift
测试窗口的检测率，不是 calibration FPR。

| 方法 | Test FPR | R@5 | R@10 | R@20 | Mean drift recall |
|---|---:|---:|---:|---:|---:|
| SADA | 0.00% | 90.47% | 100.00% | 100.00% | 99.09% |
| DDM | 0.07% | 0.20% | 2.97% | 42.63% | 11.11% |
| EDDM | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% |
| ADWIN | 0.03% | 0.30% | 7.30% | 86.03% | 26.36% |
| HDDM-W | 2.60% | 11.90% | 65.60% | 97.63% | 62.48% |
| Page-Hinkley | 16.43% | 13.87% | 41.20% | 90.43% | 48.13% |
| KSWIN | 6.47% | 2.50% | 34.13% | 91.47% | 44.64% |
| OPTWIN | N/A | N/A | N/A | N/A | N/A |

主要结论：

1. 在显式正确率这一原生适用条件下，有标签方法并未全部失效。HDDM-W、
   ADWIN 和 KSWIN 在 10% 到 20% 漂移比例下能够获得较高检测率。
2. SADA 的优势主要体现在低漂移比例。5% 漂移时，SADA 为 90.47%，最佳
   binary-error 基线 HDDM-W 为 11.90%。
3. 部分方法满足 calibration FPR 约束，但在独立 no-drift 测试集上出现更高
   FPR。这反映跨任务或跨分布的阈值泛化问题，不是校准规则被违反。

Correct-option NLL 补充结果中，ADWIN 的 R@10/R@20 为 57.40%/99.00%，
但 test FPR 为 15.57%；KSWIN 为 25.67%/93.97%，test FPR 为 5.27%。

## 4. Post-Change Sample Budget

每个窗口保持 200 个样本，分别放入 10、20、40、50、100、200 个漂移样本。

| 方法 | 10 | 20 | 40 | 50 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| SADA | 91.10% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% |
| HDDM-W | 13.20% | 63.57% | 97.37% | 99.07% | 99.53% | 100.00% |
| ADWIN | 3.50% | 54.80% | 99.03% | 99.60% | 99.10% | 100.00% |
| KSWIN | 1.07% | 24.43% | 94.10% | 96.47% | 97.00% | 99.97% |
| Page-Hinkley | 0.40% | 3.23% | 51.33% | 84.83% | 98.27% | 100.00% |

该实验说明监督顺序检测器并非无法检测漂移，而是通常需要更多 post-change
样本积累；SADA 在仅 10 到 20 个漂移样本时已经达到很高检测率。

## 5. Task-Macro Results

Task-macro 使用 Tasks 1/3/5/6/7 分别构造窗口，再对任务等权平均。

| 方法 | Macro R@5 | Macro R@10 | Macro R@20 |
|---|---:|---:|---:|
| SADA | 99.99% | 100.00% | 100.00% |
| HDDM-W | 14.05% | 58.40% | 86.01% |
| ADWIN | 3.86% | 35.77% | 96.21% |
| KSWIN | 1.80% | 38.68% | 81.23% |
| Page-Hinkley | 0.33% | 3.30% | 35.01% |
| EDDM | 0.06% | 0.26% | 2.40% |
| DDM | 0.00% | 0.00% | 0.00% |

SADA 的优势不是由某一个大任务的样本量主导。监督方法在不同任务上的检测率
差异较大，尤其是 ADWIN、HDDM-W 和 KSWIN 的 task-level 标准差较高。

## 6. Heterogeneous Tasks with Open-Ended Generation

以下结果使用 task-balanced 统计。监督列为所有可校准方法的 method-macro
平均。

| 格式支持级别 | SADA recall / FPR | Global NLL recall / FPR | Format-aware recall / FPR |
|---|---:|---:|---:|
| MCQ | 98.39% / 0.03% | 28.15% / 20.13% | 28.15% / 20.13% |
| + Yes/No | 95.76% / 0.10% | 26.64% / 19.87% | 26.64% / 19.87% |
| + Open short | 89.37% / 0.37% | 6.12% / 8.60% | 24.46% / 8.60% |
| + Caption | 94.00% / 1.07% | 5.57% / 8.16% | 24.85% / 7.99% |
| + Long report | 95.40% / 0.73% | 2.51% / 7.90% | 12.89% / 7.82% |

### 6.1 全局监督信号失效机制

统一 NLL 的 pooled calibration q95 约为 3.000。不同格式的损失尺度存在明显
差异：

| 格式与 split | Mean NLL | q95 | 超过 pooled q95 的比例 |
|---|---:|---:|---:|
| Calibration MCQ | 0.293 | 1.129 | 0.00% |
| Calibration open short | 1.388 | 4.115 | 11.58% |
| Calibration caption | 2.194 | 2.980 | 4.76% |
| No-drift open short | 1.382 | 4.082 | 13.74% |
| Drift long report | 1.927 | 2.244 | 0.00% |

因此，简单的 pooled high-loss event 同时存在两类问题：

1. 对 no-drift open-short 样本产生较多高损失事件，造成误报。
2. 1179 条 drift long-report 样本全部低于 pooled q95，导致全局监督事件流对
   开放长文本漂移近乎失明。

Format-aware ECDF 能缓解格式尺度错配，但不能完全恢复检测能力。在最终
long-report mixture 中，最佳方法 OPTWIN-style 从 global NLL 的 14.77%
提升到 format-aware NLL 的 31.24%，仍明显低于 SADA 的 95.40%。

### 6.2 解释边界

`+ Open short` 和 `+ Caption` 两级扩展的是 calibration/no-drift 格式支持和
SADA reference support。由于当前 natural-drift 数据中没有对应的 open-short
或 caption drift task，这两级的 drift pool 仍为 MCQ+Yes/No。只有
`+ Long report` 级别真正将开放长文本生成样本加入 drift stream。

前两级属于 progressive format-support/calibration ablation；
它们扩展参考与校准格式支持，未向 drift stream 加入对应生成任务。

## 7. Sensitivity to Stream Ordering

Ordering 实验固定使用 160 个 clean 样本和 40 个 drift 样本，比较 shuffled、
block 1/2/5/10/20 和 fully contiguous。

| 方法 | Shuffled | Block 5 | Block 20 | Contiguous |
|---|---:|---:|---:|---:|
| SADA | 100.00% | 100.00% | 100.00% | 100.00% |
| DDM | 28.23% | 32.33% | 46.70% | 43.90% |
| HDDM-W | 32.97% | 39.60% | 89.90% | 97.47% |
| ADWIN | 53.47% | 59.07% | 92.30% | 99.23% |
| Page-Hinkley | 39.67% | 43.30% | 55.67% | 51.57% |
| KSWIN | 12.83% | 12.10% | 66.17% | 94.23% |

SADA 对窗口内排列保持不变，而多个监督顺序检测器会随漂移样本连续性增加而
显著提高检测率。

Shuffled/block 指标为 `window detection rate`；这些布局不存在唯一的真实
onset，因此不采用 change-point recall。

## 8. SADA Within-Window Permutation Invariance

- 检查窗口数: 100
- 每窗口排列数: 20
- 检测分歧窗口数: 0
- 检测分歧率: 0
- 最大 fused-score span: `1.78e-15`
- 平均 fused-score span: `7.55e-16`

结果仅存在浮点舍入量级差异，验证了 SADA 的窗口级排列不变性。

## 9. 指标定义与解释范围

1. SADA 是窗口检测器。Budget 和 ordering 中为 SADA 记录的固定窗口预算
   表示窗口内样本数，与 sequential detection delay 的定义不同。
2. Sequential detector 的 conditional/censored delay 适用于 contiguous onset
   协议。
3. Calibration FPR 与独立 no-drift test FPR 是不同数据集上的统计；后者可以超过 5%。
4. `OPTWIN` 对应 `local OPTWIN-style reproduction`。
5. `not_calibratable` 表示没有配置满足 calibration FPR 约束，对应结果为 N/A。
6. Heterogeneous compact table 的监督结果是预先规定的 method-macro 平均，
   不是在 test set 上后验选择的最佳方法。

## 10. 关键产物

本仓库发布的表格和联合审计记录：

- `paper_artifacts/heterogeneous_long_report_per_method.md`
- `audit/formal_output_audit.md`

以下文件由 `scripts/summarize_label_aware_formal_results.py` 在完整实验工作区的
`paper_artifacts/` 输出目录生成，未随本仓库发布。生成这些文件需要完整实验
记录、监督损失文件和排列不变性输入：

- `paper_artifacts/closed_set_binary_main.md`
- `paper_artifacts/closed_set_continuous_supplement.md`
- `paper_artifacts/closed_set_task_macro.md`
- `paper_artifacts/recall_vs_post_change_budget.png`
- `paper_artifacts/ordering_block_contiguity_sensitivity.png`
- `paper_artifacts/answer_format_loss_ecdf.png`
- `paper_artifacts/heterogeneous_global_vs_format_aware.md`
- `paper_artifacts/sada_permutation_invariance.json`
