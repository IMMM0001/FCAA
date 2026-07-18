# MO-FCAA 自动化实验流水线 — 完整使用指南

> **Automated Experiment Pipeline for Multi-Objective Fiddler Crab Asymmetric Algorithm**
>
> 面向审稿人大修（Major Revision）意见：统计严谨性、算法泛化能力、敏感性分析和可扩展性验证

---

## 目录

1. [概述与设计目标](#1-概述与设计目标)
2. [项目文件结构](#2-项目文件结构)
3. [模块一：30 轮独立运行与统计检验](#3-模块一30-轮独立运行与统计检验)
    - [3.1 批量运行脚本：`run_experiments.py`](#31-批量运行脚本run_experimentspy)
    - [3.2 统计分析脚本：`stats_analyzer.py`](#32-统计分析脚本stats_analyzerpy)
    - [3.3 公共模块：`common.py`](#33-公共模块commonpy)
4. [模块二：评估模型与基线算法扩展](#4-模块二评估模型与基线算法扩展)
    - [4.1 底层 ML 模型切换](#41-底层-ml-模型切换)
    - [4.2 标准化基线算法接口（OPTIMIZER_REGISTRY）](#42-标准化基线算法接口optimizer_registry)
    - [4.3 如何添加新基线算法](#43-如何添加新基线算法)
5. [模块三：参数敏感性分析](#5-模块三参数敏感性分析)
    - [5.1 网格搜索脚本：`sensitivity_analysis.py`](#51-网格搜索脚本sensitivity_analysispy)
    - [5.2 搜索参数空间](#52-搜索参数空间)
    - [5.3 输出热力图说明](#53-输出热力图说明)
6. [模块四：计算成本与可扩展性测试](#6-模块四计算成本与可扩展性测试)
    - [6.1 可扩展性测试脚本：`scalability_test.py`](#61-可扩展性测试脚本scalability_testpy)
    - [6.2 控制变量设置](#62-控制变量设置)
    - [6.3 输出折线图说明](#63-输出折线图说明)
7. [完整命令行参考](#7-完整命令行参考)
8. [输出文件格式规范](#8-输出文件格式规范)
9. [典型工作流](#9-典型工作流)
10. [故障排查与注意事项](#10-故障排查与注意事项)

---

## 1. 概述与设计目标

### 1.1 审稿意见对应关系

| 审稿意见关注点 | 对应模块 | 交付脚本 |
|:---|:---|:---|
| 统计严谨性（30 轮独立运行 + 假设检验） | 模块一 | `run_experiments.py` + `stats_analyzer.py` |
| 算法泛化能力（多模型、多数据集） | 模块二 | `common.py`（OPTIMIZER_REGISTRY）+ `run_experiments.py` |
| 参数敏感性分析（Grid Search + 热力图） | 模块三 | `sensitivity_analysis.py` |
| 计算可扩展性（种群规模/维度 scaling） | 模块四 | `scalability_test.py` |

### 1.2 设计原则

1. **不修改核心算法**：`src/algorithms/`、`src/evaluators/`、`src/utils/` 中的 FCAA 数学逻辑和更新机制**完全零修改**。
2. **可复现性**：所有随机性通过 `--seed` 控制，所有运行结果结构化保存（JSON + NPZ + CSV）。
3. **模块化设计**：公共功能抽取到 `common.py`，各实验脚本独立可运行。
4. **断点续跑**：长时间运行支持 `--resume` 跳过已完成的结果文件。
5. **发表级图片**：所有图表同时输出 PNG（预览）和 PDF（矢量，直接用于 LaTeX）。

---

## 2. 项目文件结构

```
MO-FCAA_Project/
│
├── src/                                # 核心算法（未修改）
│   ├── algorithms/
│   │   ├── base.py                     # BaseOptimizer 抽象基类
│   │   ├── operators.py                # Cauchy 突变 & Gaussian 游走
│   │   ├── multi_objective.py          # 非支配排序 & 拥挤距离
│   │   ├── fcaa.py                     # FCAA 优化器（未修改）
│   │   ├── nsga2.py                    # NSGA-II 优化器（未修改）
│   │   └── mopso.py                    # MOPSO 优化器（未修改）
│   ├── evaluators/
│   │   ├── base.py                     # BaseEvaluator 抽象基类
│   │   ├── models.py                   # SVR/KRR/RF/MLP 模型封装
│   │   └── feature_selection_hpo.py    # FS+HPO 联合评估器
│   └── utils/
│       ├── data.py                     # 数据加载 & 合成数据生成
│       ├── encoding.py                 # 解向量编码/解码
│       └── metrics.py                  # RMSE, Hypervolume, IGD
│
├── experiments/                        # ★ 实验脚本（本次交付）
│   ├── common.py                       # [新增] 公共模块
│   ├── run_comparison.py               # [重构] 原有对比实验（行为不变）
│   ├── run_experiments.py              # [新增] 30 轮批量运行
│   ├── stats_analyzer.py               # [新增] 统计分析
│   ├── sensitivity_analysis.py         # [新增] 参数敏感性
│   ├── scalability_test.py             # [新增] 可扩展性测试
│   ├── quick_test.py                   # [保留] 原有快速测试
│   ├── advanced_plots.py               # [保留] 原有高级可视化
│   └── split_figures.py               # [保留] 原有分图脚本
│
├── data/                               # 实验数据集
│   ├── B_data.xlsx                     # 硼基 HEC 熔点数据
│   ├── C_data.xlsx                     # 碳基 HEC 数据
│   └── C_tm_forFS.xlsx                 # 碳基熔点（用于特征选择）
│
├── results/                            # 输出目录（运行后自动生成）
│   ├── 30runs/                         # [run_experiments.py 输出]
│   │   ├── summary_30runs.csv          # 全局汇总
│   │   ├── FCAA/B_data/svr/            # 按算法/数据集/模型分层
│   │   │   ├── run_00_scalars.json     # 标量指标
│   │   │   ├── run_00_arrays.npz       # Pareto 前沿数组
│   │   │   ├── run_00_history.npy      # 适应度历史
│   │   │   └── ...
│   │   ├── NSGA-II/...
│   │   └── MOPSO/...
│   ├── analysis/                       # [stats_analyzer.py 输出]
│   │   ├── analysis_report.md          # Markdown 格式统计报告
│   │   ├── descriptive_stats.csv       # 描述性统计表
│   │   ├── wilcoxon_results.csv        # Wilcoxon 检验结果
│   │   ├── boxplot_comparison.png/pdf  # 箱线图
│   │   ├── cd_diagram.png/pdf          # 临界差异图
│   │   └── convergence_bands.png/pdf   # 收敛曲线（95% CI）
│   ├── sensitivity/                    # [sensitivity_analysis.py 输出]
│   │   ├── sensitivity_grid_results.csv
│   │   ├── heatmap_*.png/pdf           # 6 张热力图
│   │   └── interaction_plot.png/pdf    # 边际效应图
│   └── scalability/                    # [scalability_test.py 输出]
│       ├── scalability_population.csv
│       ├── scalability_dimension.csv
│       ├── scalability_analysis.png/pdf
│       └── per_eval_cost.png/pdf
│
├── README.md                           # 项目总览
├── FCAA_ALGORITHM_GUIDE.md             # 算法技术文档
├── EXPERIMENT_AUTOMATION_GUIDE.md      # ★ 本文档
├── requirements.txt
└── Dockerfile
```

---

## 3. 模块一：30 轮独立运行与统计检验

### 3.1 批量运行脚本：`run_experiments.py`

#### 功能描述

对 MO-FCAA 及所有基线算法（NSGA-II、MOPSO）执行 N 次独立运行（默认 30 次），每次使用不同的随机种子。所有结果结构化存储到磁盘，支持断点续跑。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--n-runs` | int | 30 | 每个组合的独立运行次数 |
| `--quick` | flag | — | 快速烟雾测试：5 runs × 50 gens × 1 数据集 × 1 模型 |
| `--dataset` | str | B_data | 单个数据集名（`B_data` / `C_data` / `C_tm_forFS`） |
| `--all-datasets` | flag | — | 遍历所有可用数据集 |
| `--model` | str | svr | 单个 ML 模型（`svr` / `krr` / `random_forest` / `mlp`） |
| `--all-models` | flag | — | 遍历所有可用模型 |
| `--algorithm` | str | — | 单个算法（默认：全部三个） |
| `--algorithms` | str[] | — | 指定算法列表，例如 `--algorithms FCAA NSGA-II` |
| `--pop-size` | int | 80 | 种群规模 |
| `--max-generations` | int | 200 | 最大迭代代数 |
| `--cv-folds` | int | 5 | 交叉验证折数 |
| `--poly-degree` | int | 3 | 多项式特征展开阶数 |
| `--seed` | int | 42 | 基础随机种子（run 0 用 seed，run N 用 seed+N） |
| `--output-dir` | str | results/30runs | 输出根目录 |
| `--no-resume` | flag | — | 不跳过已有结果（重新运行所有） |

#### 使用示例

```bash
# ===== 最常用：完整 30 轮实验（B_data + SVR，约 4 小时） =====
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr

# ===== 快速烟雾测试（验证 pipeline 是否正常，约 10 分钟） =====
python experiments/run_experiments.py --quick

# ===== 仅运行 FCAA 和 NSGA-II（跳过 MOPSO） =====
python experiments/run_experiments.py --n-runs 30 --algorithms FCAA NSGA-II

# ===== 单数据集 + 两个模型 =====
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model random_forest

# ===== 中断后继续（默认开启 --resume） =====
python experiments/run_experiments.py --n-runs 30
# 如果中途被 kill，重新运行相同命令即可从断点继续

# ===== 强制从头重新运行 =====
python experiments/run_experiments.py --n-runs 30 --no-resume

# ===== 自定义优化参数 =====
python experiments/run_experiments.py --n-runs 30 --pop-size 100 --max-generations 300

# ===== 全量实验（3 数据集 × 4 模型 × 3 算法 × 30 runs = 1080 次运行，非常耗时！）=====
python experiments/run_experiments.py --n-runs 30 --all-datasets --all-models
```

#### 输出文件说明

**目录树结构：**
```
results/30runs/
├── summary_30runs.csv                     # 全局汇总表
├── FCAA/
│   ├── B_data/
│   │   ├── svr/
│   │   │   ├── run_00_scalars.json        # Run 0 标量指标
│   │   │   ├── run_00_arrays.npz          # Run 0 Pareto 解集 + 适应度
│   │   │   ├── run_00_history.npy         # Run 0 适应度历史（降采样至 50 点）
│   │   │   ├── run_01_scalars.json
│   │   │   ├── ...
│   │   │   └── run_29_scalars.json
│   │   └── random_forest/
│   │       └── ...
│   └── C_data/
│       └── ...
├── NSGA-II/
│   └── ...
└── MOPSO/
    └── ...
```

**`run_XX_scalars.json` 字段说明：**

| 字段 | 类型 | 说明 |
|:---|:---|:---|
| `algorithm` | str | 算法名称 |
| `dataset` | str | 数据集名称 |
| `model` | str | ML 模型名称 |
| `run_id` | int | 运行编号（0-29） |
| `seed` | int | 实际使用的随机种子 |
| `best_rmse` | float | Pareto 前沿中最低的 RMSE |
| `mean_rmse` | float | Pareto 前沿中平均 RMSE |
| `median_rmse` | float | Pareto 前沿中 RMSE 中位数 |
| `worst_rmse` | float | Pareto 前沿中最高 RMSE |
| `best_feature_ratio` | float | Pareto 前沿中最低特征保留比例 |
| `mean_feature_ratio` | float | Pareto 前沿中平均特征保留比例 |
| `pareto_size` | int | Pareto 前沿解的数量 |
| `hypervolume` | float | 超体积指标（越大越好） |
| `hypervolume_history` | list[float] | 每代的超体积值（50 个采样点） |
| `time_seconds` | float | 总运行时间（秒） |
| `n_evaluations` | int | 适应度评估总次数 |

**`run_XX_arrays.npz` 内容：**

| 数组名 | 形状 | 说明 |
|:---|:---|:---|
| `pareto_population` | (K, D) | 最终 Pareto 前沿上的解向量 |
| `pareto_fitnesses` | (K, 2) | 对应的目标值 [RMSE, feature_ratio] |
| `fitness_history_final` | (pop_size, 2) | 最后一代的种群适应度值 |

---

### 3.2 统计分析脚本：`stats_analyzer.py`

#### 功能描述

读取 `run_experiments.py` 生成的 30 轮结构化结果，自动执行：

1. **描述性统计**：Mean、Std、Median、Min、Max、95% 置信区间
2. **Wilcoxon 符号秩检验**：FCAA 与每个基线算法的成对比较
3. **Friedman 检验**：跨所有算法的全局显著性检验
4. **Nemenyi 事后检验**：多算法比较的临界差异（Critical Difference）
5. **可视化**：箱线图 + CD 图 + 收敛曲线（含 95% 置信带）

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--results-dir` | str | results/30runs | 30 轮实验输出目录 |
| `--output-dir` | str | results/analysis | 分析和图表输出目录 |
| `--metric` | str | best_rmse | 分析的主指标（也可用 `hypervolume` 等） |
| `--metric-label` | str | Best RMSE | 指标在图表中的显示标签 |
| `--alpha` | float | 0.05 | 显著性水平 |
| `--dataset` | str | — | 按数据集筛选（例如仅分析 B_data） |
| `--model` | str | — | 按 ML 模型筛选 |

#### 使用示例

```bash
# ===== 标准分析（30 轮实验完成后） =====
python experiments/stats_analyzer.py --results-dir results/30runs --output-dir results/analysis

# ===== 分析特定数据集的结果 =====
python experiments/stats_analyzer.py --results-dir results/30runs --dataset B_data

# ===== 分析超体积指标 =====
python experiments/stats_analyzer.py --results-dir results/30runs --metric hypervolume --metric-label "Hypervolume"

# ===== 更严格的显著性水平 =====
python experiments/stats_analyzer.py --results-dir results/30runs --alpha 0.01
```

#### 输出说明

**生成文件列表：**

| 文件 | 格式 | 说明 |
|:---|:---|:---|
| `analysis_report.md` | Markdown | 完整统计报告（可直接粘贴到论文附录） |
| `descriptive_stats.csv` | CSV | 描述性统计表（可转为 LaTeX） |
| `wilcoxon_results.csv` | CSV | Wilcoxon 检验结果表 |
| `boxplot_comparison.png/pdf` | 图片 | 箱线图：显示 30 次运行的分布 |
| `cd_diagram.png/pdf` | 图片 | Critical Difference 图（Demšar 2006 风格） |
| `convergence_bands.png/pdf` | 图片 | 收敛曲线：均值 ± 95% 置信带 |

**分析报告示例：**

```markdown
## Descriptive Statistics: Best RMSE

| Algorithm | N | Mean | Std | Median | 95% CI | Min | Max |
|-----------|----|------|-----|--------|--------|-----|-----|
| FCAA | 30 | 112.34 | 8.21 | 110.80 | [109.2, 115.5] | 98.5 | 128.3 |
| NSGA-II | 30 | 128.67 | 12.45 | 127.20 | [123.8, 133.5] | 108.2 | 155.1 |
| MOPSO | 30 | 121.89 | 10.33 | 120.50 | [117.9, 125.9] | 104.6 | 142.8 |

## Wilcoxon Signed-Rank Test (α = 0.05)

| Comparison | N | W Statistic | p-value | Significant | Effect Size (r) |
|------------|---|-------------|---------|-------------|-----------------|
| FCAA vs NSGA-II | 30 | 44.0 | 0.000027 | Yes * | 0.7665 |
| FCAA vs MOPSO | 30 | 88.0 | 0.002188 | Yes * | 0.5593 |

## Friedman Test (Omnibus)

- χ² statistic: 10.40
- p-value: 0.0055 **← Significant**
- Nemenyi Critical Difference (CD): 0.856

**Average Ranks** (lower = better):
  - FCAA: 1.53
  - MOPSO: 2.13
  - NSGA-II: 2.33
```

#### 统计方法说明

**Wilcoxon 符号秩检验** 是一种非参数配对检验，用于比较两个算法在 30 个相同问题实例上的表现差异。零假设 H₀：两个算法的性能中位数没有差异。备择假设 H₁：存在显著差异。

**效应量 r** 的计算公式为：
```
r = |Z| / √(2N)
```
解释标准：r < 0.3（小效应）、0.3 ≤ r < 0.5（中效应）、r ≥ 0.5（大效应）。

**Friedman 检验** 是非参数版本的重复测量 ANOVA，用于检验三个算法是否来自同一分布。若 p < 0.05，拒绝零假设，表明至少一对算法存在显著差异。

**Nemenyi 事后检验** 在 Friedman 显著时，计算临界差异 CD。任何两算法的平均排名差超过 CD 即视为显著不同。

---

### 3.3 公共模块：`common.py`

#### 功能描述

提取所有实验脚本的公共功能，避免代码重复。提供了稳定、可复用的 API。

#### 核心函数

```python
# 1. 数据加载
X_train, X_test, y_train, y_test = load_and_preprocess_data(config)

# 2. 优化器工厂（通过注册表模式）
optimizer = create_optimizer("FCAA", evaluator, config, run_id=0)

# 3. 单次运行封装
result = run_single_algorithm("FCAA", evaluator, config, run_id=0)

# 4. 指标计算
metrics = compute_scalar_metrics(pareto_fitnesses, fitness_history, elapsed, n_evaluations)

# 5. 结果持久化
json_path, npz_path = save_run_results(result, output_dir, run_id)

# 6. 批量加载
df = collect_all_runs(Path("results/30runs"))
```

---

## 4. 模块二：评估模型与基线算法扩展

### 4.1 底层 ML 模型切换

`FeatureSelectionHPOEvaluator` 支持通过 `model_name` 参数切换底层回归模型。该机制通过 `src/evaluators/models.py` 中的 `MODEL_REGISTRY` 实现：

```python
MODEL_REGISTRY = {
    "svr": SVRWrapper,           # 支持向量回归（RBF 核）
    "krr": KRRWrapper,           # 核岭回归
    "random_forest": RandomForestWrapper,  # 随机森林
    "mlp": MLPWrapper,           # 多层感知机
}
```

**命令行中切换模型：**

```bash
# SVR（默认）
python experiments/run_experiments.py --n-runs 30 --model svr

# 随机森林
python experiments/run_experiments.py --n-runs 30 --model random_forest

# 核岭回归
python experiments/run_experiments.py --n-runs 30 --model krr

# 多层感知机
python experiments/run_experiments.py --n-runs 30 --model mlp

# 遍历所有模型
python experiments/run_experiments.py --n-runs 30 --all-models
```

**各模型的超参数搜索空间：**

| 模型 | 超参数 | 搜索范围 | 尺度 |
|:---|:---|:---|:---|
| SVR | C | [0.01, 10⁴] | log |
| SVR | ε | [0.001, 1.0] | log |
| SVR | γ | [10⁻⁴, 10] | log |
| KRR | α | [10⁻⁶, 10] | log |
| KRR | γ | [10⁻⁴, 10] | log |
| Random Forest | n_estimators | [10, 500] | integer |
| Random Forest | max_depth | [2, 30] | integer |
| Random Forest | min_samples_split | [2, 20] | integer |
| Random Forest | max_features | [0.1, 1.0] | linear |
| MLP | hidden_size | [16, 256] | integer |
| MLP | α | [10⁻⁶, 10] | log |
| MLP | learning_rate_init | [10⁻⁵, 0.1] | log |

### 4.2 标准化基线算法接口（OPTIMIZER_REGISTRY）

`experiments/common.py` 中定义了标准化的算法注册表，所有优化器通过统一的工厂函数 `create_optimizer()` 构建：

```python
OPTIMIZER_REGISTRY = {
    "FCAA": {
        "class": FCAAOptimizer,
        "extra_params": {
            "alpha_init": 1.0,
            "sigma_init": 0.15,
            "sigma_final": 0.002,
            "claw_ratio_init": 0.80,
            "claw_ratio_final": 0.15,
            "elite_ratio": 0.3,
            "elite_sigma": 0.015,
        },
    },
    "NSGA-II": {
        "class": NSGA2Optimizer,
        "extra_params": {
            "crossover_prob": 0.9,
            "eta_crossover": 20.0,
            "eta_mutation": 20.0,
        },
    },
    "MOPSO": {
        "class": MOPSOOptimizer,
        "extra_params": {
            "archive_size": 50,
            "inertia": 0.5,
            "cognitive": 1.5,
            "social": 1.5,
            "mutation_rate": 0.1,
        },
    },
}
```

**标准化接口约束：** 所有新基线算法必须：
1. 继承 `BaseOptimizer`（`src/algorithms/base.py`）
2. 实现 `optimize() -> Tuple[np.ndarray, np.ndarray]`（返回 Pareto 种群和适应度）
3. 构造函数接受：`dimension`, `pop_size`, `max_generations`, `fitness_fn`, `lower_bound`, `upper_bound`, `seed`

满足以上约束后，`create_optimizer()`、`run_experiments.py`、`stats_analyzer.py` 会自动与新算法兼容。

### 4.3 如何添加新基线算法

以添加 **NSGA-III** 为例：

**Step 1：创建新优化器类**

```python
# src/algorithms/nsga3.py
from .base import BaseOptimizer
from .multi_objective import non_dominated_sort, crowding_distance

class NSGA3Optimizer(BaseOptimizer):
    def __init__(self, dimension, pop_size, max_generations, fitness_fn,
                 lower_bound=0.0, upper_bound=1.0, seed=42,
                 n_divisions=12, crossover_prob=0.9,
                 eta_crossover=20.0, eta_mutation=20.0):
        super().__init__(dimension, pop_size, max_generations,
                        fitness_fn, lower_bound, upper_bound, seed)
        self.n_divisions = n_divisions
        # ... 初始化参考点、参考方向等

    def optimize(self, verbose=True):
        # ... 实现 NSGA-III 主循环
        return pareto_population, pareto_fitnesses
```

**Step 2：在 `common.py` 中注册**

```python
# experiments/common.py
from src.algorithms.nsga3 import NSGA3Optimizer

OPTIMIZER_REGISTRY["NSGA-III"] = {
    "class": NSGA3Optimizer,
    "extra_params": {
        "n_divisions": 12,
        "crossover_prob": 0.9,
        "eta_crossover": 20.0,
        "eta_mutation": 20.0,
    },
}
```

**Step 3：直接使用（所有脚本自动兼容）**

```bash
python experiments/run_experiments.py --n-runs 30 --algorithms FCAA NSGA-II NSGA-III
python experiments/stats_analyzer.py --results-dir results/30runs
```

---

## 5. 模块三：参数敏感性分析

### 5.1 网格搜索脚本：`sensitivity_analysis.py`

#### 功能描述

对 FCAA v2 的三个核心自适应参数执行完整的 3D 网格搜索（5 × 5 × 5 = 125 个组合），每个组合运行 N 次取平均。自动生成 6 张 2D 热力图和 1 张边际效应交互图。

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--n-runs` | int | 3 | 每个网格点的重复运行次数 |
| `--quick` | flag | — | 快速模式：1 run × 50 gens × 3³ = 27 网格点 |
| `--output-dir` | str | results/sensitivity | 输出目录 |
| `--model` | str | svr | ML 模型 |
| `--dataset` | str | B_data | 数据集名称 |

#### 使用示例

```bash
# ===== 完整敏感性分析（125 点 × 3 runs = 375 次运行，约 10-20 小时） =====
python experiments/sensitivity_analysis.py --n-runs 3

# ===== 快速模式（27 点 × 1 run = 27 次运行，约 30-60 分钟） =====
python experiments/sensitivity_analysis.py --quick

# ===== 指定模型和数据集 =====
python experiments/sensitivity_analysis.py --n-runs 3 --model random_forest --dataset C_data
```

### 5.2 搜索参数空间

| 参数 | 物理含义 | 搜索值 | FCAA 默认值 |
|:---|:---|:---|:---|
| `claw_ratio_init` (ρ) | 初始大小螯比例，控制探索/开发平衡 | [0.6, 0.7, 0.8, 0.9, 1.0] | **0.80** |
| `alpha_init` (α) | Cauchy 变异初始尺度，控制全局跳跃幅度 | [0.5, 0.75, 1.0, 1.25, 1.5] | **1.0** |
| `sigma_init` (σ) | Gaussian 游走初始标准差，控制局部搜索精度 | [0.05, 0.10, 0.15, 0.20, 0.25] | **0.15** |

**参数解释：**

- **ρ 过高 (> 0.9)**：探索过强 → 种群发散，难以收敛
- **ρ 过低 (< 0.6)**：开发过强 → 过早收敛，陷入局部最优
- **α 过大**：Cauchy 尾部太重 → 过度跳跃，破坏已找到的优良解
- **α 过小**：Cauchy 接近 Gaussian → 失去重尾探索优势
- **σ 过大**：局部搜索太粗糙 → 超参数无法精确调优
- **σ 过小**：局部搜索停滞 → 后期改进缓慢

### 5.3 输出热力图说明

**生成文件列表：**

| 文件 | 说明 |
|:---|:---|
| `sensitivity_grid_results.csv` | 所有 125 × N 次运行的完整记录 |
| `heatmap_claw_ratio_init_vs_alpha_init_best_rmse_mean.png/pdf` | ρ × α → RMSE 热力图 |
| `heatmap_claw_ratio_init_vs_alpha_init_hypervolume_mean.png/pdf` | ρ × α → HV 热力图 |
| `heatmap_claw_ratio_init_vs_sigma_init_best_rmse_mean.png/pdf` | ρ × σ → RMSE 热力图 |
| `heatmap_claw_ratio_init_vs_sigma_init_hypervolume_mean.png/pdf` | ρ × σ → HV 热力图 |
| `heatmap_alpha_init_vs_sigma_init_best_rmse_mean.png/pdf` | α × σ → RMSE 热力图 |
| `heatmap_alpha_init_vs_sigma_init_hypervolume_mean.png/pdf` | α × σ → HV 热力图 |
| `interaction_plot.png/pdf` | 3×2 边际效应图 |

**热力图解读指南：**

- **颜色越深** → 指标越好（RMSE 热力图：蓝色=低 RMSE=好；HV 热力图：红色=高 HV=好）
- **红色虚线框** → 标记全局最优参数组合
- **网格值** → 在第三个参数维度上取平均后的指标值
- 最佳参数组合即为论文推荐参数

---

## 6. 模块四：计算成本与可扩展性测试

### 6.1 可扩展性测试脚本：`scalability_test.py`

#### 功能描述

执行两个控制变量实验，测量算法运行时间随问题规模的变化规律：

1. **实验 1（种群规模缩放）**：固定特征维度 D=50，变化种群大小 N ∈ [20, 50, 100, 150, 200]
2. **实验 2（特征维度缩放）**：固定种群规模 N=50，变化特征数 D ∈ [25, 50, 100, 200, 400]

#### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|:---|:---|:---|:---|
| `--n-runs` | int | 3 | 每个配置的重复次数（取平均） |
| `--quick` | flag | — | 快速模式：1 run × 50 gens × 缩减网格 |
| `--output-dir` | str | results/scalability | 输出目录 |
| `--skip-dimension` | flag | — | 跳过维度缩放实验（仅运行种群缩放，更快） |
| `--skip-population` | flag | — | 跳过种群缩放实验（仅运行维度缩放） |

#### 使用示例

```bash
# ===== 完整可扩展性分析（2 实验 × 5 点 × 3 算法 × 3 runs = 90 次运行） =====
python experiments/scalability_test.py --n-runs 3

# ===== 快速模式（1 run × 50 gens，约 30-60 分钟） =====
python experiments/scalability_test.py --quick

# ===== 仅测试种群规模缩放（跳过维度实验，更快） =====
python experiments/scalability_test.py --quick --skip-dimension

# ===== 仅测试维度缩放 =====
python experiments/scalability_test.py --quick --skip-population
```

### 6.2 控制变量设置

**实验 1：种群规模缩放**

| 参数 | 固定值 |
|:---|:---|
| 特征维度 D | 50（合成数据） |
| 样本数 | 200 |
| 最大代数 | 100 |
| ML 模型 | SVR |
| 种群大小 N | [20, 50, 100, 150, 200] |

**实验 2：特征维度缩放**

| 参数 | 固定值 |
|:---|:---|
| 种群规模 N | 50 |
| 样本数 | 200 |
| 最大代数 | 100 |
| ML 模型 | SVR |
| 特征维度 D | [25, 50, 100, 200, 400] |

**测量指标：**
- 总运行时间（total_time_s）
- 每代平均时间（time_per_gen_s）
- 每次适应度评估时间（time_per_eval_ms）

### 6.3 输出折线图说明

**生成文件列表：**

| 文件 | 说明 |
|:---|:---|
| `scalability_population.csv` | 种群规模缩放原始数据 |
| `scalability_dimension.csv` | 特征维度缩放原始数据 |
| `scalability_analysis.png/pdf` | 双面板图：(a) Runtime vs N, (b) Runtime vs D |
| `per_eval_cost.png/pdf` | 双面板图：(a) 每次评估耗时 vs N, (b) 每次评估耗时 vs D |

**图表特点：**
- 每个算法用不同颜色区分（FCAA=红色、NSGA-II=蓝色、MOPSO=绿色）
- 误差条表示多次运行的 ±1 标准差
- D 轴使用对数尺度（log₂）以更好展示 scaling 关系
- 可据此估算大规模实验的计算预算

---

## 7. 完整命令行参考

### 7.1 `run_experiments.py` — 批量运行

```
usage: python experiments/run_experiments.py [OPTIONS]

Experiment size:
  --n-runs N            Runs per combination (default: 30)
  --quick               Quick smoke test: 5 runs × 50 gens

Dataset selection:
  --dataset NAME        Single dataset: B_data | C_data | C_tm_forFS
  --all-datasets        Run all available datasets

Model selection:
  --model NAME          ML model: svr | krr | random_forest | mlp
  --all-models          Run all available models

Algorithm selection:
  --algorithm NAME      Single algorithm: FCAA | NSGA-II | MOPSO
  --algorithms A B ...  Space-separated list

Optimization params:
  --pop-size N          Population size (default: 80)
  --max-generations N   Max generations (default: 200)
  --cv-folds N          Cross-validation folds (default: 5)
  --poly-degree N       Polynomial expansion degree (default: 3)

I/O:
  --output-dir DIR      Output root directory (default: results/30runs)
  --seed N              Base random seed (default: 42)
  --no-resume           Do NOT skip existing results
```

### 7.2 `stats_analyzer.py` — 统计分析

```
usage: python experiments/stats_analyzer.py [OPTIONS]

  --results-dir DIR     Input directory with 30-run results
  --output-dir DIR      Output directory for report and figures
  --metric NAME         Metric to analyze (default: best_rmse)
  --metric-label LABEL  Display label for metric
  --alpha FLOAT         Significance level (default: 0.05)
  --dataset NAME        Filter to specific dataset
  --model NAME          Filter to specific ML model
```

### 7.3 `sensitivity_analysis.py` — 参数敏感性

```
usage: python experiments/sensitivity_analysis.py [OPTIONS]

  --n-runs N            Runs per grid point (default: 3)
  --quick               Fast mode: 1 run × 50 gens × 27 points
  --output-dir DIR      Output directory
  --model NAME          ML model (default: svr)
  --dataset NAME        Dataset name (default: B_data)
```

### 7.4 `scalability_test.py` — 可扩展性

```
usage: python experiments/scalability_test.py [OPTIONS]

  --n-runs N            Runs per config (default: 3)
  --quick               Fast mode: 1 run × 50 gens × reduced grid
  --output-dir DIR      Output directory
  --skip-dimension      Skip dimension scaling experiment
  --skip-population     Skip population scaling experiment
```

---

## 8. 输出文件格式规范

### 8.1 `run_XX_scalars.json`

```json
{
  "algorithm": "FCAA",
  "dataset": "B_data",
  "model": "svr",
  "run_id": 0,
  "seed": 42,
  "best_rmse": 109.0314,
  "mean_rmse": 145.2341,
  "median_rmse": 137.5621,
  "worst_rmse": 198.2347,
  "best_feature_ratio": 0.1466,
  "mean_feature_ratio": 0.2266,
  "pareto_size": 80,
  "hypervolume": 45.23,
  "hypervolume_history": [10.36, 12.03, ..., 45.23],
  "time_seconds": 318.5,
  "n_evaluations": 16000
}
```

### 8.2 `summary_30runs.csv`

CSV 文件，每行对应一次运行，包含 `run_XX_scalars.json` 中的所有标量字段。可直接用 pandas 读取：

```python
import pandas as pd
df = pd.read_csv("results/30runs/summary_30runs.csv")
```

### 8.3 `run_XX_arrays.npz`

```python
import numpy as np

data = np.load("results/30runs/FCAA/B_data/svr/run_00_arrays.npz")
pareto_pop = data["pareto_population"]      # (K, D)
pareto_fit = data["pareto_fitnesses"]        # (K, 2)
final_fit = data["fitness_history_final"]    # (pop_size, 2)
```

### 8.4 `run_XX_history.npy`

```python
import numpy as np
history = np.load("results/30runs/FCAA/B_data/svr/run_00_history.npy", allow_pickle=True)
# list of np.ndarray, each shape (pop_size, 2), ~50 samples across generations
```

---

## 9. 典型工作流

### 9.1 论文返修标准流程（从零开始）

```bash
# Step 0: 安装依赖
pip install -r requirements.txt openpyxl

# Step 1: 快速验证 pipeline（约 10 分钟）
python experiments/run_experiments.py --quick

# Step 2: 运行完整 30 轮实验（SVR + B_data，约 4 小时）
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr

# Step 3: 扩展实验（随机森林 + B_data）
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model random_forest

# Step 4: 运行统计分析
python experiments/stats_analyzer.py --results-dir results/30runs --output-dir results/analysis

# Step 5: 参数敏感性分析（约 10-20 小时，建议 overnight）
python experiments/sensitivity_analysis.py --n-runs 3

# Step 6: 可扩展性测试（约 2-4 小时）
python experiments/scalability_test.py --n-runs 3

# Step 7: 查看结果
#  - results/analysis/analysis_report.md  → 统计报告，可直接粘贴到论文附录
#  - results/analysis/*.pdf               → 矢量图，可直接用于 LaTeX
#  - results/sensitivity/*.pdf            → 热力图
#  - results/scalability/*.pdf            → scaling 图
```

### 9.2 快速验证工作流（检查代码是否正常）

```bash
# 三步快速验证（总计约 40 分钟）
python experiments/run_experiments.py --quick
python experiments/stats_analyzer.py --results-dir results/test_quick --output-dir results/test_analysis
python experiments/scalability_test.py --quick --skip-dimension

# 检查输出
ls results/test_analysis/
ls results/test_scalability/
```

### 9.3 审稿人要求额外基线时的流程

```bash
# 1. 在 src/algorithms/ 中实现新算法（如 nsga3.py）
# 2. 在 common.py 的 OPTIMIZER_REGISTRY 中注册
# 3. 运行（其余一切自动兼容）
python experiments/run_experiments.py --n-runs 30 --algorithms FCAA NSGA-II MOPSO NSGA-III
python experiments/stats_analyzer.py --results-dir results/30runs
```

---

## 10. 故障排查与注意事项

### 10.1 常见问题

| 问题 | 原因 | 解决方案 |
|:---|:---|:---|
| `Missing optional dependency 'openpyxl'` | 缺少 Excel 读取库 | `pip install openpyxl` |
| 运行到一半被 kill | 系统或终端中断 | 重新运行相同命令（默认 `--resume` 会跳过已完成的结果） |
| `results/30runs/` 中部分 run 缺少 | 某些配置运行失败（如内存不足） | 查看 `summary_30runs.csv` 中的 `error` 列 |
| 统计脚本报 `No data found` | results-dir 路径错误 | 确保目录中存在 `*_scalars.json` 文件 |
| 热力图显示异常 | 网格点数太少导致插值失败 | 确保 `--n-runs` ≥ 1，每个网格点至少有结果 |

### 10.2 性能参考

| 任务 | 配置 | 预估时间 | 说明 |
|:---|:---|:---|:---|
| 单次 200 代优化（FCAA, N=80） | SVR, 75 特征 | ~40-45 秒 | 取决于 CPU |
| 30 轮实验 | 1 数据集 × 1 模型 × 3 算法 | ~3.5-4 小时 | 建议 overnight |
| 30 轮 × 4 模型 | 1 数据集 × 4 模型 × 3 算法 | ~14-16 小时 | 建议分多次运行 |
| 敏感性分析 | 125 点 × 3 runs | ~10-20 小时 | 建议 overnight |
| 可扩展性 | 完整 | ~2-4 小时 | — |

### 10.3 并行化建议

当前脚本为单进程串行。如需加速，可以：
- 分数据集/模型手动并行：在不同终端运行不同 `--dataset --model` 组合
- 利用 `--resume` 机制：多个终端写入同一输出目录，但**需确保不写入同一文件**

---

## 附录 A：核心设计决策记录

### A.1 为什么结果存储用 JSON + NPZ 而非全部 CSV？

- **JSON**：存储嵌套结构（如 `hypervolume_history` 列表）比 CSV 扁平化更自然
- **NPZ**：存储大矩阵（Pareto 前沿解集维度可达 80×78）比 CSV 更高效紧凑
- **CSV**：仅用于顶层汇总（`summary_30runs.csv`），方便快速查看和 Excel 打开

### A.2 为什么统计检验同时用 Wilcoxon 和 Friedman？

- **Wilcoxon**（配对）：直接比较 FCAA vs 单个基线，给出效应量和 pairwise p 值
- **Friedman**（全局）：同时比较所有算法，控制多重比较的 family-wise error
- **Nemenyi**（事后）：Friedman 显著后，确定具体哪些算法对之间存在差异

这是机器学习领域标准的非参数统计检验流程（参考 Demšar, JMLR 2006）。

### A.3 为什么不修改 `src/` 下的任何文件？

保证 FCAA 算法核心的数学逻辑和更新机制**绝对不变**。所有自动化脚本通过外部的 `experiments/` 层调用已有接口，确保：

- 已报告的基准结果可完全复现
- 新的统计分析基于同一算法版本
- 审稿人可以直接对比论文中报告的算法行为

---

## 附录 B：依赖清单

```
numpy >= 1.24.0
scipy >= 1.10.0
scikit-learn >= 1.3.0
matplotlib >= 3.7.0
pandas >= 2.0.0
seaborn >= 0.12.0
tqdm >= 4.65.0
openpyxl (for Excel data loading)
```

---

> **文档版本**: v1.0  
> **最后更新**: 2026-07-14  
> **适用项目**: MO-FCAA (Multi-Objective Fiddler Crab Asymmetric Algorithm)
