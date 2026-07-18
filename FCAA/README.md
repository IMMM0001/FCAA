<p align="center">
  <h1 align="center">MO-FCAA</h1>
  <p align="center"><strong>Multi-Objective Fiddler Crab Asymmetric Algorithm</strong></p>
  <p align="center">面向小样本复杂物理体系的特征选择与模型超参数联合优化框架</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-yellow.svg" alt="License">
  <img src="https://img.shields.io/badge/status-Major%20Revision-orange.svg" alt="Status">
</p>

---

[**English**](#english) | [**中文**](#chinese)

---

<a name="english"></a>
## English

### Overview

MO-FCAA is a multi-objective optimization framework based on the **Fiddler Crab Asymmetric Algorithm (FCAA)**, a nature-inspired metaheuristic that mimics the asymmetric claw morphology of fiddler crabs (*Uca* genus). The algorithm simultaneously performs **feature selection** and **hyperparameter optimization** for machine learning models (SVR, KRR, Random Forest, MLP).

It is specifically designed for **small-sample, high-dimensional** materials informatics problems — exactly the regime where conventional methods succumb to overfitting.

> **Status**: This repository accompanies a journal manuscript currently under **Major Revision**. The experiment automation pipeline (30-run statistical analysis, sensitivity analysis, scalability testing) has been added to address reviewer concerns about statistical rigor and algorithmic generalization.

---

### Key Features

| Feature | Description |
|:---|:---|
| **FCAA v2 Asymmetric Update** | Cauchy mutation (heavy-tailed exploration) + Gaussian walk (fine-grained exploitation) |
| **Adaptive Scheduling** | Quadratic sigma decay, dynamic claw-ratio transition, adaptive leader selection |
| **Multi-Objective** | NSGA-II-style non-dominated sorting + crowding distance for Pareto front discovery |
| **Dual Objectives** | Minimize cross-validated RMSE + minimize number of selected features |
| **Built-in Baselines** | NSGA-II and MOPSO for head-to-head comparison |
| **30-Run Statistical Pipeline** | Wilcoxon signed-rank test, Friedman test, Nemenyi post-hoc analysis |
| **Sensitivity Analysis** | Grid search over core hyperparameters with heatmap visualization |
| **Scalability Testing** | Controlled-variable runtime scaling vs. population size and feature dimension |

---

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/MO-FCAA.git
cd MO-FCAA

# Install dependencies
pip install -r requirements.txt
pip install openpyxl   # Required for Excel data loading
```

**Requirements**: Python 3.10+, NumPy, SciPy, scikit-learn, Matplotlib, Pandas, Seaborn, tqdm, openpyxl.

---

### Quick Start

#### 1. Run a single comparison experiment

```bash
python experiments/run_comparison.py
```

This loads the Boron HEC dataset (150 samples), expands features via polynomial expansion (degree=3), adds noise distractors, and runs all three algorithms for 200 generations.

#### 2. 30-run statistical experiment (for publication)

```bash
# Full experiment: ~9 hours
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr

# Statistical analysis
python experiments/stats_analyzer.py --results-dir results/30runs --output-dir results/analysis
```

#### 3. Sensitivity analysis

```bash
python experiments/sensitivity_analysis.py --quick
```

#### 4. Scalability testing

```bash
python experiments/scalability_test.py --n-runs 3
```

---

### Algorithm

FCAA is inspired by the asymmetric claw morphology of fiddler crabs:

- **Major Claw (Exploration)**: Cauchy mutation with heavy-tailed jumps — escapes local optima
- **Minor Claw (Exploitation)**: Gaussian random walk with quadratically decaying variance — fine-tunes solutions
- **Dynamic Scheduling**: The claw ratio shifts from 80% exploration (early) to 15% (late), while the Gaussian sigma decays quadratically from σ₀ to nearly zero

| Phase | Major Claw | Minor Claw | σ | Claw Ratio |
|:------|:-----------|:-----------|:----|:-----------|
| Early | Cauchy → Leader | 70% Guided + 30% Diverse | 0.15 → 0.10 | 0.80 → 0.60 |
| Mid | Cauchy → Leader | 70% Guided + 30% Diverse | 0.10 → 0.02 | 0.60 → 0.35 |
| Late | Cauchy → Leader | 70% Guided + 30% Diverse | 0.02 → 0.002 | 0.35 → 0.15 |

Two objectives are minimized simultaneously: **f₁ = Cross-validated RMSE**, **f₂ = Fraction of selected features**.

For full algorithmic details, see [`FCAA_ALGORITHM_GUIDE.md`](FCAA_ALGORITHM_GUIDE.md) / [`FCAA_ALGORITHM_GUIDE.pdf`](FCAA_ALGORITHM_GUIDE.pdf).

---

### 30-Run Benchmark Results

*B_data.xlsx (Boron HEC Melting Point), 150 samples → 75 expanded features (poly degree=3 + 20 noise), SVR model, 5-fold CV, 200 generations, 30 independent runs.*

#### Descriptive Statistics (Best RMSE)

| Algorithm | Mean ± Std | 95% CI | Best |
|:----------|:----------:|:------:|:----:|
| **FCAA v2** | **108.08 ± 2.07** | [107.31, 108.85] | 105.08 |
| MOPSO | 110.75 ± 2.74 | [109.72, 111.77] | 106.43 |
| NSGA-II | 111.19 ± 2.90 | [110.11, 112.27] | 106.25 |

#### Statistical Significance

| Test | Result |
|:---|:---|
| **Wilcoxon**: FCAA vs NSGA-II | **p = 0.00017** ★★★ (large effect, r = 0.69) |
| **Wilcoxon**: FCAA vs MOPSO | **p = 0.00031** ★★★ (large effect, r = 0.66) |
| **Friedman** (omnibus) | χ² = 14.47, **p = 0.00072** ★★★ |
| **Nemenyi** CD | 0.856; FCAA significantly outranks MOPSO |

> **Conclusion**: FCAA v2 significantly outperforms both baselines with large effect sizes, demonstrating both statistical significance and methodological substance.

See [`FINAL_EXPERIMENT_RESULTS.md`](FINAL_EXPERIMENT_RESULTS.md) for the complete experimental report.

---

### Experiment Automation Pipeline

This repository includes a comprehensive experiment automation pipeline designed to address common reviewer concerns:

| Script | Purpose | Reviewer Concern Addressed |
|:---|:---|:---|
| `experiments/run_experiments.py` | 30-run batch execution with structured JSON+NPZ output | Statistical rigor, reproducibility |
| `experiments/stats_analyzer.py` | Descriptive stats, Wilcoxon, Friedman, Nemenyi, boxplots, CD diagrams | Statistical testing |
| `experiments/sensitivity_analysis.py` | Grid search over core FCAA parameters with heatmaps | Parameter sensitivity |
| `experiments/scalability_test.py` | Runtime scaling vs. population size & feature dimension | Computational scalability |

For a detailed walkthrough, see [`EXPERIMENT_AUTOMATION_GUIDE.md`](EXPERIMENT_AUTOMATION_GUIDE.md).

---

### Project Structure

```
MO-FCAA/
├── src/                          # Core source code
│   ├── algorithms/               # FCAA, NSGA-II, MOPSO optimizers
│   │   ├── base.py               # Abstract optimizer interface
│   │   ├── operators.py          # Cauchy mutation & Gaussian walk
│   │   ├── multi_objective.py    # Non-dominated sorting & crowding distance
│   │   ├── fcaa.py               # FCAA v2 optimizer
│   │   ├── nsga2.py              # NSGA-II optimizer
│   │   └── mopso.py              # MOPSO optimizer
│   ├── evaluators/               # Fitness evaluation
│   │   ├── base.py               # Abstract evaluator
│   │   ├── models.py             # SVR, KRR, RF, MLP wrappers
│   │   └── feature_selection_hpo.py  # FS + HPO evaluator
│   └── utils/                    # Data loading, encoding, metrics
├── experiments/                  # Experiment scripts
│   ├── common.py                 # Shared utilities
│   ├── run_experiments.py        # 30-run batch experiment
│   ├── stats_analyzer.py         # Statistical analysis
│   ├── sensitivity_analysis.py   # Parameter sensitivity
│   ├── scalability_test.py       # Scalability testing
│   ├── run_comparison.py         # Basic comparison
│   └── advanced_plots.py         # Advanced visualizations
├── data/                         # Experimental datasets
├── Dockerfile
├── requirements.txt
├── README.md                     # This file
├── FCAA_ALGORITHM_GUIDE.md/pdf   # Detailed algorithm documentation
├── EXPERIMENT_AUTOMATION_GUIDE.md # Experiment pipeline guide
└── FINAL_EXPERIMENT_RESULTS.md   # Complete experimental results
```

---

### Extending the Framework

#### Adding a New ML Model

```python
# In src/evaluators/models.py
class MyModelWrapper(ModelWrapper):
    @staticmethod
    def hyperparameter_bounds():
        return {"param1": (0.01, 100, "log")}
    
    @staticmethod
    def build_model(**hparams):
        return MyRegressor(param1=hparams["param1"])

# Register it
MODEL_REGISTRY["my_model"] = MyModelWrapper
```

#### Adding a New Baseline Algorithm

```python
# In experiments/common.py
from src.algorithms.your_algo import YourOptimizer

OPTIMIZER_REGISTRY["Your-Algo"] = {
    "class": YourOptimizer,
    "extra_params": {"param1": default_value, ...},
}
# All scripts (run_experiments, stats_analyzer) now auto-support it!
```

---

### Citation

```bibtex
@software{mo_fcaa_2026,
  title = {MO-FCAA: Multi-Objective Fiddler Crab Asymmetric Algorithm},
  year = {2026},
  author = {Zhang, A. et al.},
  note = {Feature selection and hyperparameter optimization for small-sample materials informatics}
}
```

---

### License

MIT License. See [LICENSE](LICENSE) file for details.

---

<a name="chinese"></a>
## 中文

### 概述

MO-FCAA 是基于**招潮蟹不对称算法（Fiddler Crab Asymmetric Algorithm, FCAA）**的多目标优化框架。该算法灵感来源于招潮蟹的非对称螯足形态——大螯用于大范围挥舞示威（探索），小螯用于近距离精确摄食（开发）。框架同时完成**特征选择**和机器学习模型**超参数优化**，支持 SVR、KRR、Random Forest 和 MLP 四种回归模型。

本框架专为**小样本、高维度**的材料信息学问题设计——这正是传统方法容易过拟合的场景。

> **当前状态**：本仓库配套一篇正在**大修（Major Revision）**的期刊论文。已新增完整的实验自动化流水线（30 轮统计检验、敏感性分析、可扩展性测试），以回应审稿人关于统计严谨性和算法泛化能力的意见。

---

### 核心特性

| 特性 | 说明 |
|:---|:---|
| **FCAA v2 非对称更新** | 柯西变异（重尾探索） + 高斯游走（精细开发） |
| **自适应调度机制** | 高斯 σ 二次衰减、动态螯比例转换、自适应领导者选择 |
| **多目标优化** | NSGA-II 风格非支配排序 + 拥挤距离，发现 Pareto 前沿 |
| **双目标函数** | 最小化交叉验证 RMSE + 最小化特征选择比例 |
| **内置基线算法** | NSGA-II 和 MOPSO，可直接对比 |
| **30 轮统计检验流水线** | Wilcoxon 符号秩检验、Friedman 检验、Nemenyi 事后检验 |
| **参数敏感性分析** | 核心参数网格搜索 + 热力图可视化 |
| **可扩展性测试** | 控制变量法测量运行时间与种群规模/特征维度的关系 |

---

### 安装

```bash
git clone https://github.com/your-username/MO-FCAA.git
cd MO-FCAA
pip install -r requirements.txt
pip install openpyxl
```

**依赖**: Python 3.10+, NumPy, SciPy, scikit-learn, Matplotlib, Pandas, Seaborn, tqdm, openpyxl.

---

### 快速开始

#### 1. 运行单次对比实验

```bash
python experiments/run_comparison.py
```

#### 2. 30 轮统计实验（用于论文）

```bash
# 完整实验（约 9 小时）
python experiments/run_experiments.py --n-runs 30 --dataset B_data --model svr

# 统计分析
python experiments/stats_analyzer.py --results-dir results/30runs --output-dir results/analysis
```

#### 3. 敏感性分析

```bash
python experiments/sensitivity_analysis.py --quick
```

#### 4. 可扩展性测试

```bash
python experiments/scalability_test.py --n-runs 3
```

---

### 算法原理

FCAA 受招潮蟹的非对称螯足形态启发：

- **大螯（探索）**：柯西变异，利用重尾分布产生大幅度跳跃，逃离局部最优
- **小螯（开发）**：高斯随机游走，方差按二次函数衰减，在后期精细调优
- **动态调度**：螯比例从早期 80% 探索逐渐过渡到后期 15% 开发；高斯 σ 从 σ₀ 二次衰减至接近零

| 阶段 | 大螯（探索） | 小螯（开发） | σ | 螯比例 |
|:------|:-----------|:-----------|:----|:--------|
| 早期 | Cauchy → 领导者 | 70% 引导 + 30% 多样 | 0.15 → 0.10 | 0.80 → 0.60 |
| 中期 | Cauchy → 领导者 | 70% 引导 + 30% 多样 | 0.10 → 0.02 | 0.60 → 0.35 |
| 后期 | Cauchy → 领导者 | 70% 引导 + 30% 多样 | 0.02 → 0.002 | 0.35 → 0.15 |

两个目标同时被最小化：**f₁ = 交叉验证 RMSE**、**f₂ = 选中特征比例**。

详细算法文档参见 [`FCAA_ALGORITHM_GUIDE.md`](FCAA_ALGORITHM_GUIDE.md) / [`FCAA_ALGORITHM_GUIDE.pdf`](FCAA_ALGORITHM_GUIDE.pdf)。

---

### 30 轮基准测试结果

*数据集：B_data.xlsx（硼基高熵陶瓷熔点），150 样本 → 75 特征（3 阶多项式展开 + 20 噪声），SVR 模型，5 折交叉验证，200 代，30 次独立运行。*

#### 描述性统计（Best RMSE）

| 算法 | 均值 ± 标准差 | 95% 置信区间 | 最优值 |
|:------|:----------:|:------:|:----:|
| **FCAA v2** | **108.08 ± 2.07** | [107.31, 108.85] | 105.08 |
| MOPSO | 110.75 ± 2.74 | [109.72, 111.77] | 106.43 |
| NSGA-II | 111.19 ± 2.90 | [110.11, 112.27] | 106.25 |

#### 统计显著性

| 检验 | 结果 |
|:---|:---|
| **Wilcoxon**: FCAA vs NSGA-II | **p = 0.00017** ★★★（大效应，r = 0.69） |
| **Wilcoxon**: FCAA vs MOPSO | **p = 0.00031** ★★★（大效应，r = 0.66） |
| **Friedman**（全局） | χ² = 14.47，**p = 0.00072** ★★★ |
| **Nemenyi** CD | 0.856；FCAA 显著优于 MOPSO |

> **结论**：FCAA v2 在统计显著性和效应量上均显著优于两个基线算法，可以为论文提供强有力的统计支撑。

完整实验报告参见 [`FINAL_EXPERIMENT_RESULTS.md`](FINAL_EXPERIMENT_RESULTS.md)。

---

### 实验自动化流水线

本仓库包含完整的实验自动化脚本，对应审稿意见中的各项要求：

| 脚本 | 功能 | 对应审稿意见 |
|:---|:---|:---|
| `experiments/run_experiments.py` | 30 轮批量运行，结构化 JSON+NPZ 输出 | 统计严谨性、可复现性 |
| `experiments/stats_analyzer.py` | 描述性统计、Wilcoxon、Friedman、Nemenyi、箱线图、CD 图 | 统计检验 |
| `experiments/sensitivity_analysis.py` | 核心参数网格搜索 + 热力图 | 参数敏感性 |
| `experiments/scalability_test.py` | 运行时间 vs 种群规模/特征维度 scaling | 计算可扩展性 |

详细使用指南参见 [`EXPERIMENT_AUTOMATION_GUIDE.md`](EXPERIMENT_AUTOMATION_GUIDE.md)。

---

### 项目结构

```
MO-FCAA/
├── src/                          # 核心源码
│   ├── algorithms/               # FCAA、NSGA-II、MOPSO 优化器
│   ├── evaluators/               # 适应度评估器
│   └── utils/                    # 数据加载、编码、指标计算
├── experiments/                  # 实验脚本
│   ├── common.py                 # 公共模块
│   ├── run_experiments.py        # 30 轮批量运行
│   ├── stats_analyzer.py         # 统计分析
│   ├── sensitivity_analysis.py   # 参数敏感性
│   ├── scalability_test.py       # 可扩展性测试
│   └── run_comparison.py         # 基础对比实验
├── data/                         # 实验数据集
├── Dockerfile
├── requirements.txt
├── README.md                     # 本文件（中英双语）
├── FCAA_ALGORITHM_GUIDE.md/pdf   # 算法技术文档
├── EXPERIMENT_AUTOMATION_GUIDE.md # 实验流水线指南
└── FINAL_EXPERIMENT_RESULTS.md   # 完整实验结果报告
```

---

### 扩展框架

#### 添加新 ML 模型

```python
# 在 src/evaluators/models.py 中
class MyModelWrapper(ModelWrapper):
    @staticmethod
    def hyperparameter_bounds():
        return {"param1": (0.01, 100, "log")}
    
    @staticmethod
    def build_model(**hparams):
        return MyRegressor(param1=hparams["param1"])

MODEL_REGISTRY["my_model"] = MyModelWrapper
```

#### 添加新基线算法

```python
# 在 experiments/common.py 中注册后，所有脚本自动支持！
OPTIMIZER_REGISTRY["My-Algo"] = {
    "class": MyOptimizer,
    "extra_params": {"param1": default_value},
}
```

---

### 引用

```bibtex
@software{mo_fcaa_2026,
  title = {MO-FCAA: Multi-Objective Fiddler Crab Asymmetric Algorithm},
  year = {2026},
  author = {Zhang, A. et al.},
  note = {Feature selection and hyperparameter optimization for small-sample materials informatics}
}
```

---

### 许可证

MIT License。详见 [LICENSE](LICENSE) 文件。
