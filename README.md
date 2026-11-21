# CSE558: Data Science — Assignment 1
## Dimensionality Reduction and Random Projections for K-Means and Linear Regression

### Course & Student Information
* **Course:** CSE558: Data Science
* **Assignment Title:** Assignment 1 — Dimensionality Reduction & Randomized Linear Algebra
* **Group Members:**
  * **Atharv Goel** — Roll Number: `MT26155`
  * **Ishwaani Batra** — Roll Number: `MT26166`

---

## 📁 Repository Structure

This repository contains the complete implementation, experimental results, and report for Assignment 1:

```text
├── MT26155-MT26166-DSA1-Jupyter-File.ipynb   # Executed Jupyter notebook with all code, outputs, tables, and answers
├── MT26155-MT26166-DSA1-Report.pdf         # 7-page PDF report containing answers, summary tables & comparative plots
└── README.md                               # Project documentation, methodology, reproduction guide, and findings
```

* **[`MT26155-MT26166-DSA1-Jupyter-File.ipynb`](MT26155-MT26166-DSA1-Jupyter-File.ipynb):** Fully executed top-to-bottom from a clean kernel with all random seeds pinned (`SEED = 42`). Contains self-contained implementations without external random projection wrappers.
* **[`MT26155-MT26166-DSA1-Report.pdf`](MT26155-MT26166-DSA1-Report.pdf):** Clean 7-page report containing exclusively written answers, labeled experimental tables, and high-resolution visualizations without code clutter.

---

## 📊 Dataset & Preprocessing

The experiments are conducted on the default 10% subset of the **KDD Cup 1999** network intrusion dataset ($n = 494,021$ connection records):

1. **One-Hot Encoding:** The 3 categorical text features (`protocol_type`, `service`, `flag`) were one-hot encoded into 80 binary indicator columns. Combined with the 38 numerical features, this produced an initial feature matrix of 118 columns.
2. **Dropping Constant Features:** Exactly 2 columns had identical values across all 494,021 rows (`num_outbound_cmds` and `is_host_login`, both all zeros) and were removed.
3. **Min-Max Rescaling:** All remaining columns were scaled to $[0, 1]$ via $x \mapsto (x - x_{\min}) / (x_{\max} - x_{\min})$, stored in memory as `float32`.
4. **Final Data Matrix:** $\mathbf{D} \in \mathbb{R}^{494021 \times 116}$ ($n = 494,021$, $d = 116$).
5. **Target Vector:** Binary vector $\mathbf{y} \in \{0, 1\}^{494021}$ (97,278 normal connections vs. 396,743 network attacks).

---

## 🔬 Core Implementations & Methodology

### 1. Question 1: Random Projections for K-Means Clustering
* **Gaussian Projection Matrix:** $\mathbf{A} \in \mathbb{R}^{m \times d}$ with entries $A_{ij} \sim \mathcal{N}(0, 1/m)$ drawn independently.
* **Consistent Float64 Cost Function:** All clustering costs $\text{cost}(\mathbf{D}, \mathbf{B}) = \sum_{i=1}^n \min_{\mathbf{b} \in \mathbf{B}} \|\mathbf{D}_i - \mathbf{b}\|_2^2$ are computed using a chunked `float64` implementation to eliminate single-precision floating-point discrepancies.
* **Reconstruction Strategies Benchmark ($m \in \{5, 10, 15, 20, 25\}$, 5 repetitions):**
  * **$\hat{\mathbf{B}}$ (B-hat):** Data points are assigned to clusters in the projected space $\mathbb{R}^m$, then cluster centers are recomputed as true means in original $\mathbb{R}^d$:
    $$\hat{\mathbf{b}}_j = \frac{1}{|C_j|} \sum_{i \in C_j} \mathbf{D}_i$$
  * **$\mathbf{B}^+$ (B-plus):** Cluster centers found in projected space $\mathbb{R}^m$ are mapped back using the Moore-Penrose pseudoinverse:
    $$\mathbf{B}^+ = \mathbf{A}^+ \mathbf{a}$$
* **Distance vs. Cost Preservation:** Compares pairwise distance ratios $\|\mathbf{E}_i - \mathbf{E}_l\|_2^2 / \|\mathbf{D}_i - \mathbf{D}_l\|_2^2$ across 10,000 non-identical pairs against fixed-clustering cost preservation $\text{cost}_{\mathbf{E}}(C_{\mathbf{D}}) / \text{cost}_{\mathbf{D}}(C_{\mathbf{D}})$ across 20 draws of $\mathbf{A}$.

---

### 2. Question 2: Randomized Linear Algebra for Linear Regression
* **Full-Data Baseline:** Solves $\min_{\mathbf{b}} \|\mathbf{D}\mathbf{b} - \mathbf{y}\|_2^2$ on the full dataset using `scipy.linalg.lstsq` / `numpy.linalg.lstsq` to obtain baseline loss $\text{Loss}(\mathbf{D}, \mathbf{y}, \mathbf{b}) = \|\mathbf{D}\mathbf{b} - \mathbf{y}\|_2^2$ and numerical rank ($\text{rank}(\mathbf{D}) = 113$).
* **CountSketch Implementation:** Implements ultra-sparse streaming sketch matrix $\mathbf{S} \in \mathbb{R}^{m \times n}$ without materializing dense matrices:
  * Hash function $h: [n] \rightarrow [m]$ and random Rademacher signs $s: [n] \rightarrow \{-1, +1\}$.
  * In-place row accumulation via `np.add.at` in $O(\text{nnz}(\mathbf{D}))$ time.
  * Evaluated across $m \in \{50, 100, 200, 500, 1000, 2000, 5000\}$ over 5 repetitions.
* **Norm Preservation & Variance Decay:** Evaluates $\|\mathbf{S}\mathbf{v}\|_2^2 / \|\mathbf{v}\|_2^2$ across 100 independent CountSketches for $m \in \{50, 500, 5000\}$ on the full residual vector $\mathbf{v} = \mathbf{D}\mathbf{b} - \mathbf{y}$.
* **CountSketch vs. Dense Gaussian Sketch (50k Subset):** Evaluates sketching time and solution accuracy on a 50,000-row subset ($m = 500$) and provides asymptotic FLOP/memory comparisons for the full dataset.

---

## 📈 Key Findings & Experimental Results

### Numerical Results Summary Table

| Question / Metric | Dimension / Parameter | Code Output & Markdown Text | Finding / Interpretation |
| :--- | :--- | :--- | :--- |
| **Q1(a) Baseline Cost** | Full $\mathbb{R}^{116}$, $k=15$ | **53,500.91** (Time: **22.67 s**) | Float32 inertia cross-check is 53,474.09 (0.05% difference due to internal float32 rounding). |
| **Q1(b) $\hat{\mathbf{B}}$ Cost Ratio** | $m = 5 \rightarrow 20 \rightarrow 25$ | **1.178** $\rightarrow$ **1.003** $\rightarrow$ **1.010** | Monotonically decays toward 1.0 as $m$ grows. $m=25$ ratio is within 5-run sampling noise. |
| **Q1(b) Sub-1.0 Ratios** | $m = 20, 25$ | **0.9630** (lowest), **0.9688**, **0.9821** | Lloyd's algorithm on $\mathbb{R}^{116}$ finds a local minimum; projected initialization discovers a better partition. |
| **Q1(c) $\mathbf{B}^+$ Cost Ratio** | $m \in \{5, 10, 15, 20, 25\}$ | **64.82** to **80.73** ($64\times\text{–}81\times$ higher) | **Failure of $\mathbf{B}^+$:** $\mathbf{A}^+ \mathbf{A}$ projects onto $m$ dimensions, completely losing $d - m \ge 91$ dimensions. |
| **Q1(d) $k$-Means Speedup** | $m = 5 \rightarrow 25$ | **4.96 s** $\rightarrow$ **7.76 s** | **$2.9\times$ to $4.6\times$ faster** than clustering directly on full $\mathbb{R}^{116}$ (22.67 s). |
| **Q1(e) Variance Spread** | Fixed-Cost vs Pairwise | $\sigma_{\text{cost}} \in [\mathbf{0.076}, \mathbf{0.144}]$ vs $\sigma_{\text{pair}} \in [\mathbf{0.192}, \mathbf{0.526}]$ | Fixed-clustering cost aggregates across 494,021 points, canceling directional variance ($O(1/\sqrt{m})$ decay). |
| **Q2(a) Full Regression** | Full $\mathbf{D}$, $d = 116$ | Loss: **2949.79**, Rank: **113**, Time: **9.19 s** | Numerical rank is 113 (3 linearly dependent columns). |
| **Q2(b) Underdetermined** | $m < 113$ ($m = 50, 100$) | Sketched Loss: **0.0000**, Full Loss: **3.80**, **16.83** | Underdetermined system interpolates sketched points but overfits severely (peak ratio **23.62** at $m=100$). |
| **Q2(b) Overdetermined** | $m \ge 200 \rightarrow 5000$ | Loss Ratio: **2.28** $\rightarrow$ **1.28** $\rightarrow$ **1.025** | Loss ratio monotonically decays to near 1.0 as $m$ increases beyond the rank of $\mathbf{D}$. |
| **Q2(c) Regression Speedup** | $m \in [50, 5000]$ | Runtime: **~1.62 s to ~2.13 s** | **$4.3\times$ to $5.7\times$ faster** than full least-squares (9.19 s); runtime is dominated by single streaming pass. |
| **Q2(d) Norm Preservation** | $m = 50, 500, 5000$ (100 draws) | Mean Ratio: **1.0454**, **1.0043**, **0.9979** | Confirms CountSketch is an **unbiased estimator** ($\mathbb{E}[\|\mathbf{S}\mathbf{v}\|_2^2] = \|\mathbf{v}\|_2^2$). |
| **Q2(d) Spread Decay** | $m = 50 \rightarrow 500 \rightarrow 5000$ | Std: **0.2073** $\rightarrow$ **0.0589** $\rightarrow$ **0.0197** | Standard deviation drops by $\approx 3.52$ and $2.99$, closely matching theoretical $\sqrt{10} \approx 3.16$ ($O(1/\sqrt{m})$). |
| **Q2(e) 50k Subset Solve** | CountSketch vs Gaussian ($m=500$) | Sketch Time: **0.154 s** vs **0.669 s** | CountSketch is **$>4.3\times$ faster** with identical solution accuracy (loss ratio **1.2761** vs **1.2774**). |
| **Q2(e) Scalability to Full $\mathbf{D}$** | Dense Gaussian on $n=494,021$ | FLOPs: $\approx \mathbf{5.73 \times 10^{10}}$, RAM: $\approx \mathbf{1.98\text{ GB}}$ | Dense Gaussian requires materializing a 1.98 GB matrix; CountSketch streams in $O(\text{nnz}(\mathbf{D}))$ with $<4\text{ MB}$. |

---

## 🚀 Execution & Reproduction Guide

### Prerequisites
Install the required packages in Python 3.10+:
```bash
pip install numpy scipy pandas scikit-learn matplotlib
```

### Running the Notebook
Open the notebook in Jupyter and run all cells:
```bash
jupyter notebook MT26155-MT26166-DSA1-Jupyter-File.ipynb
```
* The dataset will download automatically on first run via `scikit-learn.datasets.fetch_kddcup99` into `./data/`.
* All random projections, clusterings, regressions, and plots will execute deterministically with fixed seeds.

---

## 📜 Declarations & Academic Integrity

* **Collaboration Statement:** All discussions and analytical work were conducted strictly and internally between group members **Atharv Goel** (`MT26155`) and **Ishwaani Batra** (`MT26166`).
* **External URLs Consulted:**
  * [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  * [NumPy Least-Squares Solver (`numpy.linalg.lstsq`)](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
  * [NumPy In-Place Addition (`numpy.add.at`)](https://numpy.org/doc/stable/reference/generated/numpy.add.at.html)
  * [KDD Cup 1999 Dataset (`fetch_kddcup99`)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)

### AI Disclosure
AI-assisted tools were used as supporting resources during the assignment for programming assistance, technical clarification, and documentation. The work was reviewed with reference to the assignment requirements and the experimental outputs.
