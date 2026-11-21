# CSE558: Data Science — Assignment 1
## Dimensionality Reduction and Random Projections for K-Means and Linear Regression

### Course & Submission Metadata
* **Course:** CSE558: Data Science
* **Assignment Title:** Assignment 1 — Dimensionality Reduction & Randomized Linear Algebra
* **Group Members:**
  * **Atharv Goel** (Roll Number: `MT26155`)
  * **Ishwaani Batra** (Roll Number: `MT26166`)
* **Submission File:** `MT26155-MT26166-DSA1.zip`
* **Report Deliverable:** `MT26155-MT26166-DSA1.pdf` (Answers & Visualizations, 7 pages)
* **Notebook Deliverable:** `CSE558_Assignment_1.ipynb` (Clean, top-to-bottom executed)

---

## 1. Project Overview

This repository implements, evaluates, and analyzes randomized dimensionality reduction and sketching algorithms applied to high-dimensional large-scale data, using the default 10% subset of the **KDD Cup 1999** dataset ($n = 494,021$ connection records):

1. **Preprocessing & Featurization:**
   - 3 categorical features (`protocol_type`, `service`, `flag`) transformed into 80 one-hot binary columns.
   - Combined with 38 numerical features $\rightarrow 118$ raw columns.
   - Dropped 2 constant zero columns (`num_outbound_cmds`, `is_host_login`).
   - Scaled all columns to $[0, 1]$ via min-max normalization, yielding final matrix $\mathbf{D} \in \mathbb{R}^{494021 \times 116}$ ($d = 116$).
   - Target vector $\mathbf{y} \in \{0, 1\}^{494021}$ (97,278 normal connections vs. 396,743 network attacks).

2. **Question 1: Random Projections for K-Means Clustering**
   - Implements Gaussian random projection matrix $\mathbf{A} \in \mathbb{R}^{m \times d}$ with $A_{ij} \sim \mathcal{N}(0, 1/m)$.
   - Evaluates baseline $k$-means on the full 116-dimensional space using a numerically consistent chunked float64 squared distance cost function.
   - Benchmarks two reconstruction strategies across $m \in \{5, 10, 15, 20, 25\}$ over 5 independent repetitions:
     - **$\hat{\mathbf{B}}$ (B-hat):** Recomputing true cluster means in the original space $\mathbb{R}^d$ from projected cluster assignments.
     - **$\mathbf{B}^+$ (B-plus):** Mapping low-dimensional cluster centroids back via the pseudoinverse $\mathbf{A}^+ \mathbf{a}$.
   - Evaluates pairwise distance preservation (Johnson-Lindenstrauss lemma) versus fixed-clustering cost preservation across 10,000 non-identical pairs.

3. **Question 2: Randomized Linear Algebra for Least-Squares Regression**
   - Solves full least-squares regression $\min_{\mathbf{b}} \|\mathbf{D}\mathbf{b} - \mathbf{y}\|_2^2$ to find baseline loss and dataset rank ($\text{rank}(\mathbf{D}) = 113$).
   - Implements ultra-sparse streaming **CountSketch** $\mathbf{S} \in \mathbb{R}^{m \times n}$ using in-place hashing (`np.add.at`) across $m \in \{50, 100, 200, 500, 1000, 2000, 5000\}$ over 5 independent runs.
   - Analyzes the structural transition below rank ($m < 113$, exact interpolation with zero sketched loss and severe full-loss spikes) versus above rank ($m \ge 200$, rapid monotonic decay to $1.025$).
   - Studies empirical norm preservation $\mathbb{E}[\|\mathbf{S}\mathbf{v}\|_2^2] = \|\mathbf{v}\|_2^2$ over 100 independent sketches per $m \in \{50, 500, 5000\}$ and verifies $O(1/\sqrt{m})$ standard deviation decay.
   - Directly compares CountSketch against dense Gaussian sketching on a 50,000-row subset ($m = 500$), demonstrating $>4.3\times$ speedup with identical subspace accuracy, and provides memory/FLOP scaling proofs for full $\mathbf{D}$.

---



---

## 3. Environment & Prerequisites

The assignment is executed in Python 3.10+ with standard scientific computing packages:

```bash
pip install numpy scipy pandas scikit-learn matplotlib beautifulsoup4 nbconvert
```

### Reproducibility & Fixed Seeds
To guarantee 100% deterministic replication across runs, all random generators use explicitly pinned seeds:
* Master random seed: `SEED = 42`
* $k$-means random initialization: `random_state = 42` with `n_init = 5`
* Random projection seeds for Q1: `seed = 42 + rep * 1000 + m`
* CountSketch hash seeds for Q2: `seed = 42 + rep * 10000 + m`
* Norm preservation seeds for Q2(d): `seed = 42 + i * 100`

---

## 4. How to Run & Reproduce

### 1. Run the Jupyter Notebook Top-to-Bottom
Launch Jupyter Lab or Notebook and run all cells sequentially from a fresh kernel:
```bash
jupyter notebook CSE558_Assignment_1.ipynb
```
*(All cell execution outputs, display tables, and high-resolution matplotlib figures will regenerate deterministically.)*

### 2. Regenerate the HTML & PDF Report
To rebuild `MT26155-MT26166-DSA1.pdf` directly from the executed notebook without code cells:
```bash
python generate_final_report.py
```
This script:
1. Runs `jupyter nbconvert --to html --no-input CSE558_Assignment_1.ipynb`
2. Formats all 11 tables and 4 figures with clean styling.
3. Formats wide tables with proper line breaks at natural underscore boundaries to prevent clipping.
4. Generates a balanced 7-page PDF via Chromium / Edge headless:
```powershell
Start-Process "msedge.exe" -ArgumentList @("--headless", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf=`"MT26155-MT26166-DSA1.pdf`"", "`"file:///$(pwd)/MT26155-MT26166-DSA1.html`"") -Wait
```

### 3. Rebuild Submission Archive
```powershell
Compress-Archive -Path "MT26155-MT26166-DSA1.pdf", "CSE558_Assignment_1.ipynb", "README.md" -DestinationPath "MT26155-MT26166-DSA1.zip" -Force
```

---

## 5. Summary of Experimental Results

### Question 1: Random Projections for K-Means Clustering
* **Baseline Cost ($d = 116$):** $\text{cost}(\mathbf{D}, \mathbf{B}) = \mathbf{53,500.91}$ (float64 chunked distance function). Scikit-learn float32 inertia cross-check is $\mathbf{53,474.09}$ (0.05% numerical difference due to internal float32 rounding). Computation time: $\mathbf{22.67\text{ s}}$.
* **$\hat{\mathbf{B}}$ (B-hat) Quality:** Mean cost ratio $\text{cost}(\mathbf{D}, \hat{\mathbf{B}}) / \text{cost}(\mathbf{D}, \mathbf{B})$ monotonically decreases from $\mathbf{1.178}$ ($m=5$) $\rightarrow$ $\mathbf{1.105}$ ($m=10$) $\rightarrow$ $\mathbf{1.018}$ ($m=15$) $\rightarrow$ $\mathbf{1.003}$ ($m=20$). At $m=25$, ratio is $\mathbf{1.010}$ (minor sample fluctuation across 5 runs).
* **Ratios Below 1.0:** In several runs at $m=20$ and $m=25$, ratios fall strictly below 1.0 (down to $\mathbf{0.9630}$). This occurs because Lloyd's algorithm on high-dimensional $\mathbb{R}^{116}$ gets trapped in suboptimal local minima; clustering on the low-dimensional projection explores an alternative optimization path whose recomputed original-space cluster means achieve a superior clustering cost.
* **Why Choose $\hat{\mathbf{B}}$ over $\mathbf{B}^+$:** $\hat{\mathbf{B}}$ produces near-optimal costs ($\sim 1.003\text{–}1.178$). In contrast, $\mathbf{B}^+ = \mathbf{A}^+ \mathbf{a}$ produces disastrous cost ratios of $\mathbf{64.82}$ to $\mathbf{80.73}$ because $\mathbf{A}^+ \mathbf{A}$ projects onto the $m$-dimensional row space of $\mathbf{A}$, completely discarding $d - m \ge 91$ orthogonal dimensions of data variation.
* **Speedup:** Projection + $k$-means takes $\mathbf{4.96\text{ s}}$ ($m=5$) to $\mathbf{7.76\text{ s}}$ ($m=25$), delivering a $\mathbf{2.9\times\text{ to }4.6\times\text{ speedup}}$ over clustering on the full 116-dimensional dataset.
* **Spread Comparison (1(e)):** Fixed-clustering cost ratios exhibit a vastly tighter spread ($\sigma = 0.1444$ down to $0.0763$) than pairwise distance ratios ($\sigma = 0.5261$ down to $0.1918$), because the clustering cost aggregates squared differences across all 494,021 points, canceling directional variance.

---

### Question 2: Randomized Linear Algebra for Linear Regression
* **Full Least-Squares Baseline:** Loss $\mathbf{2949.79}$, computation time $\mathbf{9.19\text{ s}}$, matrix numerical rank $\text{rank}(\mathbf{D}) = \mathbf{113}$ (3 redundant columns).
* **Behavior Below Rank ($m < 113$):** Underdetermined sketched system produces sketched loss of exactly $\mathbf{0.0000}$ via `np.linalg.lstsq`. However, the resulting minimum-norm interpolator severely overfits the sketched subspace, leading to huge full-dataset loss ratios ($\mathbf{3.80}$ at $m=50$, spiking to a peak of $\mathbf{16.83}$ mean / $\mathbf{23.62}$ max at $m=100 \approx \text{rank}(\mathbf{D})$).
* **Behavior Above Rank ($m \ge 200$):** Sketched loss rises toward baseline ($\mathbf{1345.71}$ at $m=200 \rightarrow \mathbf{2890.27}$ at $m=5000$). Full-data loss ratio rapidly drops toward 1: $\mathbf{2.28}$ ($m=200$) $\rightarrow$ $\mathbf{1.28}$ ($m=500$) $\rightarrow$ $\mathbf{1.13}$ ($m=1000$) $\rightarrow$ $\mathbf{1.06}$ ($m=2000$) $\rightarrow$ $\mathbf{1.025}$ ($m=5000$).
* **Execution Time (2(c)):** CountSketch runtime is remarkably flat across $m$ ($\mathbf{\sim 1.62\text{ to }2.13\text{ s}}$), providing a $\mathbf{4.3\times\text{ to }5.7\times\text{ speedup}}$ over full least squares.
* **Residual Norm Preservation (2(d)):** Over 100 independent sketches, mean ratio $\|\mathbf{S}\mathbf{v}\|_2^2 / \|\mathbf{v}\|_2^2$ is $\mathbf{1.0454}$ ($m=50$), $\mathbf{1.0043}$ ($m=500$), and $\mathbf{0.9979}$ ($m=5000$), empirically proving unbiasedness ($\mathbb{E}[\|\mathbf{S}\mathbf{v}\|_2^2] = \|\mathbf{v}\|_2^2$). The standard deviation decays as $\mathbf{O(1/\sqrt{m})}$ ($0.2073 \rightarrow 0.0589 \rightarrow 0.0197$, decaying by factors of $\approx 3.52$ and $2.99$ vs $\sqrt{10} \approx 3.16$).
* **CountSketch vs Gaussian Sketch (2(e)):** On the 50,000-row subset ($m=500$), CountSketch forms the sketch **over $4.3\times$ faster** than the Gaussian sketch ($\mathbf{0.1538\text{ s}}$ vs $\mathbf{0.6685\text{ s}}$) while achieving identical accuracy (loss ratio $\mathbf{1.2761}$ vs $\mathbf{1.2774}$). On full $\mathbf{D}$, dense Gaussian sketching is intractable, requiring $\approx 5.73 \times 10^{10}$ FLOPs and $\approx 1.98\text{ GB}$ purely to store $\mathbf{S}$, whereas CountSketch streams in $O(\text{nnz}(\mathbf{D}))$ time with $<4\text{ MB}$ overhead.

---

## 6. Academic Integrity & Acknowledgments

* **Group Discussions:** All discussions were strictly internal between group members Atharv Goel and Ishwaani Batra.
* **Documentation & References Consulted:**
  * [Scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
  * [NumPy Least-Squares Solver (`linalg.lstsq`)](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
  * [NumPy In-Place Addition (`add.at`)](https://numpy.org/doc/stable/reference/generated/numpy.add.at.html)
  * [KDD Cup 1999 Dataset (`fetch_kddcup99`)](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_kddcup99.html)

### AI Disclosure Statement
AI-assisted tools were used as supporting resources during the assignment for programming assistance, technical clarification, and documentation. The work was reviewed with reference to the assignment requirements and the experimental outputs.
