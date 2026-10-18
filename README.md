# scExpand

**Stratified pseudo-sampling for single-cell studies with few biological samples.**

scExpand repeatedly samples cells within each annotated cluster of each original
sample, averages their expression values, and concatenates the cluster-specific
means into a sample-level feature vector. Each vector retains the source sample's
label. The accompanying Python code uses these vectors for binary classification.

Repository: [WeiJiang-ZJU/scExpand](https://github.com/WeiJiang-ZJU/scExpand)

## Applications

scExpand is intended for settings with few independent biological samples but many
cells and expression features. It was developed for a CyTOF study of acute lung
injury around cardiopulmonary bypass, but its sampling procedure is not specific
to CyTOF, a disease, or a fixed marker panel.

| Data type | Input features | Cluster annotations |
| --- | --- | --- |
| CyTOF / flow cytometry | Preprocessed protein-marker expression | Cell populations or cell states |
| Single-cell RNA sequencing | Appropriately normalized gene expression; a selected gene set may be used | Cell types or transcriptional states |
| CITE-seq and related single-cell assays | An appropriately processed RNA or protein feature matrix | Shared cell types or states |

For scRNA-seq, the same cluster-wise averaging can be applied to genes instead
of protein markers. Use a suitable normalized expression layer and a manageable
feature set. The RNA example below demonstrates this interface; it does not
establish a performance improvement on sequencing cohorts.

The method requires multiple measured cells, sample IDs, and cluster labels.
It is not a generic augmentation method for any high-dimensional table that lacks
within-sample observations. Repeated draws may overlap and do not create new
independent donors. Keep all pseudo-samples from one donor together in training,
validation, and testing. This distinction also matters in
[single-cell statistical analysis](https://www.nature.com/articles/s41467-021-21038-1).

## Files

```text
scExpand.R               Expansion functions for matrices and Seurat objects
machine_learning.py      Standalone model comparison and evaluation
machine_learning.ipynb   Notebook interface, without saved outputs
requirements.txt         Python dependencies
README.md                Instructions, applications, and citation
LICENSE                  Short use and citation terms
.gitignore               Excludes local data and generated results
```

No manuscript-specific feature lists, clinical measurements, trained study models,
or historical results are required or included.

## 1. Expand single-cell measurements in R

The original algorithm was implemented in R 4.4.1. Matrix input needs base R;
Seurat input additionally needs `SeuratObject` (normally installed with Seurat).

Required cell metadata:

| Column | Meaning |
| --- | --- |
| `sample` | Original biological sample ID |
| `cluster` | Cell-type or cell-state annotation |
| `Patient` | Donor or other independent biological unit; the name is retained for compatibility |
| `Outcome` | Source sample's outcome; use 0/1 for generic Python classification |
| `Time` | Optional timepoint; defaults to `single` if omitted |

Each sample must have one donor, outcome, and timepoint. Cell IDs must match
between the expression columns and metadata row names.

```r
source("scExpand.R")

# CyTOF: use the prepared expression layer from your analysis.
expanded <- expand_samples(
  seurat_obj, k = 0.2, A = 100, seed = 24,
  assay = "integrated", layer = "scale.data"
)
write.csv(expanded, "expanded.csv", row.names = FALSE)
```

For an already normalized and annotated scRNA-seq Seurat object:

```r
# Set sample, cluster, Patient, and Outcome in rna_obj's cell metadata first.
# normalized_gene_names is an optional, user-defined set of genes present in data.
expanded_rna <- expand_samples(
  rna_obj, k = 0.2, A = 100, seed = 24,
  assay = "RNA", layer = "data"
)
write.csv(expanded_rna, "expanded_rna.csv", row.names = FALSE)

# Optional: restrict expression features before expansion.
# expanded_rna <- expand_samples(rna_obj, assay = "RNA", layer = "data",
#                                markers = normalized_gene_names)
```

`data` contains normalized expression and `scale.data` contains scaled expression;
choose the layer appropriate for your analysis. Seurat v5 layers split across
samples must be joined before extracting a single shared matrix. See the
[Seurat layer documentation](https://satijalab.org/seurat/articles/seurat5_essential_commands.html).
This workflow averages the supplied values; it does not perform normalization,
batch correction, cell annotation, or count-based differential expression.

For other software, pass a numeric feature-by-cell matrix, optionally sparse,
and the metadata directly:

```r
expanded <- expand_matrix(expression_matrix, cell_metadata, k = 0.2, A = 100)
```

For each present cluster, `max(1, floor(k * number_of_cells))` cells are sampled
without replacement per draw. Sampling is repeated `A` times per original sample.
Feature names are `cluster__feature`, and neither component may contain `__`.
Cluster order is shared across samples; absent clusters receive zeros. Zero is
a compatibility convention and must not be interpreted as evidence of biological
absence. NaN/Inf expression and missing required metadata are rejected. Cluster
proportions are not included.

## 2. Train and compare models in Python

Use Python 3.9 or later and install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the artificial example to check the installation:

```bash
python machine_learning.py --demo --output results/demo
```

Then use your expanded table:

```bash
python machine_learning.py --input expanded.csv --output results/my_analysis
```

If the CSV contains several timepoints, select one explicitly:

```bash
python machine_learning.py --input expanded.csv --timepoint CPB_pre --output results/pre
```

Alternatively, open `machine_learning.ipynb`. Its first data cell generates
artificial data; replace that cell's input path for a real analysis.

All nonmetadata expression columns are used by default. An optional
`--features features.txt` selects a subset, one exact column name per line.
The number and identity of features are not hard-coded. The Python workflow
supports binary labels `0`/`1`, plus `non-ALI`/`ALI` for the original application.

The 15 candidates include RandomForest, LogisticRegressionCV, LDA, QDA, AdaBoost,
SVC, DecisionTree, ExtraTrees, GradientBoosting, KNeighbors, LinearSVC, SGD,
PassiveAggressive, Perceptron, and ExtraTree. Use `--models RandomForestClassifier`
to evaluate a prespecified candidate rather than compare all models.

The default holdout fraction is 0.3. Five outer CV folds and three inner LR folds
are formed on unique donors and mapped back to pseudo-samples. Enough donors of
each class are needed in every partition; use `--cv-folds 3` only if appropriate
for your sample size. Donor separation is maintained regardless of expansion size.

Candidate selection uses training-CV donor AUC. Held-out donor scores are the mean
of their pseudo-sample scores. Outputs include AUC, average precision, confusion
matrices, plots, split manifests, fit warnings, and a fitted model in the chosen
output directory. Use a new empty directory for each run. Default classification
thresholds are fixed at 0.5 for probability scores and 0 for decision margins.
SVC margins are not calibrated probabilities.

Feature selection and upstream preprocessing must be designed without consulting
held-out outcomes. The code does not implement feature discovery or guarantee
better prediction from augmentation. This organized model workflow uses donor
CV and fixed thresholds, unlike the original notebook's internal row-wise CV
and held-out threshold selection. It does not reproduce the paper's results.

## Citation and use

If you use the scExpand algorithm, this implementation, or its model workflow,
cite the GitHub repository. No journal article citation is required at present:

> scExpand. GitHub repository: https://github.com/WeiJiang-ZJU/scExpand.
> Include the version or commit used and the access date where possible.

Use is permitted under the short, custom [use and citation terms](LICENSE),
which require citation of this GitHub repository.

## Validation

The code is checked using artificial protein and RNA inputs, including donor
isolation, cluster averaging, sparse matrices, and Seurat RNA assays. Artificial
examples do not establish clinical or sequencing performance.
