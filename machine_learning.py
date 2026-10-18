"""scExpand: donor-aware binary classification of expanded expression features.

Run: python machine_learning.py --input expanded.csv --output results/run1
Demo: python machine_learning.py --demo --output results/demo
Source and citation: https://github.com/WeiJiang-ZJU/scExpand
"""

import argparse
import csv
import json
import platform
import tempfile
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn import discriminant_analysis, ensemble, linear_model, neighbors, svm, tree
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


METADATA = {"Outcome", "Patient", "Time", "sample", "pseudo_sample_id"}


def load_table(path, timepoint=None, feature_file=None, expected_features=None):
    """Read a CSV without changing expression values or inferring clinical labels."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle))
    if len(set(header)) != len(header):
        raise ValueError("Duplicate CSV column names are not supported.")
    frame = pd.read_csv(path, dtype={"Patient": str, "Time": str, "sample": str})
    # R write.csv() includes an unnamed row-name column unless row.names=FALSE.
    if len(frame.columns) and frame.columns[0] == "Unnamed: 0":
        frame = frame.drop(columns=frame.columns[0])
    if not {"Outcome", "Patient"}.issubset(frame):
        raise ValueError("The CSV must contain Outcome and Patient.")
    if "Time" not in frame:
        frame["Time"] = "single"
    if frame[["Outcome", "Patient", "Time"]].isna().any().any():
        raise ValueError("Outcome, Patient, and Time must not be missing.")
    if frame["Patient"].str.strip().eq("").any():
        raise ValueError("Patient IDs must not be empty.")
    mapping = {"ALI": 1, "non-ALI": 0, "1": 1, "0": 0}
    labels = frame["Outcome"].astype(str).map(mapping)
    if labels.isna().any():
        raise ValueError("Outcome must be ALI/non-ALI or binary 0/1.")
    frame["Outcome"] = labels.astype(int)
    if frame.groupby("Patient")["Outcome"].nunique().gt(1).any():
        raise ValueError("Each Patient must have one consistent outcome.")
    if timepoint is None:
        times = frame["Time"].unique()
        if len(times) != 1:
            raise ValueError("Multiple timepoints found; select one using --timepoint.")
        timepoint = times[0]
    frame = frame.loc[frame["Time"].eq(timepoint)].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows found for Time={timepoint!r}.")
    if "sample" in frame:
        if frame["sample"].isna().any():
            raise ValueError("sample must not be missing when provided.")
        if frame.groupby("Patient")["sample"].nunique().gt(1).any():
            raise ValueError("Use one original sample per Patient per timepoint.")
        if frame.groupby("sample")["Patient"].nunique().gt(1).any():
            raise ValueError("Each sample must belong to one Patient.")
    if "pseudo_sample_id" in frame and (
        frame["pseudo_sample_id"].isna().any() or frame["pseudo_sample_id"].duplicated().any()
    ):
        raise ValueError("pseudo_sample_id must be nonmissing and unique.")
    if feature_file:
        features = [line.strip() for line in Path(feature_file).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
    else:
        features = [c for c in frame if c not in METADATA]
    if not features or len(features) != len(set(features)):
        raise ValueError("The feature list must be nonempty and unique.")
    if set(features) & METADATA or not set(features).issubset(frame):
        raise ValueError("Feature names must identify expression columns in the CSV.")
    if expected_features is not None and len(features) != expected_features:
        raise ValueError(f"Expected {expected_features} features, found {len(features)}.")
    frame[features] = frame[features].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
        raise ValueError("Feature values must be finite; no automatic imputation is applied.")
    if frame["Outcome"].nunique() != 2:
        raise ValueError("Both outcome classes are required.")
    return frame, features


def donor_table(frame):
    """Return one outcome per donor, in stable donor-ID order."""
    if frame.groupby("Patient")["Outcome"].nunique().gt(1).any():
        raise ValueError("Each donor must have one outcome.")
    return frame[["Patient", "Outcome"]].drop_duplicates().sort_values("Patient")


def donor_holdout(frame, test_size=0.3, seed=24):
    """Stratify unique donors, then map their IDs back to pseudo-sample rows."""
    donors = donor_table(frame)
    if donors["Outcome"].value_counts().min() < 2:
        raise ValueError("At least two donors per class are needed for a holdout split.")
    train_ids, test_ids = train_test_split(
        donors["Patient"], test_size=test_size, random_state=seed,
        stratify=donors["Outcome"],
    )
    train = np.flatnonzero(frame["Patient"].isin(train_ids))
    test = np.flatnonzero(frame["Patient"].isin(test_ids))
    for idx in (train, test):
        if frame.iloc[idx]["Outcome"].nunique() != 2:
            raise ValueError("Both classes must be present in each donor partition.")
    return train, test


def donor_folds(frame, n_splits=5, seed=24):
    """Stratified K-fold on unique donors; returned indices refer to frame rows."""
    donors = donor_table(frame).reset_index(drop=True)
    counts = donors["Outcome"].value_counts()
    if len(counts) != 2 or counts.min() < n_splits or n_splits < 2:
        raise ValueError(f"Need at least {n_splits} donors per class for {n_splits} folds.")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train, valid in splitter.split(donors["Patient"], donors["Outcome"]):
        train_ids = donors.iloc[train]["Patient"]
        valid_ids = donors.iloc[valid]["Patient"]
        folds.append((np.flatnonzero(frame["Patient"].isin(train_ids)),
                      np.flatnonzero(frame["Patient"].isin(valid_ids))))
    return folds


CANDIDATES = (
    "AdaBoostClassifier", "ExtraTreesClassifier", "GradientBoostingClassifier",
    "RandomForestClassifier", "LogisticRegressionCV", "PassiveAggressiveClassifier",
    "SGDClassifier", "Perceptron", "KNeighborsClassifier", "SVC", "LinearSVC",
    "DecisionTreeClassifier", "ExtraTreeClassifier", "LinearDiscriminantAnalysis",
    "QuadraticDiscriminantAnalysis",
)


def build_model(name, training_frame, seed=24, inner_folds=3):
    """Build a fresh estimator; LR internal folds refer only to this training set.

    No additional normalization is fitted here: input features are assumed to
    have already undergone appropriate, study-specific preprocessing.
    """
    factories = {
        "AdaBoostClassifier": lambda: ensemble.AdaBoostClassifier(random_state=seed),
        "ExtraTreesClassifier": lambda: ensemble.ExtraTreesClassifier(random_state=seed, n_jobs=1),
        "GradientBoostingClassifier": lambda: ensemble.GradientBoostingClassifier(random_state=seed),
        "RandomForestClassifier": lambda: ensemble.RandomForestClassifier(random_state=seed, n_jobs=1),
        "PassiveAggressiveClassifier": lambda: linear_model.PassiveAggressiveClassifier(random_state=seed),
        "SGDClassifier": lambda: linear_model.SGDClassifier(random_state=seed),
        "Perceptron": lambda: linear_model.Perceptron(random_state=seed),
        "KNeighborsClassifier": neighbors.KNeighborsClassifier,
        # Use margins to avoid SVC's additional, row-wise probability calibration.
        "SVC": lambda: svm.SVC(probability=False, random_state=seed),
        "LinearSVC": lambda: svm.LinearSVC(random_state=seed),
        "DecisionTreeClassifier": lambda: tree.DecisionTreeClassifier(random_state=seed),
        "ExtraTreeClassifier": lambda: tree.ExtraTreeClassifier(random_state=seed),
        "LinearDiscriminantAnalysis": discriminant_analysis.LinearDiscriminantAnalysis,
        "QuadraticDiscriminantAnalysis": discriminant_analysis.QuadraticDiscriminantAnalysis,
    }
    if name == "LogisticRegressionCV":
        cv = donor_folds(training_frame, inner_folds, seed)
        return linear_model.LogisticRegressionCV(
            Cs=10, cv=cv, scoring="roc_auc", solver="lbfgs", penalty="l2",
            max_iter=2000, tol=0.001, random_state=seed, n_jobs=1,
        )
    if name not in factories:
        raise ValueError(f"Unknown candidate: {name}")
    return factories[name]()


def continuous_scores(model, X):
    """Return scores for the positive class (1) and their type; margins are not probabilities."""
    if hasattr(model, "predict_proba"):
        positive = np.flatnonzero(model.classes_ == 1)
        if len(positive) != 1:
            raise ValueError("The fitted model must contain the positive class 1.")
        scores = model.predict_proba(X)[:, positive[0]]
        kind = "probability"
    else:
        scores = model.decision_function(X)
        kind = "decision_function"
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Model scores must be finite and one-dimensional.")
    return scores, kind


def aggregate_donors(frame, scores):
    """Mean score across pseudo-samples, one row per original donor."""
    if len(scores) != len(frame):
        raise ValueError("Each input row needs one score.")
    scored = frame[["Patient", "Outcome"]].copy()
    scored["score"] = np.asarray(scores)
    return scored.groupby("Patient", as_index=False, sort=True).agg(
        Outcome=("Outcome", "first"), score=("score", "mean"),
        n_pseudo_samples=("score", "size"),
    )


def _warning_text(captured):
    return " | ".join(sorted({f"{w.category.__name__}: {w.message}" for w in captured}))


def _plots(donors, threshold, output):
    # Select a noninteractive backend so the CLI also works on servers.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    )

    y, scores = donors["Outcome"], donors["score"]
    for name, draw in (
        ("roc", lambda ax: RocCurveDisplay.from_predictions(y, scores, ax=ax)),
        ("precision_recall", lambda ax: PrecisionRecallDisplay.from_predictions(y, scores, ax=ax)),
        ("confusion_matrix", lambda ax: ConfusionMatrixDisplay.from_predictions(
            y, (scores >= threshold).astype(int), labels=[0, 1],
            display_labels=["Class 0", "Class 1"], ax=ax, cmap="Blues", colorbar=False)),
    ):
        fig, ax = plt.subplots(figsize=(6, 5))
        draw(ax)
        ax.set_title("Held-out donors")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=200)
        plt.close(fig)


def run_analysis(input_csv, timepoint, output_dir, feature_file=None,
                 expected_features=None, models=None, seed=24, test_size=0.3,
                 cv_folds=5, inner_folds=3, threshold=None, plots=True):
    """Select a model using training-donor CV, then evaluate the holdout once.

    Fixed features must be defined independently of held-out outcomes. This
    function cannot undo leakage from upstream feature selection or integration.
    """
    frame, features = load_table(input_csv, timepoint, feature_file, expected_features)
    timepoint = str(frame["Time"].iloc[0])
    names = list(CANDIDATES if models is None else models)
    if not names or len(set(names)) != len(names) or not set(names).issubset(CANDIDATES):
        raise ValueError("Choose one or more unique supported candidate models.")
    if threshold is not None and not np.isfinite(threshold):
        raise ValueError("threshold must be finite.")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing analysis runs.")
    train_idx, heldout_idx = donor_holdout(frame, test_size, seed)
    training = frame.iloc[train_idx].reset_index(drop=True)
    heldout = frame.iloc[heldout_idx].reset_index(drop=True)
    folds = donor_folds(training, cv_folds, seed)
    # Validate all LR inner partitions before starting a potentially long run.
    if "LogisticRegressionCV" in names:
        for fit_idx, _ in folds:
            donor_folds(training.iloc[fit_idx].reset_index(drop=True), inner_folds, seed)
        donor_folds(training, inner_folds, seed)
    output.mkdir(parents=True, exist_ok=True)
    partition = donor_table(frame).copy()
    partition["partition"] = np.where(
        partition["Patient"].isin(training["Patient"]), "train", "holdout")
    partition.to_csv(output / "donor_split.csv", index=False)
    fold_manifest = []
    for fold_id, (fit_idx, valid_idx) in enumerate(folds, start=1):
        for label, idx in (("fit", fit_idx), ("validation", valid_idx)):
            d = donor_table(training.iloc[idx]).copy()
            d["fold"] = fold_id
            d["partition"] = label
            fold_manifest.append(d)
    pd.concat(fold_manifest).to_csv(output / "cv_donor_splits.csv", index=False)

    comparisons = []
    fold_results = []
    for name in names:
        aucs = []
        captured = []
        error = ""
        try:
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                for fold_id, (fit_idx, valid_idx) in enumerate(folds, start=1):
                    fit = training.iloc[fit_idx].reset_index(drop=True)
                    valid = training.iloc[valid_idx].reset_index(drop=True)
                    model = build_model(name, fit, seed, inner_folds)
                    model.fit(fit[features], fit["Outcome"])
                    scores, _ = continuous_scores(model, valid[features])
                    donors = aggregate_donors(valid, scores)
                    auc = float(roc_auc_score(donors["Outcome"], donors["score"]))
                    aucs.append(auc)
                    fold_results.append({"model": name, "fold": fold_id, "donor_auc": auc})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        comparisons.append({
            "model": name, "status": "failed" if error else "ok",
            "cv_donor_auc_mean": float(np.mean(aucs)) if not error else np.nan,
            "cv_donor_auc_sd": float(np.std(aucs, ddof=1)) if not error else np.nan,
            "completed_folds": len(aucs), "warnings": _warning_text(captured), "error": error,
        })
    comparison = pd.DataFrame(comparisons).sort_values(
        ["cv_donor_auc_mean", "model"], ascending=[False, True], na_position="last")
    comparison.to_csv(output / "cv_comparison.csv", index=False)
    pd.DataFrame(fold_results).to_csv(output / "cv_fold_metrics.csv", index=False)
    successful = comparison.loc[comparison["status"].eq("ok")]
    if successful.empty:
        raise RuntimeError("All candidates failed. Inspect cv_comparison.csv.")
    selected = successful.iloc[0]["model"]
    model = build_model(selected, training, seed, inner_folds)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(training[features], training["Outcome"])
        scores, kind = continuous_scores(model, heldout[features])
    final_warnings = _warning_text(captured)
    donors = aggregate_donors(heldout, scores)
    cutoff = float(threshold if threshold is not None else (0.5 if kind == "probability" else 0.0))
    if kind == "probability" and not 0 <= cutoff <= 1:
        raise ValueError("A probability threshold must lie in [0, 1].")
    predictions = (donors["score"] >= cutoff).astype(int)
    y = donors["Outcome"]
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    metrics = {
        "selected_model": selected, "selection": "highest mean training CV donor AUC; ties by model name",
        "timepoint": timepoint, "score_kind": kind, "threshold": cutoff,
        "threshold_source": "user_prespecified" if threshold is not None else "fixed_default",
        "n_features": len(features), "n_training_donors": int(training["Patient"].nunique()),
        "n_holdout_donors": len(donors), "n_training_pseudo_samples": len(training),
        "n_holdout_pseudo_samples": len(heldout),
        "donor_auc": float(roc_auc_score(y, donors["score"])),
        "donor_average_precision": float(average_precision_score(y, donors["score"])),
        "donor_accuracy": float(accuracy_score(y, predictions)),
        "donor_sensitivity": float(recall_score(y, predictions, zero_division=0)),
        "donor_specificity": float(tn / (tn + fp)),
        "donor_precision": float(precision_score(y, predictions, zero_division=0)),
        "donor_confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "pseudo_sample_auc_exploratory": float(roc_auc_score(heldout["Outcome"], scores)),
        "fit_warnings": final_warnings,
    }
    donors["prediction"] = predictions
    donors.to_csv(output / "holdout_donor_predictions.csv", index=False)
    (output / "holdout_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    config = {
        "seed": seed, "test_size": test_size, "cv_folds": cv_folds,
        "inner_folds": inner_folds, "models": names, "timepoint": timepoint,
        "features": features, "expected_features": expected_features,
        "threshold": cutoff, "score_kind": kind,
        "versions": {p: version(p) for p in ["numpy", "pandas", "scikit-learn", "scipy", "joblib", "matplotlib"]},
        "python": platform.python_version(),
        "selected_parameters": model.get_params(),
    }
    # LR cv stores row index arrays; serialize them explicitly for reproducibility.
    def json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Cannot serialize {type(obj).__name__}")
    (output / "run_config.json").write_text(json.dumps(config, indent=2, default=json_default) + "\n")
    joblib.dump({"model": model, "features": features, "timepoint": timepoint,
                 "threshold": cutoff, "score_kind": kind}, output / "model.joblib")
    if plots:
        _plots(donors, cutoff, output)
    return metrics


def make_demo_table(path, seed=24, timepoint="demo"):
    """Write artificial expanded features; no clinical or sequencing data are used."""
    rng = np.random.default_rng(seed)
    rows = []
    for donor in range(32):
        outcome = donor % 2
        center = rng.normal(0, 0.8, 6) + outcome * 0.5
        for replicate in range(8):
            values = center + rng.normal(0, 0.25, 6)
            row = {f"cluster1__feature{j + 1}": value for j, value in enumerate(values)}
            row.update(Outcome=outcome, Patient=f"SIM{donor:03d}", Time=timepoint,
                       sample=f"SIM{donor:03d}_{timepoint}",
                       pseudo_sample_id=f"SIM{donor:03d}_{timepoint}_{replicate:03d}")
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return Path(path)


def main():
    parser = argparse.ArgumentParser(description="scExpand donor-aware binary classification")
    parser.add_argument("--input", help="Expanded feature CSV with Outcome and Patient")
    parser.add_argument("--demo", action="store_true", help="Run an artificial example instead of reading study data")
    parser.add_argument("--timepoint", help="Value of Time to analyze; optional for single-timepoint data")
    parser.add_argument("--output", required=True, help="Empty output directory")
    parser.add_argument("--features", help="Optional text file: one feature name per line")
    parser.add_argument("--models", nargs="+", choices=CANDIDATES, help="Default: all 15 candidates")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--threshold", type=float, help="Prespecified threshold; default 0.5 probability / 0 margin")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if bool(args.input) == args.demo:
        parser.error("Provide exactly one of --input and --demo.")
    if args.demo and args.features:
        parser.error("--features is intended for user-supplied input, not --demo.")

    def analyze(path):
        return run_analysis(
            path, args.timepoint, args.output, feature_file=args.features,
            models=args.models, seed=args.seed, test_size=args.test_size,
            cv_folds=args.cv_folds, inner_folds=args.inner_folds,
            threshold=args.threshold, plots=not args.no_plots,
        )

    if args.demo:
        print("Artificial demonstration only; no clinical or sequencing data were used.")
        with tempfile.TemporaryDirectory(prefix="scexpand_demo_") as directory:
            path = make_demo_table(Path(directory) / "demo.csv", args.seed,
                                   args.timepoint or "demo")
            metrics = analyze(path)
    else:
        metrics = analyze(args.input)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
