"""
SubFraudGMM — Fraud Detection in Brazilian Public Procurement via Gaussian Mixture Models

Algorithm overview:
    1. For each equipment dataset and anomaly threshold, generate all non-empty subsets
       of the 8 financial/procurement features.
    2. For each feature subset, fit the best Gaussian Mixture Model (selected by BIC
       over a grid of component counts and covariance types).
    3. Retain only subsets where at least one GMM cluster concentrates fraud cases above
       the threshold; validate each subset with Leave-One-Out cross-validation over the
       known fraud instances.
    4. Aggregate results across all subsets: count how many subsets flagged each record
       as suspicious, compute the Euclidean distance from every record to the centroid
       of confirmed fraud cases.
    5. Save an intermediate CSV per (equipment, threshold) combination for downstream
       analysis and benchmark comparison.
"""

import os
import copy
import time
import glob
import warnings
import concurrent.futures
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_COMPONENTS_RANGE = range(1, 7)
CV_TYPES = ["spherical", "tied", "diag", "full"]

FEATURE_COLUMNS = [
    "duration",
    "unique",
    "win",
    "period",
    "num",
    "met",
    "num_partic",
    "unit_price",
]

EQUIPMENT_DATASETS = [
    "rolo-compactador",
    "escavadeira",
    "trator-esteira",
    "motoniveladora",
]

DEFAULT_THRESHOLDS = [50, 60, 70, 80, 90, 100]

OUTPUT_DIR = "results/intermediary"


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------


def generate_subsets(columns):
    """Generate all non-empty feature subsets (ordered largest to smallest).

    Parameters
    ----------
    columns : list of str
        Feature column names to combine.

    Returns
    -------
    list of tuple
        All non-empty combinations of *columns*, from full set down to singletons.
    """
    subsets = []
    for i in range(len(columns), 0, -1):
        subsets.extend(combinations(columns, i))
    return subsets


def fit_best_gmm(X):
    """Fit GMM, select best by BIC over a grid of component counts and covariance types.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (already scaled).

    Returns
    -------
    GaussianMixture
        The fitted GMM with the lowest Bayesian Information Criterion score.
    """
    lowest_bic = np.infty
    best_gmm = None
    for cv_type in CV_TYPES:
        for n_components in N_COMPONENTS_RANGE:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=cv_type,
                random_state=42,
            )
            gmm.fit(X)
            bic = gmm.bic(X)
            if bic < lowest_bic:
                lowest_bic = bic
                best_gmm = gmm
    return best_gmm


def evaluate_model_loo_fraud_only(df, X, best_gmm, threshold):
    """LOO model validation — check that a held-out fraud is clustered with other frauds.

    For each known fraud record, temporarily remove it from the training set, refit the
    GMM, and verify that the predicted cluster for the held-out record still contains at
    least *threshold* % of all remaining fraud cases.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with a ``fraude`` column (1 = fraud, 0 = non-fraud).
    X : pd.DataFrame
        Scaled feature matrix aligned with *df*.
    best_gmm : GaussianMixture
        Pre-fitted GMM to use as the starting model structure.
    threshold : float
        Minimum percentage of fraud cases required in the predicted cluster.

    Returns
    -------
    consistency_percentage : float
        Fraction of LOO folds where the held-out fraud was clustered with others.
    fraud_cluster_consistency : int
        Raw count of successful LOO folds.
    """
    fraud_indices = df[df["fraude"] == 1].index
    loo = LeaveOneOut()
    fraud_cluster_consistency = 0

    for train_index, test_index in loo.split(X.loc[fraud_indices]):
        train_index_full = df.index.difference([fraud_indices[test_index[0]]])
        X_train = X.loc[train_index_full]
        df_train = df.loc[train_index_full]

        X_test = X.loc[fraud_indices[test_index]]

        best_gmm.fit(X_train)
        labels_gmm = best_gmm.predict(X_train)
        df_train["Cluster_GMM_loo"] = labels_gmm

        test_cluster = best_gmm.predict(X_test)

        grouped = df_train.groupby("Cluster_GMM_loo")
        fraude_count = grouped["fraude"].apply(lambda x: (x == 1).sum())
        total_fraud_count = fraude_count.sum()
        proporcao_fraude = ((fraude_count / total_fraud_count) * 100).round(2)

        if proporcao_fraude.get(test_cluster[0], 0) >= threshold:
            fraud_cluster_consistency += 1

    consistency_percentage = fraud_cluster_consistency / loo.get_n_splits(
        X.loc[fraud_indices]
    )
    return consistency_percentage, fraud_cluster_consistency


def evaluate_model(df, X, best_gmm, threshold):
    """Apply the GMM model and compute fraud cluster metrics.

    Fits the GMM on *X*, identifies clusters where the proportion of known fraud cases
    exceeds *threshold*, and runs LOO validation on those clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset with a ``fraude`` column.
    X : pd.DataFrame
        Scaled feature matrix.
    best_gmm : GaussianMixture
        GMM model (will be refit in place).
    threshold : float
        Minimum fraud-proportion (%) for a cluster to be flagged.

    Returns
    -------
    tuple or None
        ``(consistency_percentage, result_df, df_labeled,
           list_possible_fraud, list_fraud, fraud_cluster_consistency)``
        Returns ``None`` when no cluster meets the threshold or data is insufficient.
    """
    if X.drop_duplicates().shape[0] <= best_gmm.n_components:
        return None

    labels_gmm = best_gmm.fit_predict(X)
    df["Cluster_GMM"] = labels_gmm

    grouped = df.groupby("Cluster_GMM")
    fraude_count = grouped["fraude"].apply(lambda x: (x == 1).sum())
    not_fraude_count = grouped["fraude"].apply(lambda x: (x == 0).sum())
    total_fraud_count = fraude_count.sum()
    total_count = grouped.size()
    proporcao_fraude_total = ((fraude_count / total_count) * 100).round(2)
    proporcao_fraude = ((fraude_count / total_fraud_count) * 100).round(2)
    media_proporcao_conjunta = (proporcao_fraude_total + proporcao_fraude) / 2

    high_fraud_clusters = proporcao_fraude[proporcao_fraude >= threshold].index

    if high_fraud_clusters.empty:
        return None

    list_possible_fraud = []
    list_fraud = []

    for cluster_id in high_fraud_clusters:
        indices_possible_fraud = df[
            (df["Cluster_GMM"] == cluster_id) & (df["fraude"] == 0)
        ].index
        indices_fraud = df[
            (df["Cluster_GMM"] == cluster_id) & (df["fraude"] == 1)
        ].index
        list_possible_fraud.extend(indices_possible_fraud)
        list_fraud.extend(indices_fraud)

    consistency_percentage, fraud_cluster_consistency = evaluate_model_loo_fraud_only(
        df.copy(), X, best_gmm, threshold
    )

    result = pd.DataFrame(
        {
            "Total_Registros": total_count,
            "Not_Fraude_Count": not_fraude_count,
            "Fraude_Count": fraude_count,
            "Proporcao_Fraude_Total (%)": proporcao_fraude_total,
            "Proporcao_Fraude (%)": proporcao_fraude,
            "Média": media_proporcao_conjunta,
        }
    )

    return (
        consistency_percentage,
        result,
        df,
        list_possible_fraud,
        list_fraud,
        fraud_cluster_consistency,
    )


def evaluate_subset(df, subset, threshold):
    """Evaluate a single feature subset: fit GMM and return results if threshold is met.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset (already scaled).
    subset : tuple of str
        Feature names for this subset.
    threshold : float
        Fraud-proportion threshold.

    Returns
    -------
    tuple or None
        Full result tuple ``(subset, loo, result, list_possible_fraud,
        list_fraud, df_label, fraud_cluster_consistency)`` or ``None``.
    """
    X = df[list(subset)]
    best_gmm = fit_best_gmm(X)
    df_copy = copy.deepcopy(df)
    evaluated = evaluate_model(df_copy, X, best_gmm, threshold)
    if evaluated is not None:
        loo, result, df_label, list_possible_fraud, list_fraud, fraud_cluster_consistency = evaluated
        return (subset, loo, result, list_possible_fraud, list_fraud, df_label, fraud_cluster_consistency)
    return None


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def count_index_occurrences(best_subsets):
    """Count how many subsets flagged each record index as suspicious.

    Parameters
    ----------
    best_subsets : list of tuple
        Output from :func:`process_data` or :func:`process_data_parallel`.

    Returns
    -------
    dict
        Mapping ``{index: count}`` sorted by index.
    """
    index_counts = {}
    for _, _, _, list_possible_fraud, _, _, _ in best_subsets:
        for index in list_possible_fraud:
            index_counts[index] = index_counts.get(index, 0) + 1
    return {key: index_counts[key] for key in sorted(index_counts.keys())}


def combine_indices(index_lists, method="union"):
    """Combine index lists using set intersection or union.

    Parameters
    ----------
    index_lists : list of set or list
        Collection of index sets to combine.
    method : str, optional
        ``'union'`` (default) or ``'intersection'``.

    Returns
    -------
    list
        Combined index list.
    """
    if not index_lists:
        return []
    combined_indices = set(index_lists[0])
    for index_set in index_lists[1:]:
        combined_indices = getattr(combined_indices, method)(index_set)
    return list(combined_indices)


def calculate_euclidean_distances(row, df_fraud, features):
    """Compute mean Euclidean distance from *row* to all rows in *df_fraud*.

    Parameters
    ----------
    row : pd.Series
        A single record from the dataset.
    df_fraud : pd.DataFrame
        Confirmed fraud records used as reference.
    features : list of str
        Feature columns to use for distance computation.

    Returns
    -------
    float
        Mean distance from *row* to all fraud records.
    """
    return df_fraud[features].apply(lambda x: euclidean(x, row[features]), axis=1).mean()


def create_intermediate_dataframe(df, df_fraud, best_subsets, equipment, threshold, fraud_id=None):
    """Build and save the intermediate results CSV for one (equipment, threshold) run.

    For each flagged subset, assembles a DataFrame of suspicious and confirmed fraud
    records, annotates it with feature set, LOO score, and Euclidean distance to the
    fraud centroid, then writes to ``OUTPUT_DIR``.

    Parameters
    ----------
    df : pd.DataFrame
        Full (scaled) dataset for this equipment.
    df_fraud : pd.DataFrame
        Records identified as confirmed fraud (union across subsets).
    best_subsets : list of tuple
        Subset results from :func:`process_data` or :func:`process_data_parallel`.
    equipment : str
        Equipment name used in the output filename.
    threshold : int or float
        Threshold value used in the output filename.
    fraud_id : int or None, optional
        If not ``None``, appended to filename (LOO fold identifier).
    """
    dfs = []

    for subset in best_subsets:
        features = subset[0]
        loo = subset[1]
        fraud_cluster_consistency = subset[6]
        index = subset[3]
        index_fraud = subset[4]
        combined_indices = index + index_fraud
        features_str = ", ".join(features)

        df_temp = df.loc[combined_indices].copy()
        df_temp["features"] = features_str
        df_temp["loo"] = loo
        df_temp["num_fraud_reclustered"] = fraud_cluster_consistency
        dfs.append(df_temp)

    if not dfs:
        return None

    df_copy = pd.concat(dfs)
    df_copy["euclidean_distance_to_fraud"] = df_copy.apply(
        lambda row: calculate_euclidean_distances(
            row, df_fraud, row["features"].split(", ")
        ),
        axis=1,
    )

    if fraud_id is None:
        output_csv_path = f"{OUTPUT_DIR}/{equipment}_{threshold}.csv"
    else:
        output_csv_path = f"{OUTPUT_DIR}/{equipment}_{threshold}_{fraud_id}.csv"

    df_copy.to_csv(output_csv_path, index=False)


# ---------------------------------------------------------------------------
# Pipeline entry points
# ---------------------------------------------------------------------------


def process_data(df, columns, threshold):
    """GMM entry point, build suspect lists — sequential version.

    Iterates over all feature subsets, fits the best GMM for each, and collects
    results for subsets that meet the fraud-concentration threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled dataset.
    columns : list of str
        Feature columns to subset.
    threshold : float
        Fraud-proportion threshold.

    Returns
    -------
    list of tuple
        One entry per qualifying subset.
    """
    subsets = generate_subsets(columns)
    best_subsets = []
    for subset in subsets:
        X = df[list(subset)]
        best_gmm = fit_best_gmm(X)
        df_copy = copy.deepcopy(df)
        evaluated = evaluate_model(df_copy, X, best_gmm, threshold)
        if evaluated is not None:
            loo, result, df_label, list_possible_fraud, list_fraud, fraud_cluster_consistency = evaluated
            best_subsets.append(
                (subset, loo, result, list_possible_fraud, list_fraud, df_label, fraud_cluster_consistency)
            )
    return best_subsets


def process_data_parallel(df, columns, threshold):
    """GMM entry point, build suspect lists — parallel version.

    Same as :func:`process_data` but distributes subset evaluation across CPU cores
    using :class:`concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled dataset.
    columns : list of str
        Feature columns to subset.
    threshold : float
        Fraud-proportion threshold.

    Returns
    -------
    list of tuple
        One entry per qualifying subset (order not guaranteed).
    """
    subsets = generate_subsets(columns)
    best_subsets = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Submit one task per subset
        futures = {
            executor.submit(evaluate_subset, df, subset, threshold): subset
            for subset in subsets
        }
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                best_subsets.append(result)

    return best_subsets


def main(df, columns, threshold, equipment, fraud_id=None):
    """Main pipeline — sequential execution.

    Normalizes features, runs GMM across all subsets, counts occurrences, computes
    Euclidean distances, and writes the intermediate CSV.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (unscaled) dataset for one equipment type.
    columns : list of str
        Feature columns.
    threshold : float
        Fraud-proportion threshold.
    equipment : str
        Equipment name for output filename.
    fraud_id : int or None, optional
        LOO fold fraud identifier; ``None`` for the full-data run.

    Returns
    -------
    pd.DataFrame
        Dataset annotated with ``ocorrencias`` and ``porcentagem_ocorrencias``.
    """
    df = df.copy()

    # Normalize features
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])

    # Run GMM on all subsets
    best_subsets = process_data(df, columns, threshold)

    # Occurrence counting
    index_counts = count_index_occurrences(best_subsets)
    df["ocorrencias"] = df.index.map(index_counts.get).fillna(0)
    df["porcentagem_ocorrencias"] = df["ocorrencias"] / len(best_subsets) * 100

    indices_list_fraud = [set(subset[4]) for subset in best_subsets]
    common_indices_fraud = combine_indices(indices_list_fraud)
    df_fraud = df.loc[common_indices_fraud].copy()

    create_intermediate_dataframe(df, df_fraud, best_subsets, equipment, threshold, fraud_id)

    return df


def main_parallel(df, columns, threshold, equipment, fraud_id=None):
    """Main pipeline — parallel execution.

    Same as :func:`main` but uses :func:`process_data_parallel` for subset evaluation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw (unscaled) dataset for one equipment type.
    columns : list of str
        Feature columns.
    threshold : float
        Fraud-proportion threshold.
    equipment : str
        Equipment name for output filename.
    fraud_id : int or None, optional
        LOO fold fraud identifier.

    Returns
    -------
    pd.DataFrame
        Dataset annotated with ``ocorrencias`` and ``porcentagem_ocorrencias``.
    """
    df = df.copy()
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])

    best_subsets = process_data_parallel(df, columns, threshold)

    # Occurrence counting
    index_counts = count_index_occurrences(best_subsets)
    df["ocorrencias"] = df.index.map(index_counts.get).fillna(0)
    df["porcentagem_ocorrencias"] = df["ocorrencias"] / len(best_subsets) * 100

    indices_list_fraud = [set(subset[4]) for subset in best_subsets]
    common_indices_fraud = combine_indices(indices_list_fraud)
    df_fraud = df.loc[common_indices_fraud].copy()

    create_intermediate_dataframe(df, df_fraud, best_subsets, equipment, threshold, fraud_id)

    return df


# ---------------------------------------------------------------------------
# Task wrapper
# ---------------------------------------------------------------------------


def process_main_task(args):
    """Process one (equipment, threshold[, fraud_id]) task and return timing metadata.

    Intended to be submitted to a :class:`concurrent.futures.ProcessPoolExecutor`.

    Parameters
    ----------
    args : tuple
        ``(df_name, df, threshold, fraud_id)`` where *fraud_id* is ``None`` for the
        full-data run or an integer fraud ID for LOO fold runs.

    Returns
    -------
    dict
        Keys: ``df_name``, ``threshold``, ``fraud_id``, ``task_type``,
        ``elapsed_seconds``, ``result_message``.
    """
    df_name, df, threshold, fraud_id = args
    start_time = time.perf_counter()

    if fraud_id is None:
        task_type = "main dataset"
        main(df, FEATURE_COLUMNS, threshold, df_name)
        result_msg = f"Completed main dataset: {df_name} with threshold {threshold}"
    else:
        task_type = "fraud LOO fold"
        main(df, FEATURE_COLUMNS, threshold, df_name, fraud_id)
        result_msg = (
            f"Completed LOO fold for fraud {fraud_id} — {df_name}, threshold {threshold}"
        )

    elapsed = time.perf_counter() - start_time

    return {
        "df_name": df_name,
        "threshold": threshold,
        "fraud_id": fraud_id,
        "task_type": task_type,
        "elapsed_seconds": elapsed,
        "result_message": result_msg,
    }


if __name__ == "__main__":
    pass
