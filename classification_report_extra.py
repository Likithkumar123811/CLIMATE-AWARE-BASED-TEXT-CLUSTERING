import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def generate_report(csv_path="output/final_results.csv"):

    df = pd.read_csv(csv_path)

    y_true = df["Category"]
    y_pred = df["final_category"]

    # ==========================================================
    # TABLE 1: OVERALL METRICS
    # ==========================================================
    overall_table = pd.DataFrame({
        "Metric": ["Accuracy", "Precision (Macro)", "Recall (Macro)", "F1-Score (Macro)"],
        "Value": [
            round(accuracy_score(y_true, y_pred), 4),
            round(precision_score(y_true, y_pred, average="macro", zero_division=0), 4),
            round(recall_score(y_true, y_pred, average="macro", zero_division=0), 4),
            round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)
        ]
    })

    overall_table.to_csv("output/table_1_overall_metrics.csv", index=False)

    # ==========================================================
    # TABLE 2: PER-CLASS PERFORMANCE
    # ==========================================================
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0
    )

    per_class_table = pd.DataFrame(report).transpose().reset_index()
    per_class_table.rename(columns={"index": "Class"}, inplace=True)
    per_class_table = per_class_table.round(4)

    per_class_table.to_csv("output/table_2_per_class_performance.csv", index=False)

    # ==========================================================
    # TABLE 3: CONFUSION MATRIX
    # ==========================================================
    labels = sorted(pd.unique(pd.concat([y_true, y_pred])))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_table = pd.DataFrame(cm, index=labels, columns=labels)

    cm_table.to_csv("output/table_3_confusion_matrix.csv")

    # ==========================================================
    # TABLE 4: ERROR ANALYSIS
    # ==========================================================
    errors = df[df["Category"] != df["final_category"]]

    error_table = errors.groupby(["Category", "final_category"]) \
                        .size() \
                        .reset_index(name="Count") \
                        .sort_values(by="Count", ascending=False) \
                        .head(10)

    error_table.columns = ["Actual Class", "Predicted Class", "Misclassified Samples"]

    error_table.to_csv("output/table_4_error_analysis.csv", index=False)

    # ==========================================================
    # TABLE 5: DISTRIBUTION + ACCURACY
    # ==========================================================
    df["correct"] = df["Category"] == df["final_category"]

    distribution = df["Category"].value_counts().reset_index()
    distribution.columns = ["Class", "Total Samples"]

    accuracy = df.groupby("Category")["correct"].mean().reset_index()
    accuracy.columns = ["Class", "Accuracy"]

    combined = pd.merge(distribution, accuracy, on="Class")
    combined["Accuracy"] = combined["Accuracy"].round(4)

    combined.to_csv("output/table_5_distribution_accuracy.csv", index=False)

    print("\n✅ Research tables generated successfully!")