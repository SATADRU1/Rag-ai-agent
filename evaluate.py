"""
Grades the reconciliation engine against the true labels we know from
data generation. This is the honest accuracy number for the pitch —
no cherry-picking, every record counted.
"""

import csv
from reconcile import reconcile


def load_ground_truth(path="data/ground_truth.csv"):
    with open(path, newline="") as f:
        return {row["txn_id"]: row["true_status"] for row in csv.DictReader(f)}


def evaluate():
    result = reconcile()
    truth = load_ground_truth()

    # Build a txn_id -> predicted_status map from reconcile output
    all_records = {r.txn_id: r.status for r in result.exceptions}

    # Any txn_id NOT in exceptions was classified as matched
    matched_txn_ids = set(truth.keys()) - set(all_records.keys())
    for txn_id in matched_txn_ids:
        all_records[txn_id] = "matched"

    correct = 0
    wrong = []

    for txn_id, true_status in truth.items():
        predicted = all_records.get(txn_id, "missing_in_bank")
        if predicted == true_status:
            correct += 1
        else:
            wrong.append((txn_id, true_status, predicted))

    accuracy = round(correct / len(truth) * 100, 2)

    print(f"Total records graded : {len(truth)}")
    print(f"Correctly classified : {correct}")
    print(f"Accuracy             : {accuracy}%")
    print(f"Misclassified        : {len(wrong)}")
    for txn_id, true_status, predicted in wrong:
        print(f"  {txn_id} — true: {true_status}, predicted: {predicted}")

    return accuracy, wrong


if __name__ == "__main__":
    evaluate()
