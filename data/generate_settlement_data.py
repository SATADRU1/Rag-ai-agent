"""
Generates two linked synthetic datasets to simulate a real settlement mismatch:

1. settlement_report.csv  -> what Razorpay says it settled
2. bank_statement.csv     -> what actually landed in the bank
3. ground_truth.csv       -> the TRUE status of each txn_id (used only for grading, not by the agent)

Some rows are made to intentionally NOT match, on purpose, to test reconciliation.
"""

import csv
import random
import uuid
from datetime import datetime, timedelta

random.seed(42)

NUM_RECORDS = 60

def make_txn_id():
    return "txn_" + uuid.uuid4().hex[:10]

def make_utr():
    return "UTR" + str(random.randint(100000000000, 999999999999))

settlement_rows = []
bank_rows = []
ground_truth_rows = []

base_date = datetime(2026, 1, 1)

for i in range(NUM_RECORDS):
    txn_id = make_txn_id()
    utr = make_utr()
    amount = round(random.uniform(200, 50000), 2)
    settle_date = base_date + timedelta(days=random.randint(0, 60))

    issue = random.choices(
        ["clean", "amount_mismatch", "missing_in_bank", "duplicate", "delayed"],
        weights=[65, 10, 10, 5, 10],
    )[0]

    true_status = "matched" if issue == "clean" else issue

    ground_truth_rows.append({"txn_id": txn_id, "true_status": true_status})

    settlement_rows.append({
        "txn_id": txn_id,
        "utr": utr,
        "amount": amount,
        "settled_date": settle_date.strftime("%Y-%m-%d"),
        "status": "settled",
    })

    if issue == "clean":
        bank_rows.append({"utr": utr, "amount": amount, "credited_date": settle_date.strftime("%Y-%m-%d")})
    elif issue == "amount_mismatch":
        bank_rows.append({"utr": utr, "amount": round(amount - random.uniform(1, 50), 2), "credited_date": settle_date.strftime("%Y-%m-%d")})
    elif issue == "missing_in_bank":
        pass
    elif issue == "duplicate":
        bank_rows.append({"utr": utr, "amount": amount, "credited_date": settle_date.strftime("%Y-%m-%d")})
        bank_rows.append({"utr": utr, "amount": amount, "credited_date": settle_date.strftime("%Y-%m-%d")})
    elif issue == "delayed":
        bank_rows.append({"utr": utr, "amount": amount, "credited_date": (settle_date + timedelta(days=random.randint(3, 10))).strftime("%Y-%m-%d")})

random.shuffle(bank_rows)

with open("data/settlement_report.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["txn_id", "utr", "amount", "settled_date", "status"])
    writer.writeheader()
    writer.writerows(settlement_rows)

with open("data/bank_statement.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["utr", "amount", "credited_date"])
    writer.writeheader()
    writer.writerows(bank_rows)

with open("data/ground_truth.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["txn_id", "true_status"])
    writer.writeheader()
    writer.writerows(ground_truth_rows)

print(f"Generated {len(settlement_rows)} settlement records and {len(bank_rows)} bank records.")
print("Saved to data/settlement_report.csv, data/bank_statement.csv, data/ground_truth.csv")
