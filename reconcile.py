"""
Reconciliation engine.

Compares Razorpay's settlement report against the bank statement and
classifies every record as matched, or flags exactly why it isn't.
"""

import csv
from custom_type import ReconcileRecord, ReconcileResult

AMOUNT_TOLERANCE = 0.01   # rupees — treat tiny rounding differences as a match
SETTLEMENT_WINDOW_DAYS = 2  # Razorpay's usual T+2 settlement promise


def _load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _days_between(d1: str, d2: str) -> int:
    from datetime import datetime
    fmt = "%Y-%m-%d"
    return abs((datetime.strptime(d2, fmt) - datetime.strptime(d1, fmt)).days)


def reconcile(settlement_path="data/settlement_report.csv", bank_path="data/bank_statement.csv") -> ReconcileResult:
    settlement_rows = _load_csv(settlement_path)
    bank_rows = _load_csv(bank_path)

    # Group bank rows by UTR, since duplicates mean more than one row can share a UTR
    bank_by_utr = {}
    for row in bank_rows:
        bank_by_utr.setdefault(row["utr"], []).append(row)

    records = []
    for s in settlement_rows:
        utr = s["utr"]
        settlement_amount = float(s["amount"])
        candidates = bank_by_utr.get(utr, [])

        if not candidates:
            records.append(ReconcileRecord(
                txn_id=s["txn_id"], utr=utr,
                settlement_amount=settlement_amount,
                settled_date=s["settled_date"],
                status="missing_in_bank",
                reason="No matching credit found in bank statement for this UTR.",
            ))
            continue

        # Use the first candidate as the primary match
        match = candidates[0]
        bank_amount = float(match["amount"])
        credited_date = match["credited_date"]

        if len(candidates) > 1:
            status, reason = "duplicate", f"Found {len(candidates)} bank credits for the same UTR — likely duplicate."
        elif abs(bank_amount - settlement_amount) > AMOUNT_TOLERANCE:
            status, reason = "amount_mismatch", f"Settlement says ₹{settlement_amount}, bank shows ₹{bank_amount}."
        elif _days_between(s["settled_date"], credited_date) > SETTLEMENT_WINDOW_DAYS:
            status, reason = "delayed", f"Credited {_days_between(s['settled_date'], credited_date)} days after settlement — beyond T+{SETTLEMENT_WINDOW_DAYS}."
        else:
            status, reason = "matched", "Amount and timing align with settlement report."

        records.append(ReconcileRecord(
            txn_id=s["txn_id"], utr=utr,
            settlement_amount=settlement_amount, bank_amount=bank_amount,
            settled_date=s["settled_date"], credited_date=credited_date,
            status=status, reason=reason,
        ))

    matched_count = sum(1 for r in records if r.status == "matched")
    exceptions = [r for r in records if r.status != "matched"]

    return ReconcileResult(
        total_records=len(records),
        matched=matched_count,
        match_rate=round(matched_count / len(records) * 100, 2) if records else 0.0,
        exceptions=exceptions,
    )


if __name__ == "__main__":
    result = reconcile()
    print(f"Total records : {result.total_records}")
    print(f"Matched       : {result.matched} ({result.match_rate}%)")
    print(f"Exceptions    : {len(result.exceptions)}")
    for e in result.exceptions:
        print(f"  [{e.status}] {e.txn_id} — {e.reason}")
