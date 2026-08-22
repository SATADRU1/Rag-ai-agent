"""
Generates a few synthetic settlement policy PDFs so the RAG pipeline
has real policy text to retrieve when answering transaction questions.
"""

from fpdf import FPDF
import os

os.makedirs("uploads", exist_ok=True)

POLICIES = {
    "settlement_timeline_policy.pdf": """
Settlement Timeline Policy

All successful transactions are settled on a T+2 basis, meaning funds are
credited to the merchant's bank account two business days after the
transaction date. Transactions flagged as high-risk may be held for
additional review, extending settlement up to T+5.

If a settlement is delayed beyond T+2 without a corresponding risk flag,
it should be treated as an operational exception and escalated to the
settlements team within 24 hours.
""",
    "amount_mismatch_policy.pdf": """
Amount Mismatch Resolution Policy

An amount mismatch occurs when the amount recorded in the settlement report
does not match the amount credited to the merchant's bank account.

Mismatches under Rs 500 are typically caused by rounding differences or
minor fee adjustments and are auto-corrected within 3 business days.

Mismatches above Rs 500 must be manually reviewed. The merchant should be
notified within 24 hours, and the case should remain open until resolved
with a matching corrected entry.
""",
    "duplicate_and_missing_policy.pdf": """
Duplicate and Missing Settlement Policy

A duplicate settlement occurs when more than one bank credit is recorded
against the same UTR. Duplicates must be reversed within 5 business days,
and the merchant should not treat duplicate credits as usable funds.

A missing settlement occurs when a transaction marked as settled in the
report has no corresponding bank credit. This is treated as the highest
priority exception and must be investigated within 24 hours, since it may
indicate a failed bank transfer or an incorrect UTR.
""",
}

for filename, text in POLICIES.items():
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    page_width = pdf.w - pdf.l_margin - pdf.r_margin
    for line in text.strip().split("\n"):
        pdf.multi_cell(page_width, 8, line)
    pdf.output(f"uploads/{filename}")
    print(f"Created uploads/{filename}")

print("Done. Upload these via the 'PDF Policy Chat' tab in the Streamlit app to ingest them.")
