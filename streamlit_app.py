import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from reconcile import reconcile

load_dotenv()

st.set_page_config(page_title="Settlement Reconciliation Agent", page_icon="💰", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.2) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


st.title("💰 Settlement Reconciliation Agent")

tab_dashboard, tab_txn, tab_pdf = st.tabs(["📊 Dashboard", "💳 Ask About a Transaction", "📄 PDF Policy Chat"])


# ---------- TAB 1: Reconciliation Dashboard ----------

with tab_dashboard:
    st.subheader("Reconciliation Results")

    if st.button("Run Reconciliation"):
        with st.spinner("Comparing settlement report against bank statement..."):
            result = reconcile()
        st.session_state["recon_result"] = result

    result = st.session_state.get("recon_result")

    if result:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", result.total_records)
        col2.metric("Matched", result.matched)
        col3.metric("Match Rate", f"{result.match_rate}%")

        st.divider()
        st.subheader(f"Exceptions ({len(result.exceptions)})")

        if result.exceptions:
            st.dataframe(
                [
                    {
                        "Txn ID": e.txn_id,
                        "Status": e.status,
                        "Settlement Amount": e.settlement_amount,
                        "Bank Amount": e.bank_amount,
                        "Reason": e.reason,
                    }
                    for e in result.exceptions
                ],
                use_container_width=True,
            )
        else:
            st.success("No exceptions — everything matched.")
    else:
        st.info("Click 'Run Reconciliation' to see results.")


# ---------- TAB 2: Transaction Q&A ----------

with tab_txn:
    st.subheader("Ask about a specific transaction")

    async def send_txn_query_event(txn_id: str, question: str) -> str:
        client = get_inngest_client()
        result = await client.send(
            inngest.Event(
                name="rag/query_txn_status",
                data={"txn_id": txn_id, "question": question},
            )
        )
        return result[0]

    with st.form("txn_query_form"):
        txn_id = st.text_input("Transaction ID (e.g. txn_a1b2c3d4e5)")
        question = st.text_input("Your question", value="Why is this transaction showing this status?")
        submitted = st.form_submit_button("Ask")

        if submitted and txn_id.strip():
            with st.spinner("Looking up transaction and generating answer..."):
                event_id = asyncio.run(send_txn_query_event(txn_id.strip(), question.strip()))
                output = wait_for_run_output(event_id)

            st.subheader("Status")
            st.write(output.get("status", "unknown"))

            st.subheader("Answer")
            st.write(output.get("answer", "(No answer)"))

            sources = output.get("sources", [])
            if sources:
                st.caption("Policy sources")
                for s in sources:
                    st.write(f"- {s}")


# ---------- TAB 3: Original PDF Chat (unchanged behavior) ----------

with tab_pdf:
    def save_uploaded_pdf(file) -> Path:
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(parents=True, exist_ok=True)
        file_path = uploads_dir / file.name
        file_bytes = file.getbuffer()
        file_path.write_bytes(file_bytes)
        return file_path

    async def send_rag_ingest_event(pdf_path: Path) -> None:
        client = get_inngest_client()
        await client.send(
            inngest.Event(
                name="rag/ingest_pdf",
                data={"pdf_path": str(pdf_path.resolve()), "source_id": pdf_path.name},
            )
        )

    st.subheader("Upload a policy PDF")
    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)

    if uploaded is not None:
        with st.spinner("Uploading and triggering ingestion..."):
            path = save_uploaded_pdf(uploaded)
            asyncio.run(send_rag_ingest_event(path))
            time.sleep(0.3)
        st.success(f"Triggered ingestion for: {path.name}")
        st.caption("You can upload another PDF if you like.")

    st.divider()
    st.subheader("Ask a question about your policy PDFs")

    async def send_rag_query_event(question: str, top_k: int) -> str:
        client = get_inngest_client()
        result = await client.send(
            inngest.Event(
                name="rag/query_pdf_ai",
                data={"question": question, "top_k": top_k},
            )
        )
        return result[0]

    with st.form("rag_query_form"):
        pdf_question = st.text_input("Your question")
        top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
        submitted = st.form_submit_button("Ask")

        if submitted and pdf_question.strip():
            with st.spinner("Sending event and generating answer..."):
                event_id = asyncio.run(send_rag_query_event(pdf_question.strip(), int(top_k)))
                output = wait_for_run_output(event_id)
                answer = output.get("answer", "")
                sources = output.get("sources", [])

            st.subheader("Answer")
            st.write(answer or "(No answer)")

            if sources:
                st.caption("Sources")
                for s in sources:
                    st.write(f"- {s}")
