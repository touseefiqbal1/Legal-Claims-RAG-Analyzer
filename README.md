https://legal-claims-rag-analyzer.streamlit.app/

---

## 📄 README — **Motor Claims Court Pack RAG Analyzer (MVP)**

---

# 🧠 Legal Claims RAG Analyzer

A **Retrieval-Augmented Generation (RAG)** system tailored for **legal and insurance claim documents**.
Upload synthetic or real case packs (multi-page PDFs) and ask natural-language questions — get **structured answers with citations** and **PDF page previews**.

This tool demonstrates:
✔️ **Accurate extraction** of key claim fields
✔️ **FAISS vector retrieval with LangChain**
✔️ **Rule-based grounding + answer summarization**
✔️ **Interactive Streamlit UI with PDF page viewer**
✔️ **Evaluation metrics against ground truth**

---

## 🚀 Features

| Feature                                   | Status |
| ----------------------------------------- | ------ |
| Page-level PDF ingestion                  | ✅      |
| FAISS vector index with semantic search   | ✅      |
| Rule-based extraction + structured output | ✅      |
| Pack selector (filter retrieval per case) | ✅      |
| Clickable citations + PDF page viewer     | ✅      |
| Evaluation dashboard (hit@k metrics)      | ✅      |
| Exportable evaluation reports             | ✅      |

---

## 📁 Repository Structure

```
📦 Legal-Claims-RAG-Analyzer
 ┣ 📄 app.py                      # Streamlit UI
 ┣ 📄 evaluate.py                 # Evaluation script
 ┣ 📄 requirements.txt            # Python deps
 ┣ 📂 data/sample_pdfs            # Place your PDF case packs here
 ┣ 📂 indexes/faiss_index         # Persisted FAISS index + manifest
 ┣ 📂 rag                        # RAG modules (ingestion, chunking, QA, extractors)
 ┃ ┣ 📄 ingest_pdf.py
 ┃ ┣ 📄 chunking.py
 ┃ ┣ 📄 index_faiss.py
 ┃ ┣ 📄 qa.py
 ┃ ┣ 📄 extractors.py
 ┗ 📄 README.md
```

---

## 🛠️ Getting Started

### 💡 Requirements

* Python 3.9+
* Windows, Linux, macOS

---

## 🧩 Installation

```bash
git clone https://github.com/touseefiqbal1/Legal-Claims-RAG-Analyzer.git
cd Legal-Claims-RAG-Analyzer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # (Windows) .\venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📄 Step 1 — Prepare Documents

Place your multi-page PDFs in:

```
data/sample_pdfs/
```

If using synthetic packs, they should include corresponding ground-truth JSON.

---

## 🧠 Step 2 — Build / Rebuild FAISS Index

Run the Streamlit app:

```bash
streamlit run app.py
```

In the sidebar → **Build / Rebuild FAISS index**.

This does:

* Load PDFs page-by-page
* Chunk text
* Build vector index for semantic retrieval

---

## ❓ Step 3 — Ask Questions

Use the UI:

* **Select a case pack** (or “All packs”)
* Enter a question:

Examples:

```
What is the claim reference?
What is the total claimed amount?
List the reported injuries.
What fraud indicators were identified?
What is the suggested reserve?
```

The UI shows:

* A rule-based answer
* Extracted fields with supportive citations
* A clickable **view page** button with PDF preview

---

## 📊 Step 4 — Run Evaluation (with Manifest)

If you generated synthetic data with `manifest.json`, you can run evaluation:

1. Ensure path in sidebar → **Manifest path**
   Example:

   ```
   indexes/manifest.json
   ```

2. Run evaluation from sidebar → **Run evaluation**

3. Metrics shown:
   ✔ Hit rate (overall)
   ✔ Per-field hit rates
   ✔ Per-pack hit rates

Evaluation report is auto-saved as:

```
evaluation_report.json
```

---

## 📌 Supported Queries (Preset Examples)

| Field             | Example Question                         |
| ----------------- | ---------------------------------------- |
| Claim reference   | `What is the claim reference?`           |
| Policy number     | `What is the policy number?`             |
| Incident details  | `When and where did the incident occur?` |
| Total claimed     | `What is the total claimed amount?`      |
| Reserve suggested | `What is the suggested reserve?`         |
| Injuries          | `List the reported injuries.`            |
| Fraud flags       | `What fraud indicators were identified?` |

---

## 🧪 How It Works — Simplified

1. **Ingest PDF pages**

   * Metadata contains `source`, `path`, and `page`

2. **Chunk text for FAISS**

   * Splits into semantic chunks

3. **Vector store retrieval**

   * Uses semantic embeddings

4. **Rule-based extraction**

   * Extracts structured fields from retrieved text

5. **UI renders**

   * Answers, citations, PDF page previews

---

## 📌 Notable Modules

| File                 | Responsibility                    |
| -------------------- | --------------------------------- |
| `rag/ingest_pdf.py`  | Load PDFs as page docs            |
| `rag/chunking.py`    | Chunk text w/ overlap             |
| `rag/index_faiss.py` | Build/persist FAISS               |
| `rag/qa.py`          | Retrieval + citations + grounding |
| `rag/extractors.py`  | Regex based field extraction      |
| `app.py`             | Streamlit UI                      |
| `evaluate.py`        | Evaluation script                 |

---

## 📈 Evaluation Explained

The evaluation computes **hit@k** for fields defined in your ground-truth JSON, such as:

* claim_reference
* policy_number
* incident_date/time
* incident_location
* police_reference
* total_claimed
* reserve_recommendation

It fetches **fetch_k** items, filters by pack, and computes whether the correct answer appears in the top-k results.

---

## 🧠 Tips for Better Results

✔ Increase `Top-k` if some fields aren’t found
✔ Ensure PDFs are text-extractable (good OCR)
✔ Build more diverse synthetic cases for evaluation
✔ Tune regex patterns in `extractors.py` for real data

---
