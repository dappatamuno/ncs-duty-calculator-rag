# Nigeria Customs Service (NCS) Automated Duty Calculator & Tariff Classifier

![NCS Duty Calculator Preview](assets/dashboard_preview.png)

An AI cognitive agent designed to eliminate HS code misclassifications and calculate exact landing costs for shipments entering Nigerian ports (Apapa, Tin Can, etc.) using the official ECOWAS Common External Tariff (CET).

## The Solution & Business Impact
Manual HS code classification leads to extensive delays, arbitrary Nigeria Customs Service (NCS) penalties, and demographic cargo detention. A 10–20% duty variance on a ₦200M shipment amounts to a loss of ₦20–40M. 

This terminal acts as an autonomous compliance assistant. It accepts raw, unstructured product specifications, matches them semantically against the official CET tariff schedules using an offline vector database, runs deterministic financial calculations in Naira (NGN), and flags mandatory regulatory pre-arrival clearings (NAFDAC / SONCAP).

## Tech Stack
- **UI Framework:** Streamlit (Custom enterprise theme)
- **Orchestration & Retrieval:** LangChain
- **LLM Engine:** Llama 3.3 (70B) via Groq Cloud API
- **Vector Database:** ChromaDB (Persistent local deployment)
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Runs locally)
- **Financial & Regulatory Engine:** Deterministic Python Math Logic
- **Report Engine:** FPDF2

## Project Architecture
1. **Semantic Ingestion:** Downloads the official compiled `CET_tariff_v2.xls` directly from the NCS database, tokenizes rows, generates dense vector embeddings, and builds an offline vector index.
2. **Context Retrieval:** Uses vector semantic distance to look up relevant parts of the tariff book based on conversational descriptions (e.g., mapping "mazda car" to Chapter 87).
3. **Structured Extraction:** Uses schema-enforced JSON parsing to isolate the precise 8-digit code, base duty rates, and statutory regulatory requirements.
4. **Deterministic Calculation:** Executes official formulas for Form M processing: Import Duty, Surcharge (7% of ID), CISS (1% of FOB), ETLS (0.5% of CIF), and VAT (7.5%).

## Local Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone [https://github.com/dappatamuno/ncs-duty-calculator-rag.git]
   cd dappatamuno