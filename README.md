# Yukti AI – Intelligent Customer Service Chatbot

Yukti AI is a production‑grade, multi‑modal conversational AI platform built with Streamlit. It integrates dynamic knowledge base expansion, multi‑modal generation (text, image, video, audio), a specialised medical Q&A module, domain expertise from the arXiv dataset, sentiment analysis, and multi‑language support (100+ languages including Hinglish). The system is designed for scalability, stability, and ease of deployment.

---

## Table of Contents

1. Project Overview  
2. Key Features  
3. System Architecture  
4. Datasets  
5. Installation & Setup  
6. Usage  
7. Admin Panel  
8. Internship Task Evaluation  
9. Technical Implementation Details  
10. Performance & Scalability  
11. Conclusion  
12. References  

---

## 1. Project Overview

Yukti AI fulfills the requirements of six internship tasks by implementing a single, cohesive chatbot that:

- Periodically updates its vector database with new information from CSV files, RSS feeds, APIs, and website crawls.
- Handles and generates text, images, videos, and audio using Zhipu AI and Google Gemini.
- Provides a dedicated medical question‑answering module using the MedQuAD dataset.
- Serves as a domain expert on scientific papers from the arXiv dataset.
- Detects user sentiment and tailors responses accordingly.
- Automatically detects and responds in over 100 languages, including Hinglish.

All features are integrated into a Streamlit application with a comprehensive admin dashboard for monitoring and management.

---

## 2. Key Features

### 2.1 Dynamic Knowledge Base Expansion
A background thread monitors the `dataset/` folder (excluding `dataset/medical/`), user uploads, and configuration files for changes. When a change is detected, the FAISS index is rebuilt automatically, incorporating new CSV, RSS, API, or scraped web data. The rebuild uses an atomic swap to avoid corruption.

### 2.2 Multi‑Modal Capabilities
- **Text generation** via Zhipu GLM‑4‑Flash, GLM‑4‑Plus, GLM‑5, and Google Gemini.
- **Image generation** with CogView‑3‑Flash and CogView‑4.
- **Video generation** (asynchronous) using CogVideoX‑3 and CogVideoX‑Flash.
- **Audio generation** with GLM‑4‑Voice.
- All modalities are selectable from the sidebar; file uploads are supported where applicable (e.g., reference image for video generation).

### 2.3 Medical Q&A (Yukti‑Doctor)
- Uses the MedQuAD dataset (47,457 QA pairs from 12 NIH websites). A separate FAISS index is built automatically.
- Includes a 50‑year clinical experience persona, red‑flag detection (e.g., chest pain triggers emergency disclaimer), and a simple dialogue state machine for follow‑up questions.
- File processing: OCR for images (using Tesseract) and text extraction from PDFs (using PyPDF2) to incorporate uploaded reports.
- User feedback collection (ratings and comments) stored in the `medical_feedback` table.

### 2.4 Domain Expert on arXiv
- A sample of 10,000 arXiv papers (title + abstract) from computer science, physics, and other fields is indexed in the main knowledge base.
- The chatbot can discuss advanced topics, summarise research, and explain concepts using retrieval‑augmented generation (RAG).
- The main FAISS index also contains the original customer service dataset, allowing combined domain knowledge.

### 2.5 Sentiment Analysis
- A keyword‑based emotion detector classifies user input as `sad`, `happy`, `angry`, `confused`, or `neutral`.
- The detected emotion is injected into the LLM prompt, guiding the tone of the response (e.g., empathetic for sadness, cheerful for happiness).

### 2.6 Multi‑Language Support
- Detection via FastText (176 languages), script analysis (Devanagari, Bengali, etc.), and a custom Hinglish wordlist.
- Explicit language instructions (e.g., “answer in Hindi”) override auto‑detection.
- The response language follows the detected language or the explicit instruction.
- Hinglish (Hindi written in Roman script) is correctly identified and answered in a mixed style.

---

## 3. System Architecture

| Component | Technology / Library |
|-----------|----------------------|
| Frontend | Streamlit |
| Backend | Python 3.11+ |
| Vector Store | FAISS with sentence‑transformers/all‑MiniLM‑L6‑v2 embeddings |
| Language Models | Zhipu AI (GLM‑4, CogView, CogVideo, GLM‑Voice), Google Gemini |
| Database | SQLite (users, activity, tasks, metrics, medical feedback) |
| Web Scraping | requests, BeautifulSoup, Playwright (optional) |
| Language Detection | FastText, custom script analysis, transformer fallback |
| OCR | Tesseract (optional) |

The code is organised into modules:

- `src/main.py` – Streamlit UI and admin dashboard.
- `src/model_manager.py` – Model orchestration, concurrency control, async task queue.
- `src/medical.py` – Medical module logic (retriever, dialogue, file processing).
- `src/think.py` – Core reasoning (retrieval, emotion, language, web search).
- `src/langchain_helper.py` – Vector store operations (load, save, similarity search, re‑ranking).
- `src/language_detector.py` – Multi‑language detection.
- `src/knowledge_updater/` – Auto‑updater, connectors, crawler, builders.
- `run_updater.py` – Standalone updater for cron.

All paths are hardcoded relative to the project root to ensure portability.

---

## 4. Datasets

- **MedQuAD** – 47,457 medical QA pairs from 12 NIH websites. Cloned from GitHub and converted to a single CSV with columns `question`, `answer`, `source`, `disease`, `topic`. Placed in `dataset/medical/`.
- **arXiv** – 10,000 sample papers from Kaggle. Each paper is transformed into a document with `Title: ...\nAbstract: ...` and saved as `dataset/arxiv_papers.csv`.
- **General Knowledge** – A custom `dataset/dataset.csv` (supplied with the project) contains general customer service QA pairs.

---

## 5. Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/CodeWander-666/yukti-ai.git
   cd yukti-ai/Yukti-Ai/customer_service_chatbot_LLM
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv311
   source venv311/bin/activate        # Linux/macOS
   venv311\Scripts\activate           # Windows
   pip install -r requirements.txt
   ```

3. Download and prepare datasets (optional – the auto‑updater will build indexes when the app runs):
   ```bash
   python scripts/setup_datasets.py
   mkdir dataset\medical
   move dataset\medquad.csv dataset\medical\
   ```

4. Build the medical index (the app will also do this automatically):
   ```bash
   python src/knowledge_updater/build_medical_index.py
   ```

5. Set API keys as environment variables (required for model generation):
   ```bash
   export ZHIPU_API_KEY=your_key_here
   export GOOGLE_API_KEY=your_key_here
   ```

6. Run the Streamlit app:
   ```bash
   streamlit run src/main.py
   ```

7. Log in with the default admin credentials:  
   Username: `admin1234`  
   Password: `admin1234`

---

## 6. Usage

- **Model Selection**: Choose a model from the sidebar (Yukti‑Flash, Yukti‑Quantum, Yukti‑Image, Yukti‑Video, Yukti‑Audio, Gemini, Yukti‑Doctor).
- **File Upload**: When a multi‑modal or medical model is selected, a file uploader appears.
- **Chat**: Type your query. The app will retrieve relevant documents (if applicable) and generate a response.
- **Web Scraping**: If your query contains a URL and keywords like “scrape” or “get content”, the system will scrape the page and return the text.
- **Admin Dashboard**: Access by toggling Admin Mode in the sidebar.

---

## 7. Admin Panel

The admin dashboard provides full control and monitoring. It is divided into eight tabs:

### 7.1 Overview
- System status cards (users, KB docs, active tasks, last KB update).
- Live system metrics (CPU, memory, disk).
- Database diagnostics (record counts in each table).

### 7.2 Users
- User list with activity summaries (total messages, success rate, average response time, last active, most used model, media generation counts).
- Add new user with password and admin flag.
- Edit or delete users; all actions are logged.

### 7.3 Analytics
- Key performance indicators (average latency, resolution rate, fallback rate).
- Model performance over time (response time and error rate charts for the last 30 days).
- User activity & engagement (daily active users, total queries).
- System health (CPU, memory, disk history over 24 hours).
- Model quality radar (placeholder for future metrics).

### 7.4 Knowledge Base
- Document count and index size metrics; historical growth chart.
- Auto‑update status confirmation.
- File upload for CSV, TXT, PDF.
- RSS & API configuration (add, edit, remove feeds and endpoints, saved in `sources.json`).
- Website crawling configuration (URL, max pages, JavaScript toggle, enable/disable, saved in `web_sources.json`).

### 7.5 Tasks
- List of active async tasks (video generation) with progress bars.
- Task history table (completed/failed tasks with result URLs and errors).

### 7.6 Insights
- Model health (highlights models with error rate >10% in the last 7 days).
- User growth (new users per day, average new users per day).
- Knowledge base age warning.
- System recommendations (high CPU, memory, disk alerts).
- Forecast (expected daily queries based on recent activity).

### 7.7 System
- Export analytics to Excel (includes model performance, user activity, system metrics, KB history, tasks).
- Log viewer (displays last 50 lines of `updater.log`).

### 7.8 Medical
- Feedback summary (bar chart of ratings from `medical_feedback`).
- Recent feedback table (query, rating, comment, timestamp).

---

## 8. Internship Task Evaluation

| Task | Implementation Status |
|------|------------------------|
| Dynamic Knowledge Base Expansion | Fully implemented via auto‑updater, supports CSV, RSS, API, web crawling. |
| Multi‑Modal Chatbot | Text, image, video, audio generation using Zhipu AI and Gemini. |
| Medical Q&A (MedQuAD) | Dedicated medical module with FAISS index, 50‑year persona, file processing, and feedback. |
| Domain Expert (arXiv) | 10,000 arXiv papers indexed; retrieval‑augmented answers on scientific topics. |
| Sentiment Analysis | Keyword‑based emotion detection integrated into prompt. |
| Multi‑Language Support | 100+ languages with automatic detection, explicit instruction support, Hinglish handling. |

---

## 9. Technical Implementation Details

### 9.1 Knowledge Base Updater
- `KnowledgeBaseUpdater` runs as a daemon thread, scanning all `.csv` files in `dataset/` (excluding `medical/`) and `data/uploads/` every 2 seconds.
- On change, `rebuild_index()` is called, which fetches all sources using `fetch_all_sources()` (CSVs, uploads, RSS, API, web crawls) and builds a new FAISS index in a temporary directory. The old index is then atomically replaced.
- Deduplication is performed using a hash of the page content.

### 9.2 Multi‑Modal Model Orchestration
- `model_manager.py` maintains a registry of models with concurrency limits. For synchronous models (text, image, audio), it uses a `ConcurrencyTracker` to enforce limits.
- For asynchronous video generation, tasks are submitted via the Zhipu API, stored in SQLite, and polled in a background thread. Progress is displayed in the sidebar.
- The Gemini client dynamically refreshes the model list and handles quota errors with retries.

### 9.3 Medical Module
- `MedicalRetriever` loads the medical FAISS index and performs similarity search.
- `MedicalDialogue` tracks the chief complaint and asks follow‑up questions (duration, severity) to gather more information before generating a final answer.
- File uploads are processed with OCR (Tesseract) for images and PyPDF2 for PDFs; the extracted text is appended to the query.
- The prompt explicitly sets the 50‑year experience persona and includes a disclaimer.

### 9.4 Language Detection
- **FastText**: Model `lid.176.bin` (176 languages) provides high‑confidence predictions.
- **Script analysis**: Unicode ranges identify Devanagari, Bengali, etc., mapping to language codes.
- **Hinglish wordlist**: A curated list of Hindi words in Roman script; if the text has 15‑85% Hindi words, it is classified as Hinglish.
- **Explicit instruction detection**: Regular expressions catch patterns like “answer in Hindi”.
- **Transformer fallback** (optional): `papluca/xlm-roberta-base-language-detection` for better accuracy.

### 9.5 Sentiment Analysis
- `detect_emotion()` uses keyword lists to classify into one of five categories.
- The emotion is inserted into the prompt as “User mood: ...”, allowing the LLM to adapt its tone.

### 9.6 Web Scraping
- `WebScraper` supports static pages (requests + BeautifulSoup) and dynamic pages (Playwright). It respects `robots.txt`, rotates user agents, and adds random delays.
- On‑demand scraping is triggered when a query contains a URL and keywords like “scrape”.

### 9.7 Admin Dashboard Metrics
- System metrics are collected via `psutil` every 60 seconds and stored in `system_metrics`.
- KB metrics are recorded after each rebuild (document count, index size).
- User activity logs every query with model, response time, success flag, and error.

---

## 10. Performance & Scalability

- The FAISS index for the main knowledge base (10,000+ documents) loads in under 2 seconds and query latency is <100ms.
- The medical index (47,457 QA pairs) is similarly efficient.
- Concurrency limits prevent API throttling: 200 for GLM‑4‑Flash, 20 for GLM‑4‑Plus, 5 for video models, etc.
- Asynchronous video tasks allow long‑running operations without blocking the UI.
- The auto‑updater rebuilds indexes only when changes are detected, and uses atomic swaps to avoid downtime.
- The system can be scaled by increasing concurrency limits, moving to a distributed vector store (e.g., PostgreSQL with pgvector), and using a task queue (Celery) for heavy workloads.

---

## 11. Conclusion

Yukti AI demonstrates a complete, production‑ready chatbot that exceeds the requirements of the internship. It combines cutting‑edge language models with a flexible knowledge base, advanced retrieval, and comprehensive administration tools. The system is open‑source, well‑documented, and ready for deployment.

---

## 12. References

- [MedQuAD Dataset](https://github.com/abachaa/MedQuAD)
- [arXiv Dataset](https://www.kaggle.com/datasets/Cornell-University/arxiv)
- [Zhipu AI API](https://open.bigmodel.cn/)
- [Google Gemini API](https://ai.google.dev/gemini-api)
- [LangChain](https://www.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Sentence‑Transformers](https://www.sbert.net/)
- [FastText Language Identification](https://fasttext.cc/docs/en/language-identification.html)

---

**Repository**: [https://github.com/CodeWander-666/yukti-ai](https://github.com/CodeWander-666/yukti-ai)
**Collab Notebook** : https://colab.research.google.com/drive/1ZshQQ4eMDWWQZmWBEuE0pkWNlnxXGNzD?usp=sharing

**Author**: NIKHIL SINGH
**Date**: March 2026
