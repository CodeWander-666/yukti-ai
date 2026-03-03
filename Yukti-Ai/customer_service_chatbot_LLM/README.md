# Yukti AI – Intelligent Customer Service Chatbot

Yukti AI is a production‑grade, multi‑model conversational AI platform built with Streamlit. It provides a rich chat interface, advanced language detection, retrieval‑augmented generation (RAG), real‑time video/audio/image generation, and an extensive admin dashboard for system monitoring, user management, and analytics. The system is designed for scalability, stability, and ease of deployment.

---

## Features

- **Yukti Model Family**  
  Includes Yukti‑Flash (fast text & reasoning), Yukti‑Quantum (deep research), Yukti‑Image (image generation), Yukti‑Video (video generation), and Yukti‑Audio (speech synthesis). All models are accessible through a unified interface.

- **Language Detection**  
  Automatically detects user language (100+ languages, including Hinglish) using FastText, script analysis, a Hinglish wordlist, and optional transformer fallback. Explicit language instructions (e.g., “answer in Hindi”) are respected.

- **Retrieval‑Augmented Generation (RAG)**  
  A FAISS vector store indexes a knowledge base (CSV by default). Retrieved documents are injected into prompts to ground responses. Optional cross‑encoder re‑ranking improves answer quality.

- **User Authentication**  
  Secure login and sign‑up with bcrypt‑hashed passwords. An admin user can be created automatically on first run via environment variables.

- **Admin Dashboard**  
  A full‑featured dashboard accessible to admin users, with real‑time system metrics (CPU, memory, disk), user CRUD operations, model performance analytics, task monitoring, and system controls.

- **Asynchronous Video Generation**  
  Video tasks are submitted and polled in the background. Progress is shown in the sidebar, and completed videos can be downloaded directly.

- **Automatic Knowledge Base Updates**  
  A standalone updater (`run_updater.py`) can be scheduled via cron to rebuild the FAISS index from configured sources (CSV, RSS, API). Deduplication and atomic index replacement prevent corruption.

- **Professional, Customisable UI**  
  Cyberpunk‑themed interface with 3D buttons and neon accents. The chat area supports text, images, audio, and video with timestamped messages.

---

## Technology Stack

- **Frontend**: Streamlit, custom CSS, Plotly (charts)
- **Backend**: Python 3.11+, with integrations for advanced AI models (Zhipu, Google Gemini) that power the Yukti model family.
- **Database**: SQLite (user data, activity logs, task queue, system metrics)
- **Vector Store**: FAISS (with `sentence-transformers/all-MiniLM-L6-v2` embeddings)
- **Language Detection**: FastText, Hugging Face Transformers (optional)
- **Monitoring**: `psutil` for system metrics

---

## System Architecture

The project is organised into several modules, each with a clear responsibility:
