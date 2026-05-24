# 📈 FinSight AI

FinSight AI is an end-to-end AI-powered financial sentiment analysis platform that analyzes real-time financial news using FinBERT, LangGraph, Groq LLMs, FastAPI, and Streamlit.

The system fetches live company-related financial news, performs sentiment analysis using a fine-tuned FinBERT model, generates executive summaries using Groq LLMs, and displays the results in a clean interactive dashboard.

---

# 🚀 Features

- Real-time financial news retrieval using Tavily Search API
- Fine-tuned FinBERT sentiment analysis
- LangGraph multi-node AI workflow
- Executive summary generation using Groq LLM
- FastAPI backend with async support
- Streamlit frontend dashboard
- Sentiment score visualization
- PDF report generation
- Company history tracking
- Swagger API documentation

---

# 🧠 Tech Stack

## AI / ML
- PyTorch
- HuggingFace Transformers
- FinBERT
- LangGraph
- Groq LLM
- Tavily Search API

## Backend
- FastAPI
- Pydantic
- Uvicorn

## Frontend
- Streamlit

## Other Tools
- FPDF
- Python Dotenv

---

# 🏗️ Architecture

```text
User
  ↓
Streamlit Frontend
  ↓
FastAPI Backend
  ↓
LangGraph Workflow
  ├── Search Node (Tavily)
  ├── Sentiment Node (FinBERT)
  ├── Summarise Node (Groq)
  └── Report Node
  ↓
Financial Report Output
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/finsight-ai.git

cd finsight-ai
```

---

## 2. Create Virtual Environment

```bash
python -m venv fin_env
```

### Windows

```bash
fin_env\Scripts\activate
```

### Linux / Mac

```bash
source fin_env/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
TAVILY_API_KEY=your_tavily_api_key

GROQ_API_KEY=your_groq_api_key
```

---

# 🤖 FinBERT Fine-Tuning

The FinBERT model was fine-tuned using:
- Financial PhraseBank Dataset
- PyTorch
- HuggingFace Transformers

Target:
- >88% Validation Accuracy

Notebook:

```text
notebooks/01_finbert_finetune.ipynb
```

---

# ▶️ Running the Backend

Start FastAPI server:

```bash
uvicorn src.api.main:app --reload
```

API Docs:

```text
http://127.0.0.1:8000/docs
```

---

# ▶️ Running the Frontend

Start Streamlit app:

```bash
streamlit run app.py
```

---

# 📊 API Endpoint

## POST `/analyze`

### Request

```json
{
  "company": "Tesla"
}
```

### Response

```json
{
  "company": "Tesla",
  "overall_sentiment": 0.19,
  "summary": {
    "positive": 3,
    "negative": 1,
    "neutral": 6
  },
  "executive_summary": "Tesla shows mixed market sentiment...",
  "report": "# Financial Sentiment Report..."
}
```

---

# 📈 Workflow

1. User enters company name
2. Tavily fetches recent financial news
3. FinBERT performs sentiment analysis
4. LangGraph processes workflow nodes
5. Groq LLM generates executive summary
6. Streamlit dashboard visualizes results
7. PDF report generated for download

---

# 📸 Dashboard Features

- Sentiment score metrics
- Positive / Negative / Neutral news count
- Executive summary
- Expandable detailed report
- PDF report download
- Recent company history sidebar

---

# 👨‍💻 Author

Dhanraj S H

---

# 📜 License

This project is licensed under the MIT License.