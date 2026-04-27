# 💹 AI Finance Assistant

> **Democratizing Financial Literacy Through Intelligent Conversational AI**

🚀 **Live Demo:** [https://ai-finance-assistant-1729.streamlit.app](https://ai-finance-assistant-1729.streamlit.app)

A production-ready multi-agent financial education assistant built with LangChain, LangGraph, and Streamlit. Six specialized AI agents work together to answer financial questions, analyze portfolios, track market data, plan goals, synthesize news, and explain tax concepts.

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│              LangGraph Workflow (StateGraph)          │
│                                                       │
│  ┌──────────┐    ┌──────────────────────────────┐   │
│  │ Classify │───▶│  Route to Specialized Agent  │   │
│  │  Intent  │    └──────────────────────────────┘   │
│  └──────────┘              │                         │
│                    ┌───────┼───────┐                 │
│               ┌────▼──┐ ┌─▼───┐ ┌─▼───┐ ...        │
│               │Finance│ │Port-│ │Mrkt │            │
│               │  Q&A  │ │folio│ │Data │            │
│               └───┬───┘ └──┬──┘ └──┬──┘            │
│                   └────────┴───────┘                 │
│                            │                         │
│              ┌─────────────▼──────────────┐          │
│              │  Guardrails + Disclaimer   │          │
│              └─────────────┬──────────────┘          │
└────────────────────────────┼─────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Streamlit UI   │
                    └─────────────────┘
```

### Six Specialized Agents

| Agent | Responsibility | Key Tools |
|-------|---------------|-----------|
| 📚 **Finance Q&A** | General financial education | RAG + Knowledge Base |
| 📊 **Portfolio Analysis** | Holdings review, P&L, risk | yfinance + Calculator |
| 📈 **Market Analysis** | Real-time quotes, trends | yfinance API |
| 🎯 **Goal Planning** | Savings projections | Compound interest model |
| 📰 **News Synthesizer** | Financial news | Tavily Search API |
| 💰 **Tax Education** | Tax concepts, accounts | RAG + Knowledge Base |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <your-repo>
cd ai_finance_assistant
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Minimum required:**
```env
OPENAI_API_KEY=sk-proj-...    # or GOOGLE_API_KEY / ANTHROPIC_API_KEY
```

**Optional (enables more features):**
```env
TAVILY_API_KEY=tvly-...        # for live news
LANGFUSE_PUBLIC_KEY=pk-lf-...  # for observability
LANGFUSE_SECRET_KEY=sk-lf-...
```

### 3. Run the App

```bash
streamlit run src/web_app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 Project Structure

```
ai_finance_assistant/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Abstract base class
│   │   ├── finance_qa_agent.py    # General Q&A + RAG
│   │   ├── portfolio_agent.py     # Portfolio analysis
│   │   ├── market_agent.py        # Real-time market data
│   │   ├── goal_planning_agent.py # Goal projections
│   │   ├── news_agent.py          # News synthesis
│   │   └── tax_agent.py           # Tax education
│   ├── core/
│   │   ├── config.py              # YAML + env config loader
│   │   ├── llm_factory.py         # LLM provider factory
│   │   ├── state.py               # LangGraph shared state
│   │   └── guardrails.py          # Safety checks + disclaimers
│   ├── rag/
│   │   ├── knowledge_base.py      # 40+ curated articles
│   │   └── retriever.py           # FAISS vector store + fallback
│   ├── utils/
│   │   ├── market_data.py         # yfinance wrapper + cache
│   │   └── portfolio_calculator.py # Metrics, P&L, projections
│   ├── workflow/
│   │   └── graph.py               # LangGraph StateGraph
│   └── web_app/
│       └── app.py                 # Streamlit multi-tab UI
├── tests/
│   └── test_all.py                # 50+ tests, 8 test classes
├── config.yaml                    # App configuration
├── .env.example                   # Environment template
├── requirements.txt
└── README.md
```

---

## 🖥️ Features

### 💬 Chat Tab
- Natural language conversations with context preservation
- Automatic intent classification routes to the right agent
- Suggested prompts for beginners
- Shows which agent responded

### 📊 Portfolio Tab
- Add holdings (ticker, shares, avg cost)
- Live price fetching via yfinance
- Metrics: total value, P&L, allocation %, sector breakdown
- Diversification score (HHI-based)
- Risk assessment + actionable recommendations
- Interactive Plotly charts (pie, bar, gauge)

### 📈 Markets Tab
- Individual stock lookup with 6-month price chart
- Company fundamentals (P/E, EPS, beta, dividend yield)
- Major indices overview (S&P 500, NASDAQ, Dow, Russell, VIX)
- Daily change visualization

### 🎯 Goals Tab
- Compound interest projections
- Risk-profile-based return assumptions
- Target amount progress bar
- Year-by-year breakdown table
- Area chart showing contributions vs growth

### 📰 News Tab
- Real-time news via Tavily API
- AI-synthesized summaries with market context
- Popular topic quick-launch buttons

---

## ⚙️ Configuration

Edit `config.yaml` to customize:

```yaml
llm:
  provider: "openai"          # openai | google | anthropic
  model: "gpt-4o-mini"
  temperature: 0.1

rag:
  top_k: 5                    # number of retrieved chunks
  chunk_size: 600

market_data:
  cache_ttl_seconds: 1800     # 30-min cache
```

---

## 🧪 Testing

```bash
# Run all tests (no API key needed for most)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Run only unit tests (no integration)
pytest tests/ -v -k "not Integration"
```

**Test coverage includes:**
- Portfolio calculator (metrics, P&L, projections)
- Guardrails (harmful content, disclaimer injection)
- Knowledge base (structure, completeness)
- RAG retriever (keyword fallback, relevance)
- Market data service (mocked yfinance)
- Intent classification (mocked LLM)
- Agent unit tests (mocked LLM)
- Integration tests (requires API keys)

---

## 🔒 Safety & Guardrails

All agent responses are automatically:
1. **Scanned** for prohibited patterns (guaranteed returns, get-rich-quick schemes)
2. **Appended** with educational disclaimer
3. **Validated** portfolio inputs before processing

---

## 📡 Observability

When `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, all LLM calls are traced in [Langfuse](https://langfuse.com) automatically via callback handler.

---

## 🛠️ Supported LLM Providers

| Provider | Model Examples |
|----------|---------------|
| OpenAI | `gpt-4o-mini`, `gpt-4o` |
| Google | `gemini-2.0-flash`, `gemini-1.5-pro` |
| Anthropic | `claude-sonnet-4-5`, `claude-haiku-4-5` |

Switch provider in sidebar or `config.yaml`.

---

## ⚠️ Disclaimer

This application is for **educational purposes only** and does not constitute financial, investment, or tax advice. Always consult a qualified financial professional before making investment decisions.

---

## 📚 Resources

- [LangChain Docs](https://python.langchain.com/)
- [LangGraph Guide](https://github.com/langchain-ai/langgraph)
- [Streamlit Docs](https://docs.streamlit.io/)
- [yfinance](https://github.com/ranaroussi/yfinance)
- [Investopedia](https://www.investopedia.com) – financial concepts reference
