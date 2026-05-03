"""
tests/test_all.py
Comprehensive test suite covering agents, RAG, portfolio calculator, workflow, and guardrails.
Target: 80%+ coverage.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ─────────────────────────────────────────────────────────────────────────────
# 1. Portfolio Calculator Tests (no external dependencies)
# ─────────────────────────────────────────────────────────────────────────────

from src.utils.portfolio_calculator import (
    calculate_portfolio_metrics,
    project_goal,
    PortfolioHolding,
)


class TestPortfolioHolding:
    def test_cost_basis(self):
        h = PortfolioHolding(ticker="AAPL", shares=10, avg_cost=150.0, current_price=180.0)
        assert h.cost_basis == 1500.0

    def test_current_value(self):
        h = PortfolioHolding(ticker="AAPL", shares=10, avg_cost=150.0, current_price=180.0)
        assert h.current_value == 1800.0

    def test_unrealized_pnl(self):
        h = PortfolioHolding(ticker="AAPL", shares=10, avg_cost=150.0, current_price=180.0)
        assert h.unrealized_pnl == 300.0

    def test_unrealized_pnl_pct(self):
        h = PortfolioHolding(ticker="AAPL", shares=10, avg_cost=150.0, current_price=180.0)
        assert abs(h.unrealized_pnl_pct - 20.0) < 0.01

    def test_zero_cost_basis(self):
        h = PortfolioHolding(ticker="X", shares=5, avg_cost=0.0, current_price=10.0)
        assert h.unrealized_pnl_pct == 0.0


class TestPortfolioMetrics:
    def _sample_holdings(self):
        return [
            {"ticker": "AAPL", "shares": 10, "avg_cost": 150.0},
            {"ticker": "MSFT", "shares": 5, "avg_cost": 300.0},
            {"ticker": "BND", "shares": 20, "avg_cost": 75.0},
        ]

    def _sample_prices(self):
        return {"AAPL": 180.0, "MSFT": 350.0, "BND": 72.0}

    def test_total_value(self):
        m = calculate_portfolio_metrics(self._sample_holdings(), self._sample_prices())
        expected = 10 * 180 + 5 * 350 + 20 * 72
        assert abs(m.total_value - expected) < 0.01

    def test_total_cost(self):
        m = calculate_portfolio_metrics(self._sample_holdings(), self._sample_prices())
        expected = 10 * 150 + 5 * 300 + 20 * 75
        assert abs(m.total_cost - expected) < 0.01

    def test_allocation_sums_to_100(self):
        m = calculate_portfolio_metrics(self._sample_holdings(), self._sample_prices())
        total_alloc = sum(m.allocation.values())
        assert abs(total_alloc - 100.0) < 0.1

    def test_diversification_score_range(self):
        m = calculate_portfolio_metrics(self._sample_holdings(), self._sample_prices())
        assert 0 <= m.diversification_score <= 100

    def test_single_holding_is_concentrated(self):
        """Single holding should have low diversification."""
        m = calculate_portfolio_metrics(
            [{"ticker": "AAPL", "shares": 10, "avg_cost": 150}],
            {"AAPL": 180}
        )
        assert m.diversification_score < 50

    def test_risk_level_values(self):
        m = calculate_portfolio_metrics(self._sample_holdings(), self._sample_prices())
        assert m.risk_level in {"Conservative", "Moderate", "Moderate-Aggressive", "Aggressive"}

    def test_recommendations_not_empty(self):
        m = calculate_portfolio_metrics(self._sample_holdings(), self._sample_prices())
        assert len(m.recommendations) > 0

    def test_empty_portfolio(self):
        m = calculate_portfolio_metrics([], {})
        assert m.total_value == 0.0
        assert m.allocation == {}

    def test_missing_price_uses_avg_cost(self):
        """If current price not provided, should fall back to avg_cost."""
        m = calculate_portfolio_metrics(
            [{"ticker": "UNKNOWN", "shares": 5, "avg_cost": 100}],
            {}  # no prices
        )
        assert m.total_value == 500.0  # 5 * 100


class TestProjectGoal:
    def test_basic_projection(self):
        result = project_goal(
            current_savings=10000,
            monthly_contribution=500,
            annual_return_pct=7.0,
            years=10,
        )
        assert len(result) == 10
        assert result[0]["year"] == 1
        assert result[-1]["year"] == 10

    def test_growth_is_positive(self):
        result = project_goal(1000, 100, 7.0, 5)
        assert result[-1]["balance"] > 1000

    def test_balance_increases_each_year(self):
        result = project_goal(0, 500, 7.0, 5)
        balances = [r["balance"] for r in result]
        assert all(b2 > b1 for b1, b2 in zip(balances, balances[1:]))

    def test_zero_return(self):
        """At 0% return, balance should equal contributions."""
        result = project_goal(0, 100, 0.0, 3)
        expected = 100 * 12 * 3
        assert abs(result[-1]["balance"] - expected) < 0.5

    def test_initial_savings_counted(self):
        r1 = project_goal(0, 500, 7.0, 10)
        r2 = project_goal(10000, 500, 7.0, 10)
        assert r2[-1]["balance"] > r1[-1]["balance"]


# ─────────────────────────────────────────────────────────────────────────────
# 2. Guardrails Tests
# ─────────────────────────────────────────────────────────────────────────────

from src.core.guardrails import check_response, validate_portfolio_input, sanitize_ticker


class TestGuardrails:
    def test_disclaimer_added(self):
        result = check_response("Stocks are great investments.")
        assert "educational purposes only" in result

    def test_harmful_pattern_blocked(self):
        result = check_response("This is a guaranteed return investment!")
        assert "unable to provide" in result.lower()

    def test_get_rich_quick_blocked(self):
        result = check_response("Use this get-rich-quick scheme!")
        assert "unable to provide" in result.lower()

    def test_normal_content_passes(self):
        text = "Dollar-cost averaging is a prudent strategy."
        result = check_response(text)
        assert text in result

    def test_validate_portfolio_valid(self):
        portfolio = [{"ticker": "AAPL", "shares": 10, "avg_cost": 150.0}]
        valid, msg = validate_portfolio_input(portfolio)
        assert valid is True
        assert msg == ""

    def test_validate_portfolio_missing_key(self):
        portfolio = [{"ticker": "AAPL", "shares": 10}]  # missing avg_cost
        valid, msg = validate_portfolio_input(portfolio)
        assert valid is False

    def test_validate_portfolio_negative_shares(self):
        portfolio = [{"ticker": "AAPL", "shares": -5, "avg_cost": 150.0}]
        valid, msg = validate_portfolio_input(portfolio)
        assert valid is False

    def test_sanitize_ticker_clean(self):
        assert sanitize_ticker("aapl") == "AAPL"

    def test_sanitize_ticker_strips_special(self):
        assert sanitize_ticker("$AAPL!") == "AAPL"

    def test_sanitize_ticker_max_length(self):
        result = sanitize_ticker("AVERYLONGTICKERSYMBOL")
        assert len(result) <= 10


# ─────────────────────────────────────────────────────────────────────────────
# 3. Knowledge Base Tests
# ─────────────────────────────────────────────────────────────────────────────

from src.rag.knowledge_base import (
    get_knowledge_base,
    get_by_category,
    get_all_categories,
)


class TestKnowledgeBase:
    def test_knowledge_base_not_empty(self):
        kb = get_knowledge_base()
        assert len(kb) >= 30

    def test_all_articles_have_required_fields(self):
        for article in get_knowledge_base():
            assert "id" in article
            assert "title" in article
            assert "content" in article
            assert "category" in article
            assert "source" in article

    def test_get_by_category_returns_correct(self):
        stocks = get_by_category("Stocks")
        assert all(a["category"] == "Stocks" for a in stocks)

    def test_get_by_category_case_insensitive(self):
        a = get_by_category("stocks")
        b = get_by_category("STOCKS")
        assert len(a) == len(b)

    def test_get_all_categories_no_duplicates(self):
        cats = get_all_categories()
        assert len(cats) == len(set(cats))

    def test_categories_include_required_topics(self):
        cats = set(get_all_categories())
        required = {"Stocks", "Bonds", "ETFs", "Tax", "Retirement"}
        assert required.issubset(cats), f"Missing categories: {required - cats}"

    def test_content_length_sufficient(self):
        for article in get_knowledge_base():
            assert len(article["content"]) >= 50, f"Article {article['id']} too short"


# ─────────────────────────────────────────────────────────────────────────────
# 4. RAG Retriever Tests (keyword fallback – no API needed)
# ─────────────────────────────────────────────────────────────────────────────

from src.rag.retriever import FinanceRAGRetriever


class TestRAGRetriever:
    def setup_method(self):
        self.retriever = FinanceRAGRetriever(top_k=3)
        # Don't build FAISS index; test keyword fallback
        self.retriever._vectorstore = None

    def test_retrieve_returns_list(self):
        results = self.retriever.retrieve("what is a stock")
        assert isinstance(results, list)

    def test_retrieve_respects_top_k(self):
        results = self.retriever.retrieve("investing", top_k=3)
        assert len(results) <= 3

    def test_retrieve_has_required_fields(self):
        results = self.retriever.retrieve("bonds interest rate")
        if results:
            assert "content" in results[0]
            assert "score" in results[0]

    def test_retrieve_relevant_to_stocks(self):
        results = self.retriever.retrieve("stock P/E ratio earnings")
        contents = " ".join(r["content"] for r in results).lower()
        assert any(w in contents for w in ["stock", "p/e", "earnings", "ratio"])

    def test_retrieve_tax_query(self):
        results = self.retriever.retrieve("Roth IRA tax advantages retirement")
        assert len(results) > 0

    def test_retrieve_empty_query(self):
        results = self.retriever.retrieve("")
        assert isinstance(results, list)

    def test_keyword_retrieve_scores_between_0_1(self):
        results = self.retriever._keyword_retrieve("compound interest", 5)
        for r in results:
            assert 0 <= r["score"] <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Market Data Service Tests (mocked)
# ─────────────────────────────────────────────────────────────────────────────

from unittest.mock import patch, MagicMock
from src.utils.market_data import MarketDataService


def _mock_finnhub(endpoint_responses: dict):
    """Return a requests.get mock that dispatches by URL substring."""
    def _side_effect(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        for key, payload in endpoint_responses.items():
            if key in url:
                resp.json.return_value = payload
                return resp
        resp.json.return_value = {}
        return resp
    return _side_effect


class TestMarketDataService:
    def setup_method(self):
        self.svc = MarketDataService()
        # Clear the module-level cache between tests
        from src.utils import market_data as md
        md._cache.clear()

    def test_sanitize_ticker_in_get_quote(self):
        """Lowercase ticker is uppercased before the API call."""
        quote_payload   = {"c": 150.0, "pc": 148.0, "dp": 1.35, "v": 1000000}
        metric_payload  = {"metric": {"52WeekHigh": 180.0, "52WeekLow": 120.0}}
        responses = {"quote": quote_payload, "metric": metric_payload}
        with patch("requests.get", side_effect=_mock_finnhub(responses)):
            with patch("src.core.config.finnhub_key", return_value="test-key"):
                result = self.svc.get_quote("aapl")
        assert result["ticker"] == "AAPL"
        assert result["price"] == 150.0

    def test_get_quote_error_returns_error_dict(self):
        with patch("requests.get", side_effect=Exception("Network error")):
            with patch("src.core.config.finnhub_key", return_value="test-key"):
                result = self.svc.get_quote("BADTICKER")
        assert "error" in result

    def test_get_history_error_returns_empty(self):
        with patch("requests.get", side_effect=Exception("API down")):
            with patch("src.core.config.finnhub_key", return_value="test-key"):
                result = self.svc.get_history("AAPL", "1mo")
        assert result == []

    def test_get_portfolio_prices_returns_dict(self):
        with patch.object(self.svc, "get_quote") as mock_quote:
            mock_quote.side_effect = lambda t: {"ticker": t, "price": 100.0}
            prices = self.svc.get_portfolio_prices(["AAPL", "MSFT"])
            assert prices["AAPL"] == 100.0
            assert prices["MSFT"] == 100.0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Workflow / Intent Classification Tests (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

from unittest.mock import patch, MagicMock
from src.workflow.graph import classify_intent
from src.core.state import AgentState


def _make_state(query: str) -> AgentState:
    return {
        "messages": [],
        "user_query": query,
        "intent": "",
        "active_agent": "",
        "agent_response": "",
        "rag_context": [],
        "market_data": {},
        "portfolio_data": {},
        "goal_data": {},
        "error": None,
        "iterations": 0,
        "session_id": "test",
    }


class TestIntentClassification:
    def _mock_llm_response(self, intent: str):
        mock_response = MagicMock()
        mock_response.content = intent
        return mock_response

    def test_classify_finance_qa(self):
        with patch("src.workflow.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = self._mock_llm_response("finance_qa")
            mock_get_llm.return_value = mock_llm

            state = _make_state("What is compound interest?")
            result = classify_intent(state)
            assert result["intent"] == "finance_qa"

    def test_classify_portfolio(self):
        with patch("src.workflow.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = self._mock_llm_response("portfolio")
            mock_get_llm.return_value = mock_llm

            state = _make_state("Analyze my portfolio holdings")
            result = classify_intent(state)
            assert result["intent"] == "portfolio"

    def test_classify_fallback_on_empty_query(self):
        state = _make_state("")
        result = classify_intent(state)
        assert result["intent"] == "fallback"

    def test_classify_invalid_response_defaults_to_qa(self):
        with patch("src.workflow.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = self._mock_llm_response("gibberish_category")
            mock_get_llm.return_value = mock_llm

            state = _make_state("Some query")
            result = classify_intent(state)
            assert result["intent"] == "finance_qa"

    def test_classify_on_llm_error_defaults_to_qa(self):
        with patch("src.workflow.graph.get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.invoke.side_effect = Exception("LLM down")
            mock_get_llm.return_value = mock_llm

            state = _make_state("What are ETFs?")
            result = classify_intent(state)
            assert result["intent"] == "finance_qa"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Agent Unit Tests (mocked LLM)
# ─────────────────────────────────────────────────────────────────────────────

class TestBaseAgent:
    def _make_mock_agent(self):
        from src.agents.finance_qa_agent import FinanceQAAgent
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Test response about investing.")
        return FinanceQAAgent(llm=mock_llm)

    def test_run_returns_state(self):
        agent = self._make_mock_agent()
        state = _make_state("What is a stock?")
        result = agent.run(state)
        assert "agent_response" in result
        assert len(result["agent_response"]) > 0

    def test_run_appends_disclaimer(self):
        agent = self._make_mock_agent()
        state = _make_state("What is a stock?")
        result = agent.run(state)
        assert "educational purposes only" in result["agent_response"]

    def test_run_sets_active_agent(self):
        agent = self._make_mock_agent()
        state = _make_state("What is a stock?")
        result = agent.run(state)
        assert result["active_agent"] == agent.name

    def test_run_handles_llm_error(self):
        from src.agents.finance_qa_agent import FinanceQAAgent
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM unavailable")
        agent = FinanceQAAgent(llm=mock_llm)

        state = _make_state("What is a stock?")
        result = agent.run(state)
        assert result["error"] is not None
        assert "encountered an issue" in result["agent_response"]


class TestGoalPlanningAgent:
    def test_extract_goal_params_years(self):
        from src.agents.goal_planning_agent import GoalPlanningAgent
        agent = GoalPlanningAgent(llm=MagicMock())
        params = agent._extract_goal_params("I want to save for 20 years")
        assert params["years"] == 20

    def test_extract_goal_params_monthly_contribution(self):
        from src.agents.goal_planning_agent import GoalPlanningAgent
        agent = GoalPlanningAgent(llm=MagicMock())
        params = agent._extract_goal_params("I can save $800 per month")
        assert params["monthly_contribution"] == 800.0

    def test_extract_goal_params_conservative(self):
        from src.agents.goal_planning_agent import GoalPlanningAgent
        agent = GoalPlanningAgent(llm=MagicMock())
        params = agent._extract_goal_params("I'm conservative and don't like risk")
        assert params["return_pct"] == 4.0

    def test_extract_goal_params_aggressive(self):
        from src.agents.goal_planning_agent import GoalPlanningAgent
        agent = GoalPlanningAgent(llm=MagicMock())
        params = agent._extract_goal_params("I want aggressive growth")
        assert params["return_pct"] == 10.0


class TestMarketAgent:
    def test_extract_tickers_finds_symbols(self):
        from src.agents.market_agent import MarketAnalysisAgent
        agent = MarketAnalysisAgent(llm=MagicMock())
        tickers = agent._extract_tickers("What's the price of AAPL and MSFT?")
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_extract_tickers_filters_common_words(self):
        from src.agents.market_agent import MarketAnalysisAgent
        agent = MarketAnalysisAgent(llm=MagicMock())
        tickers = agent._extract_tickers("IS the market UP today?")
        assert "IS" not in tickers
        assert "UP" not in tickers

    def test_summarize_history_empty(self):
        from src.agents.market_agent import MarketAnalysisAgent
        agent = MarketAnalysisAgent(llm=MagicMock())
        result = agent._summarize_history([])
        assert result == {}

    def test_summarize_history_calculates_return(self):
        from src.agents.market_agent import MarketAnalysisAgent
        agent = MarketAnalysisAgent(llm=MagicMock())
        history = [
            {"date": "2024-01-01", "close": 100.0},
            {"date": "2024-03-01", "close": 110.0},
        ]
        result = agent._summarize_history(history)
        assert abs(result["period_return_pct"] - 10.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# 8. Integration Tests (require API keys – skipped in CI without keys)
# ─────────────────────────────────────────────────────────────────────────────

import os

@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
class TestIntegration:
    def test_full_qa_workflow(self):
        from src.workflow.graph import run_query
        result = run_query("What is compound interest?")
        assert result["agent_response"]
        assert result["intent"] in {"finance_qa", "fallback"}
        assert "educational purposes only" in result["agent_response"]

    def test_full_market_workflow(self):
        from src.workflow.graph import run_query
        result = run_query("What is the current price of AAPL?")
        assert result["agent_response"]
        assert result["intent"] in {"market", "finance_qa"}

    def test_full_goal_workflow(self):
        from src.workflow.graph import run_query
        result = run_query("I have $5000 saved and can invest $500/month for 10 years. Moderate risk.")
        assert result["agent_response"]
        assert result["intent"] in {"goal", "finance_qa"}
