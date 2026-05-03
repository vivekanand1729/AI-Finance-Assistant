"""
web_app/app.py
Multi-tab Streamlit interface for the AI Finance Assistant.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# Ensure project root is in path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ── Bridge Streamlit Secrets → os.environ (Streamlit Cloud + local secrets.toml)
# Must happen before ANY other module import that touches os.getenv().
_SECRET_KEYS = [
    "OPENAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY",
    "TAVILY_API_KEY", "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST", "LANGCHAIN_API_KEY", "GUARDRAILS_API_KEY",
]
for _k in _SECRET_KEYS:
    try:
        _val = st.secrets.get(_k, "")
        if _val and not os.environ.get(_k):
            os.environ[_k] = _val
    except Exception:
        pass  # st.secrets not available in some environments — silently skip

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AI Finance Assistant",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'DM Serif Display', serif; }

    .main { background-color: #0d1117; }

    .metric-card {
        background: linear-gradient(135deg, #1c2333, #161b27);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin: 0.4rem 0;
    }
    .metric-value { font-size: 1.6rem; font-weight: 600; color: #e6edf3; }
    .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.05em; }

    .agent-badge {
        display: inline-block;
        background: #1f6feb22;
        border: 1px solid #1f6feb55;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 0.75rem;
        color: #79c0ff;
        margin-bottom: 0.5rem;
    }

    .stChatMessage { border-radius: 12px; }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialization ──────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],          # [{role, content, agent}]
        "portfolio_holdings": [],    # [{ticker, shares, avg_cost}]
        "goal_params": {},
        "session_id": __import__("uuid").uuid4().hex[:8],
        "last_market_data": {},
        "last_portfolio_metrics": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💹 AI Finance Assistant")
    st.markdown("*Democratizing Financial Literacy*")
    st.divider()

    st.markdown("### 🔑 Configuration")

    # ── Provider selector ─────────────────────────────────────────────────────
    _provider_idx = ["openai", "google", "anthropic"].index(
        st.session_state.get("provider", "openai")
    )
    provider = st.selectbox(
        "LLM Provider", ["openai", "google", "anthropic"], index=_provider_idx
    )
    st.session_state["provider"] = provider

    model_map = {
        "openai": "gpt-4o-mini",
        "google": "gemini-2.0-flash",
        "anthropic": "claude-sonnet-4-5",
    }
    model = st.text_input("Model", value=model_map[provider])
    st.session_state["model"] = model

    # ── API key input ─────────────────────────────────────────────────────────
    # Priority: sidebar input → os.environ (set by .env or Streamlit Secrets)
    _env_key_map = {
        "openai": ("OPENAI_API_KEY", "OpenAI API Key"),
        "google": ("GOOGLE_API_KEY", "Google API Key"),
        "anthropic": ("ANTHROPIC_API_KEY", "Anthropic API Key"),
    }
    _env_var, _label = _env_key_map[provider]

    # Check if key is already configured via Streamlit Secrets or environment
    _configured = bool(os.getenv(_env_var, ""))

    if _configured:
        # Key came from Secrets — don't render any input to avoid accidental overrides
        st.success(f"✅ {_label} configured", icon="🔑")
    else:
        # No key found — let user enter one manually
        api_key = st.text_input(
            _label,
            value="",
            type="password",
            placeholder=f"Paste your {_label}",
            key=f"api_key_{provider}",
        )
        if api_key:
            os.environ[_env_var] = api_key
        else:
            st.warning(f"⚠️ Enter your {_label} to use the assistant.", icon="🔐")

    st.divider()
    st.markdown("### 🛡️ Guardrails")
    from src.core.guardrails import guardrails_status  # noqa: PLC0415
    _gr = guardrails_status()
    if _gr["sdk_enabled"]:
        st.success("Guardrails AI active", icon="✅")
    elif _gr["api_key_set"]:
        st.warning("Guardrails AI key set but SDK not loaded", icon="⚠️")
    else:
        st.info("Built-in pattern guardrails active", icon="🔒")

    st.divider()
    st.markdown("### 🤖 Active Agents")
    agents_info = [
        ("📚", "Finance Q&A", "Educational content"),
        ("📊", "Portfolio", "Holdings analysis"),
        ("📈", "Market", "Real-time data"),
        ("🎯", "Goal Planning", "Projections"),
        ("📰", "News", "Market news"),
        ("💰", "Tax Education", "Tax strategy"),
    ]
    for icon, name, desc in agents_info:
        st.markdown(f"{icon} **{name}** – *{desc}*")

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown(f"*Session: `{st.session_state.session_id}`*")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_portfolio, tab_markets, tab_goals, tab_news = st.tabs(
    ["💬 Chat", "📊 Portfolio", "📈 Markets", "🎯 Goals", "📰 News"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("## 💬 Chat with Your Finance Assistant")
    st.markdown(
        "Ask anything about investing, financial concepts, market data, or planning. "
        "The assistant routes your question to the best-suited specialized agent."
    )

    # Display conversation history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            role = msg["role"]
            with st.chat_message(role, avatar="👤" if role == "user" else "🤖"):
                if role == "assistant" and msg.get("agent"):
                    st.markdown(
                        f'<div class="agent-badge">🤖 {msg["agent"]}</div>',
                        unsafe_allow_html=True,
                    )
                st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about stocks, investing, tax, goals…"):
        # Guard: ensure API key is set before proceeding
        _active_env_var = {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }.get(st.session_state.get("provider", "openai"), "OPENAI_API_KEY")

        if not os.getenv(_active_env_var):
            st.warning(
                f"⚠️ Please enter your API key in the **sidebar** before chatting.",
                icon="🔐",
            )
            st.stop()

        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Run agent
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                try:
                    from src.workflow.graph import run_query
                    from langchain_core.messages import HumanMessage, AIMessage

                    # Build message history for context
                    history = []
                    for m in st.session_state.chat_history[:-1]:
                        if m["role"] == "user":
                            history.append(HumanMessage(content=m["content"]))
                        else:
                            history.append(AIMessage(content=m["content"]))

                    result = run_query(
                        user_query=prompt,
                        portfolio_data={"holdings": st.session_state.portfolio_holdings},
                        session_id=st.session_state.session_id,
                        message_history=history,
                    )

                    response = result.get("agent_response", "I couldn't generate a response.")
                    agent_name = result.get("active_agent", "Assistant")

                    # Update cached data
                    if result.get("last_portfolio_metrics"):
                        st.session_state.last_portfolio_metrics = result["last_portfolio_metrics"]
                    if result.get("market_data"):
                        st.session_state.last_market_data = result["market_data"]

                    st.markdown(f'<div class="agent-badge">🤖 {agent_name}</div>', unsafe_allow_html=True)
                    st.markdown(response)

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": response, "agent": agent_name}
                    )
                except Exception as exc:
                    err_msg = f"⚠️ Error: {exc}\n\nPlease check your API key configuration in the sidebar."
                    st.error(err_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err_msg, "agent": "System"}
                    )

    # Suggested prompts
    if not st.session_state.chat_history:
        st.markdown("### 💡 Try asking:")
        cols = st.columns(2)
        suggestions = [
            "What's the difference between a Roth IRA and a Traditional IRA?",
            "Explain dollar-cost averaging for a beginner",
            "What is a P/E ratio and why does it matter?",
            "How much should I have in an emergency fund?",
            "What's the difference between stocks and bonds?",
            "How does compound interest work? Give me an example.",
        ]
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 2]
            with col:
                if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: PORTFOLIO
# ══════════════════════════════════════════════════════════════════════════════
with tab_portfolio:
    st.markdown("## 📊 Portfolio Analysis")

    col_input, col_results = st.columns([1, 2])

    with col_input:
        st.markdown("### Add Holdings")
        st.markdown("*Enter your stock positions for analysis*")

        with st.form("add_holding"):
            ticker = st.text_input("Ticker Symbol", placeholder="e.g., AAPL").upper()
            shares = st.number_input("Shares", min_value=0.0001, step=0.1, format="%.4f")
            avg_cost = st.number_input("Avg Cost Per Share ($)", min_value=0.01, step=0.01, format="%.2f")
            submitted = st.form_submit_button("Add to Portfolio", use_container_width=True)

        if submitted and ticker and shares > 0:
            # Remove duplicate
            st.session_state.portfolio_holdings = [
                h for h in st.session_state.portfolio_holdings if h["ticker"] != ticker
            ]
            st.session_state.portfolio_holdings.append(
                {"ticker": ticker, "shares": shares, "avg_cost": avg_cost}
            )
            st.success(f"✅ Added {ticker}")

        # Show holdings table
        if st.session_state.portfolio_holdings:
            st.markdown("### Current Holdings")
            df = pd.DataFrame(st.session_state.portfolio_holdings)
            st.dataframe(df, use_container_width=True, hide_index=True)

            if st.button("🗑️ Clear All Holdings", use_container_width=True):
                st.session_state.portfolio_holdings = []
                st.rerun()

            if st.button("🔍 Analyze Portfolio", type="primary", use_container_width=True):
                with st.spinner("Fetching prices and calculating metrics…"):
                    try:
                        from src.utils.market_data import market_service
                        from src.utils.portfolio_calculator import calculate_portfolio_metrics

                        tickers_list = [h["ticker"] for h in st.session_state.portfolio_holdings]
                        prices = market_service.get_portfolio_prices(tickers_list)
                        sector_map = {}
                        for t in tickers_list:
                            info = market_service.get_company_info(t)
                            sector_map[t] = info.get("sector", "Unknown")

                        metrics = calculate_portfolio_metrics(
                            st.session_state.portfolio_holdings, prices, sector_map
                        )
                        st.session_state.last_portfolio_metrics = {
                            "total_value": metrics.total_value,
                            "total_cost": metrics.total_cost,
                            "total_pnl": metrics.total_pnl,
                            "total_pnl_pct": metrics.total_pnl_pct,
                            "allocation": metrics.allocation,
                            "sector_allocation": metrics.sector_allocation,
                            "diversification_score": metrics.diversification_score,
                            "risk_level": metrics.risk_level,
                            "holdings_detail": [
                                {
                                    "ticker": h.ticker,
                                    "current_value": h.current_value,
                                    "unrealized_pnl": h.unrealized_pnl,
                                    "unrealized_pnl_pct": h.unrealized_pnl_pct,
                                    "current_price": h.current_price,
                                }
                                for h in metrics.holdings
                            ],
                            "recommendations": metrics.recommendations,
                        }
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Analysis failed: {exc}")

    with col_results:
        metrics = st.session_state.get("last_portfolio_metrics", {})
        if not metrics:
            st.info("👈 Add holdings and click **Analyze Portfolio** to see your dashboard.")
        else:
            # KPI row
            pnl = metrics["total_pnl"]
            pnl_pct = metrics["total_pnl_pct"]
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Value", f"${metrics['total_value']:,.2f}")
            k2.metric("Cost Basis", f"${metrics['total_cost']:,.2f}")
            k3.metric("Unrealized P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%")
            k4.metric("Risk Level", metrics["risk_level"])

            # Diversification score gauge
            div_score = metrics["diversification_score"]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=div_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Diversification Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#238636"},
                    "steps": [
                        {"range": [0, 40], "color": "#da3633"},
                        {"range": [40, 70], "color": "#d29922"},
                        {"range": [70, 100], "color": "#238636"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=220, margin=dict(t=30, b=10, l=30, r=30),
                                    paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3")
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Allocation charts
            c1, c2 = st.columns(2)
            with c1:
                alloc = metrics["allocation"]
                fig_pie = px.pie(
                    values=list(alloc.values()),
                    names=list(alloc.keys()),
                    title="Holdings Allocation",
                    hole=0.4,
                )
                fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                                      height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with c2:
                sector = metrics["sector_allocation"]
                fig_sector = px.bar(
                    x=list(sector.values()),
                    y=list(sector.keys()),
                    orientation="h",
                    title="Sector Allocation (%)",
                )
                fig_sector.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                                         height=300, margin=dict(t=40, b=10))
                st.plotly_chart(fig_sector, use_container_width=True)

            # Holdings detail table
            st.markdown("### Holdings Detail")
            detail = metrics.get("holdings_detail", [])
            if detail:
                df_detail = pd.DataFrame(detail)
                df_detail["current_value"] = df_detail["current_value"].map("${:,.2f}".format)
                df_detail["unrealized_pnl"] = df_detail["unrealized_pnl"].map("${:+,.2f}".format)
                df_detail["unrealized_pnl_pct"] = df_detail["unrealized_pnl_pct"].map("{:+.2f}%".format)
                df_detail["current_price"] = df_detail["current_price"].map("${:,.2f}".format)
                st.dataframe(df_detail, use_container_width=True, hide_index=True)

            # Recommendations
            st.markdown("### 💡 Recommendations")
            for rec in metrics.get("recommendations", []):
                st.markdown(rec)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: MARKETS
# ══════════════════════════════════════════════════════════════════════════════
with tab_markets:
    st.markdown("## 📈 Market Overview")

    col_search, col_overview = st.columns([1, 2])

    with col_search:
        st.markdown("### Stock Lookup")
        ticker_input = st.text_input("Enter Ticker", placeholder="AAPL, MSFT, TSLA…").upper()
        if st.button("🔍 Get Quote", type="primary", use_container_width=True):
            if ticker_input:
                with st.spinner(f"Fetching {ticker_input}…"):
                    try:
                        from src.utils.market_data import market_service
                        from src.core.guardrails import sanitize_ticker

                        safe_ticker = sanitize_ticker(ticker_input)
                        quote = market_service.get_quote(safe_ticker)
                        company = market_service.get_company_info(safe_ticker)
                        history = market_service.get_history(safe_ticker, period="6mo")

                        if "error" not in quote:
                            prev = quote.get("previous_close", 0)
                            price = quote["price"]
                            change = price - prev
                            change_pct = (change / prev * 100) if prev else 0

                            st.markdown(f"### {safe_ticker}")
                            st.markdown(f"**{company.get('name', safe_ticker)}**")
                            st.markdown(f"*{company.get('sector', '')} | {company.get('industry', '')}*")

                            m1, m2 = st.columns(2)
                            m1.metric("Price", f"${price:,.2f}", f"{change_pct:+.2f}%")
                            m2.metric("52W High/Low",
                                      f"${quote.get('52w_high',0):,.2f} / ${quote.get('52w_low',0):,.2f}")

                            extra = {
                                "P/E Ratio": company.get("pe_ratio", "N/A"),
                                "EPS": company.get("eps", "N/A"),
                                "Beta": company.get("beta", "N/A"),
                                "Div Yield": f"{company.get('dividend_yield', 0) * 100:.2f}%" if company.get("dividend_yield") else "N/A",
                            }
                            st.json(extra)

                            # Price chart
                            if history:
                                df_hist = pd.DataFrame(history)
                                df_hist["date"] = pd.to_datetime(df_hist["date"])
                                fig_line = px.area(df_hist, x="date", y="close",
                                                   title=f"{safe_ticker} – 6 Month Price")
                                fig_line.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                                       font_color="#e6edf3", height=300,
                                                       showlegend=False)
                                st.plotly_chart(fig_line, use_container_width=True)
                        else:
                            st.error(f"Could not fetch data for {safe_ticker}: {quote.get('error')}")
                    except Exception as exc:
                        st.error(f"Error: {exc}")

    with col_overview:
        st.markdown("### Market Indices")
        if st.button("🔄 Refresh Market Data", use_container_width=True):
            with st.spinner("Fetching market data…"):
                try:
                    from src.utils.market_data import market_service
                    overview = market_service.get_market_overview()
                    st.session_state.last_market_data = {"market_overview": overview}
                    st.rerun()
                except Exception as exc:
                    st.error(f"Market data error: {exc}")

        overview_data = st.session_state.last_market_data.get("market_overview", {})
        if overview_data:
            cols = st.columns(len(overview_data))
            for i, (name, data) in enumerate(overview_data.items()):
                with cols[i]:
                    change = data.get("change_pct", 0)
                    cols[i].metric(name, f"{data['price']:,.2f}", f"{change:+.2f}%")

            # Bar chart of daily changes
            fig_bar = px.bar(
                x=list(overview_data.keys()),
                y=[d["change_pct"] for d in overview_data.values()],
                title="Daily Change (%)",
                color=[d["change_pct"] for d in overview_data.values()],
                color_continuous_scale=["#da3633", "#238636"],
            )
            fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e6edf3",
                                  height=350, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Click **Refresh Market Data** to load indices.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: GOALS
# ══════════════════════════════════════════════════════════════════════════════
with tab_goals:
    st.markdown("## 🎯 Financial Goal Planner")
    st.markdown("Project how your savings grow toward your financial goals.")

    col_params, col_chart = st.columns([1, 2])

    with col_params:
        st.markdown("### Goal Parameters")
        goal_name = st.text_input("Goal Name", placeholder="e.g., Retirement, House Down Payment")
        current_savings = st.number_input("Current Savings ($)", min_value=0.0, step=100.0, format="%.2f")
        monthly_contrib = st.number_input("Monthly Contribution ($)", min_value=0.0, step=50.0, format="%.2f", value=500.0)
        years = st.slider("Time Horizon (Years)", 1, 40, 10)
        risk_profile = st.selectbox("Risk Profile", ["Conservative (4%)", "Moderate (7%)", "Growth (9%)", "Aggressive (11%)"])
        target_amount = st.number_input("Target Amount ($) (optional)", min_value=0.0, step=1000.0, format="%.2f")

        return_map = {
            "Conservative (4%)": 4.0,
            "Moderate (7%)": 7.0,
            "Growth (9%)": 9.0,
            "Aggressive (11%)": 11.0,
        }
        annual_return = return_map[risk_profile]

        if st.button("📊 Project Growth", type="primary", use_container_width=True):
            from src.utils.portfolio_calculator import project_goal
            projection = project_goal(current_savings, monthly_contrib, annual_return, years)
            st.session_state["current_projection"] = {
                "data": projection,
                "goal_name": goal_name or "My Goal",
                "target": target_amount,
                "annual_return": annual_return,
            }
            st.rerun()

    with col_chart:
        proj = st.session_state.get("current_projection")
        if proj:
            data = proj["data"]
            df_proj = pd.DataFrame(data)
            goal_name_display = proj["goal_name"]
            target = proj["target"]

            # Final values
            final = data[-1]
            st.markdown(f"### {goal_name_display} Projection")
            g1, g2, g3 = st.columns(3)
            g1.metric("Final Balance", f"${final['balance']:,.0f}")
            g2.metric("Total Contributions", f"${final['contributions_total']:,.0f}")
            g3.metric("Investment Growth", f"${final['growth']:,.0f}")

            if target and target > 0:
                progress = min(final["balance"] / target, 1.0)
                st.progress(progress)
                if final["balance"] >= target:
                    st.success(f"✅ You'll reach ${target:,.0f} by year {next(d['year'] for d in data if d['balance'] >= target)}!")
                else:
                    shortfall = target - final["balance"]
                    st.warning(f"⚠️ You'll be ${shortfall:,.0f} short of your ${target:,.0f} goal.")

            # Area chart
            fig_area = go.Figure()
            fig_area.add_trace(go.Scatter(
                x=df_proj["year"], y=df_proj["contributions_total"],
                name="Total Contributions", fill="tozeroy",
                line=dict(color="#388bfd"), fillcolor="rgba(56,139,253,0.2)"
            ))
            fig_area.add_trace(go.Scatter(
                x=df_proj["year"], y=df_proj["balance"],
                name="Portfolio Value", fill="tonexty",
                line=dict(color="#3fb950"), fillcolor="rgba(63,185,80,0.2)"
            ))
            if target and target > 0:
                fig_area.add_hline(y=target, line_dash="dot", line_color="#f78166",
                                   annotation_text=f"Target: ${target:,.0f}")
            fig_area.update_layout(
                title=f"Wealth Projection @ {proj['annual_return']}% Annual Return",
                xaxis_title="Years",
                yaxis_title="Value ($)",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#e6edf3",
                height=400,
                legend=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_area, use_container_width=True)

            # Year-by-year table
            with st.expander("📋 Year-by-Year Breakdown"):
                df_display = df_proj.copy()
                df_display["balance"] = df_display["balance"].map("${:,.0f}".format)
                df_display["contributions_total"] = df_display["contributions_total"].map("${:,.0f}".format)
                df_display["growth"] = df_display["growth"].map("${:,.0f}".format)
                df_display.columns = ["Year", "Portfolio Value", "Total Contributed", "Investment Growth"]
                st.dataframe(df_display, use_container_width=True, hide_index=True)
        else:
            st.info("👈 Set your parameters and click **Project Growth**.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: NEWS
# ══════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.markdown("## 📰 Financial News Synthesizer")
    st.markdown("Ask about recent financial news and the agent will fetch and synthesize the latest updates.")

    news_query = st.text_input(
        "News Topic",
        placeholder="e.g., Fed interest rate decision, inflation data, tech stocks earnings…",
    )
    if st.button("📡 Fetch & Synthesize News", type="primary", use_container_width=True):
        if news_query:
            with st.spinner("Fetching latest news…"):
                try:
                    from src.workflow.graph import run_query
                    result = run_query(
                        user_query=f"financial news: {news_query}",
                        session_id=st.session_state.session_id,
                    )
                    st.markdown("### 📋 News Summary")
                    st.markdown(result.get("agent_response", "Unable to fetch news."))
                except Exception as exc:
                    st.error(f"News agent error: {exc}")
        else:
            st.warning("Please enter a news topic.")

    st.divider()
    st.markdown("### 💡 Popular Topics")
    topics = [
        "Federal Reserve interest rate decision",
        "S&P 500 performance this week",
        "Inflation CPI report",
        "Big tech earnings results",
        "Housing market update",
        "Cryptocurrency market news",
    ]
    cols = st.columns(3)
    for i, topic in enumerate(topics):
        with cols[i % 3]:
            if st.button(topic, key=f"news_{i}", use_container_width=True):
                st.session_state["prefilled_news"] = topic
                st.rerun()
