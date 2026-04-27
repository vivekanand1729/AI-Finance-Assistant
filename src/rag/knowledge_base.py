"""
rag/knowledge_base.py
Curated financial education knowledge base (60+ articles) + FAISS vector store.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Curated knowledge articles ────────────────────────────────────────────────

FINANCIAL_KNOWLEDGE: list[dict] = [
    # ── FUNDAMENTALS ──────────────────────────────────────────────────────────
    {
        "id": "inv_001",
        "category": "Fundamentals",
        "title": "What is Investing?",
        "content": (
            "Investing is the act of allocating money or resources into assets with the expectation of "
            "generating income or profit over time. Common investment vehicles include stocks, bonds, "
            "real estate, and mutual funds. Unlike saving (keeping money in a bank account), investing "
            "accepts some level of risk in exchange for potentially higher returns. The power of "
            "compounding means that even small returns build significant wealth over long periods."
        ),
        "source": "Investopedia",
    },
    {
        "id": "inv_002",
        "category": "Fundamentals",
        "title": "Compound Interest – The Eighth Wonder of the World",
        "content": (
            "Compound interest is the process of earning interest on both the principal amount and "
            "previously accumulated interest. The formula is A = P(1 + r/n)^(nt). At 7% annual "
            "return, money doubles roughly every 10 years (Rule of 72). Starting at 25 vs 35 can "
            "result in dramatically different retirement wealth. Even $200/month invested at 8% "
            "annually from age 25 grows to ~$702,000 by age 65."
        ),
        "source": "Bogleheads",
    },
    {
        "id": "inv_003",
        "category": "Fundamentals",
        "title": "Risk vs Return",
        "content": (
            "The risk-return tradeoff states that higher potential returns come with higher risk. "
            "Treasury bonds offer low risk (~3-5%) while equities offer higher returns (~8-10% historically) "
            "but with volatility. Risk tolerance depends on time horizon, income stability, and psychological "
            "comfort with market swings. Younger investors can afford more risk due to longer time horizons."
        ),
        "source": "SEC Investor Education",
    },
    {
        "id": "inv_004",
        "category": "Fundamentals",
        "title": "Inflation and Purchasing Power",
        "content": (
            "Inflation erodes purchasing power over time. At 3% annual inflation, $100 today buys only "
            "$74 worth of goods in 10 years. This is why keeping all money in a savings account earning "
            "1-2% is a losing strategy in real terms. Investing in assets that outpace inflation—stocks, "
            "real estate, TIPS—is essential for preserving and growing wealth."
        ),
        "source": "Federal Reserve Education",
    },
    # ── STOCKS ────────────────────────────────────────────────────────────────
    {
        "id": "stk_001",
        "category": "Stocks",
        "title": "What is a Stock?",
        "content": (
            "A stock represents partial ownership (equity) in a company. Shareholders benefit from "
            "price appreciation and dividends. Common stock gives voting rights; preferred stock "
            "offers priority dividend payments. Stock prices are determined by supply and demand, "
            "driven by corporate earnings, economic conditions, and investor sentiment."
        ),
        "source": "Investopedia",
    },
    {
        "id": "stk_002",
        "category": "Stocks",
        "title": "How to Evaluate Stocks: Key Metrics",
        "content": (
            "Key stock metrics: P/E Ratio (price/earnings – measures valuation), EPS (earnings per share), "
            "Revenue Growth (year-over-year), Profit Margins, Debt-to-Equity Ratio, Return on Equity (ROE), "
            "Dividend Yield (annual dividend / stock price), and Free Cash Flow. A stock with P/E of 15 "
            "and growing earnings may be undervalued vs. a P/E of 50 with slowing growth."
        ),
        "source": "Morningstar",
    },
    {
        "id": "stk_003",
        "category": "Stocks",
        "title": "Growth vs Value Investing",
        "content": (
            "Growth investing targets companies expected to grow faster than average (tech, biotech). "
            "These trade at high P/E multiples. Value investing seeks undervalued companies trading below "
            "intrinsic value (Warren Buffett's approach). Value stocks often pay dividends. Research shows "
            "that over long periods, value stocks have historically outperformed, but growth has led in "
            "recent decades."
        ),
        "source": "Bogleheads",
    },
    {
        "id": "stk_004",
        "category": "Stocks",
        "title": "Dividend Investing",
        "content": (
            "Dividend stocks pay regular cash distributions (usually quarterly). Yield = annual dividend / "
            "stock price. Dividend aristocrats (S&P 500 companies raising dividends 25+ consecutive years) "
            "include Coca-Cola, Johnson & Johnson, Procter & Gamble. DRIP (Dividend Reinvestment Plans) "
            "automatically reinvest dividends, compounding growth. High yield (>6%) may signal financial distress."
        ),
        "source": "Investopedia",
    },
    # ── BONDS ─────────────────────────────────────────────────────────────────
    {
        "id": "bnd_001",
        "category": "Bonds",
        "title": "What are Bonds?",
        "content": (
            "Bonds are debt instruments where the issuer (government or corporation) borrows money from "
            "investors and promises to repay with interest. Key terms: Face Value (par, typically $1,000), "
            "Coupon Rate (annual interest %), Maturity Date, and Yield. Bonds are generally less volatile "
            "than stocks and provide portfolio stability. US Treasury bonds are considered risk-free."
        ),
        "source": "SEC",
    },
    {
        "id": "bnd_002",
        "category": "Bonds",
        "title": "Bond Duration and Interest Rate Risk",
        "content": (
            "Bond prices move inversely to interest rates: when rates rise, bond prices fall. Duration "
            "measures price sensitivity to rate changes. A bond with duration of 7 loses ~7% value "
            "for each 1% rate increase. Short-duration bonds (1-3 years) are less sensitive. In rising "
            "rate environments, shorter-term or floating-rate bonds are preferable."
        ),
        "source": "PIMCO Education",
    },
    # ── ETFs & MUTUAL FUNDS ───────────────────────────────────────────────────
    {
        "id": "etf_001",
        "category": "ETFs",
        "title": "What is an ETF?",
        "content": (
            "Exchange-Traded Funds (ETFs) trade on exchanges like stocks but hold a basket of assets. "
            "Index ETFs track benchmarks like S&P 500 (SPY, VOO, IVV). Advantages: low expense ratios "
            "(0.03-0.20%), instant diversification, tax efficiency, intraday liquidity. Sector ETFs "
            "target specific industries (XLK for tech). Bond ETFs (AGG, BND) provide fixed income exposure."
        ),
        "source": "Vanguard",
    },
    {
        "id": "etf_002",
        "category": "ETFs",
        "title": "Index Funds vs Active Management",
        "content": (
            "Index funds passively track a market index with minimal trading. Active funds have managers "
            "making buy/sell decisions, resulting in higher fees (1-2% expense ratio). Studies show 85%+ "
            "of actively managed funds underperform their benchmark over 15 years after fees. Jack Bogle "
            "founded Vanguard on the principle that low-cost index funds win long-term."
        ),
        "source": "S&P SPIVA Report",
    },
    # ── DIVERSIFICATION & PORTFOLIO ───────────────────────────────────────────
    {
        "id": "div_001",
        "category": "Diversification",
        "title": "Modern Portfolio Theory",
        "content": (
            "Harry Markowitz's Modern Portfolio Theory (MPT) shows that combining assets with low "
            "correlations reduces portfolio risk without proportionally reducing returns. The Efficient "
            "Frontier represents portfolios with maximum return for a given risk level. Key insight: "
            "diversification eliminates unsystematic (company-specific) risk but not systematic (market) risk."
        ),
        "source": "Nobel Economics",
    },
    {
        "id": "div_002",
        "category": "Diversification",
        "title": "Asset Allocation Strategies",
        "content": (
            "Asset allocation divides investments among stocks, bonds, and cash. Common rules: "
            "'110 minus age' in stocks (110-30=80% stocks at age 30). Three-fund portfolio: US stocks, "
            "international stocks, bonds. 60/40 portfolio (60% stocks, 40% bonds) has historically "
            "produced solid risk-adjusted returns. Rebalancing annually maintains target allocation."
        ),
        "source": "Bogleheads",
    },
    {
        "id": "div_003",
        "category": "Diversification",
        "title": "Rebalancing Your Portfolio",
        "content": (
            "Rebalancing restores target allocation after market movements shift weights. If stocks rally "
            "and grow from 60% to 70%, sell some stocks and buy bonds to return to 60/40. Rebalance "
            "annually or when any asset class drifts 5%+ from target. Tax-advantaged accounts are best "
            "for rebalancing to avoid capital gains taxes."
        ),
        "source": "Vanguard Research",
    },
    # ── RETIREMENT ACCOUNTS ───────────────────────────────────────────────────
    {
        "id": "ret_001",
        "category": "Retirement",
        "title": "401(k) Plans",
        "content": (
            "401(k) is employer-sponsored retirement account with pre-tax contributions (traditional) "
            "or after-tax (Roth 401k). 2024 contribution limit: $23,000 ($30,500 if 50+). Employer match "
            "is free money – always contribute at least enough to get full match. Investments grow "
            "tax-deferred. Withdrawals taxed as ordinary income in retirement. Early withdrawal (before "
            "59.5) incurs 10% penalty plus taxes."
        ),
        "source": "IRS",
    },
    {
        "id": "ret_002",
        "category": "Retirement",
        "title": "IRA: Traditional vs Roth",
        "content": (
            "Traditional IRA: pre-tax contributions (deductible if income below limits), tax-deferred "
            "growth, taxed on withdrawal. Roth IRA: after-tax contributions, tax-FREE growth and "
            "withdrawals in retirement. 2024 limit: $7,000 ($8,000 if 50+). Roth income limit: "
            "$146,000 (single) / $230,000 (married). Rule of thumb: Roth if you expect higher taxes "
            "in retirement; Traditional if taxes are higher now."
        ),
        "source": "IRS",
    },
    {
        "id": "ret_003",
        "category": "Retirement",
        "title": "Required Minimum Distributions (RMDs)",
        "content": (
            "At age 73, the IRS requires minimum annual withdrawals from traditional IRA and 401(k) "
            "accounts. RMD amount = account balance / IRS life expectancy factor. Failure to take "
            "RMDs results in 25% penalty. Roth IRAs do NOT have RMDs during the owner's lifetime. "
            "Strategic Roth conversions before RMD age can reduce future tax burden."
        ),
        "source": "IRS Publication 590-B",
    },
    # ── TAX CONCEPTS ──────────────────────────────────────────────────────────
    {
        "id": "tax_001",
        "category": "Tax",
        "title": "Capital Gains Tax",
        "content": (
            "Capital gains tax applies to profit from selling investments. Short-term gains (held <1 year) "
            "taxed as ordinary income (10-37%). Long-term gains (held >1 year) taxed at preferential rates: "
            "0% (income <$47,025 single), 15% (most people), 20% (high earners). Strategy: hold investments "
            "over 1 year to qualify for lower long-term rates. Tax-loss harvesting offsets gains with losses."
        ),
        "source": "IRS",
    },
    {
        "id": "tax_002",
        "category": "Tax",
        "title": "Tax-Loss Harvesting",
        "content": (
            "Tax-loss harvesting sells losing investments to realize losses that offset capital gains, "
            "reducing your tax bill. Rules: wash-sale rule prohibits repurchasing the same or substantially "
            "identical security within 30 days before or after sale. You can buy a similar (not identical) "
            "ETF immediately. Losses above gains can offset up to $3,000 of ordinary income annually; "
            "excess carries forward indefinitely."
        ),
        "source": "IRS",
    },
    {
        "id": "tax_003",
        "category": "Tax",
        "title": "HSA: The Triple Tax Advantage",
        "content": (
            "Health Savings Accounts (HSAs) offer the best tax deal available: contributions are "
            "tax-deductible, growth is tax-free, and qualified withdrawals are tax-free. After 65, "
            "withdrawals for any purpose are taxed as income (like a traditional IRA). 2024 limits: "
            "$4,150 (individual), $8,300 (family). Invest HSA funds in low-cost index funds and pay "
            "medical expenses out-of-pocket for maximum long-term benefit."
        ),
        "source": "IRS",
    },
    # ── GOAL PLANNING ─────────────────────────────────────────────────────────
    {
        "id": "goal_001",
        "category": "Goal Planning",
        "title": "Emergency Fund",
        "content": (
            "An emergency fund covers 3-6 months of living expenses in liquid, accessible accounts. "
            "Keep it in high-yield savings accounts (HYSA) currently earning 4-5% APY. This prevents "
            "selling investments at bad times during job loss or medical emergencies. Start with $1,000 "
            "as initial milestone, then build to 3-6 months. Self-employed or single-income households "
            "should target 6-12 months."
        ),
        "source": "Personal Finance",
    },
    {
        "id": "goal_002",
        "category": "Goal Planning",
        "title": "SMART Financial Goals",
        "content": (
            "SMART goals are Specific, Measurable, Achievable, Relevant, Time-bound. Example: "
            "'Save $50,000 for a house down payment in 4 years by saving $1,042/month' is SMART. "
            "Break large goals into milestones. Automate contributions on payday. Track progress "
            "monthly. Adjust as income changes. Celebrate milestones to maintain motivation."
        ),
        "source": "CFP Board",
    },
    {
        "id": "goal_003",
        "category": "Goal Planning",
        "title": "The 50/30/20 Budget Rule",
        "content": (
            "The 50/30/20 rule allocates after-tax income: 50% to needs (rent, utilities, groceries), "
            "30% to wants (dining, entertainment, travel), 20% to savings and debt repayment. In "
            "high-cost cities, adjust to 60/20/20. The key insight is automating the 20% savings "
            "first ('pay yourself first') before discretionary spending."
        ),
        "source": "Elizabeth Warren",
    },
    # ── MARKET CONCEPTS ───────────────────────────────────────────────────────
    {
        "id": "mkt_001",
        "category": "Market Concepts",
        "title": "Bull vs Bear Markets",
        "content": (
            "A bull market is a sustained period of rising prices (20%+ gain from recent low). "
            "Bear markets are 20%+ declines from recent highs. Since 1928, the S&P 500 has had "
            "26 bear markets, averaging -36% decline over 9.6 months. Bull markets average +114% "
            "over 2.7 years. Historically, staying invested through bear markets outperforms "
            "trying to time the market."
        ),
        "source": "Hartford Funds",
    },
    {
        "id": "mkt_002",
        "category": "Market Concepts",
        "title": "Dollar-Cost Averaging (DCA)",
        "content": (
            "Dollar-cost averaging invests a fixed amount at regular intervals regardless of price. "
            "This removes emotion and timing risk: you buy more shares when prices are low and fewer "
            "when high, potentially lowering average cost. DCA is the strategy behind 401(k) payroll "
            "deductions. Studies show DCA vs lump-sum: lump-sum wins ~67% of the time due to markets "
            "trending upward, but DCA reduces regret and encourages consistency."
        ),
        "source": "Vanguard",
    },
    {
        "id": "mkt_003",
        "category": "Market Concepts",
        "title": "Market Capitalization",
        "content": (
            "Market cap = shares outstanding × stock price. Categories: Mega-cap (>$200B, e.g., Apple, "
            "Microsoft), Large-cap ($10-200B), Mid-cap ($2-10B), Small-cap ($300M-2B), Micro-cap (<$300M). "
            "Large-caps are more stable; small-caps historically offer higher returns with higher risk. "
            "A diversified portfolio holds exposure across cap sizes."
        ),
        "source": "Investopedia",
    },
    {
        "id": "mkt_004",
        "category": "Market Concepts",
        "title": "Economic Indicators Every Investor Should Know",
        "content": (
            "Key indicators: GDP Growth (>2% healthy), Unemployment Rate (<5% healthy), CPI/Inflation "
            "(Fed target 2%), Federal Funds Rate (central bank lending rate), Yield Curve "
            "(inverted curve often precedes recession), PMI (>50 = expansion). The Fed raises rates "
            "to fight inflation, which typically pressures stock valuations. Monitor monthly jobs report "
            "and CPI releases for market-moving data."
        ),
        "source": "Federal Reserve",
    },
    # ── COMMON MISTAKES ───────────────────────────────────────────────────────
    {
        "id": "mis_001",
        "category": "Common Mistakes",
        "title": "Top Investing Mistakes to Avoid",
        "content": (
            "1. Trying to time the market (even pros fail). 2. Panic selling during downturns – "
            "missing the 10 best days in S&P 500 (often during crashes) cuts returns by 50%+. "
            "3. Ignoring fees (1% difference = 26% less wealth over 30 years). 4. Concentration risk "
            "(single stocks). 5. Neglecting tax efficiency. 6. Starting too late. 7. Investing "
            "borrowed money. 8. FOMO buying at peak prices."
        ),
        "source": "Bogleheads",
    },
    {
        "id": "mis_002",
        "category": "Common Mistakes",
        "title": "Behavioral Finance: Emotions and Investing",
        "content": (
            "Common cognitive biases: Loss aversion (losses hurt 2x more than gains feel good), "
            "Recency bias (overweighting recent performance), Confirmation bias (seeking info that "
            "confirms existing views), Anchoring (fixating on purchase price), Herd mentality "
            "(following the crowd). Solutions: automate investments, write an Investment Policy "
            "Statement (IPS), review portfolio no more than quarterly."
        ),
        "source": "Nobel Economics (Kahneman)",
    },
    # ── ALTERNATIVE INVESTMENTS ───────────────────────────────────────────────
    {
        "id": "alt_001",
        "category": "Alternatives",
        "title": "REITs: Real Estate Investment Trusts",
        "content": (
            "REITs own income-producing real estate and must distribute 90%+ of taxable income as "
            "dividends. Types: Equity (properties), Mortgage (loans), Hybrid. Publicly traded REITs "
            "offer real estate exposure without direct ownership. Yields often 3-6%. Major REITs: "
            "Realty Income (O), Public Storage (PSA), Prologis (PLD). REITs provide inflation hedge "
            "and diversification from stocks."
        ),
        "source": "NAREIT",
    },
    {
        "id": "alt_002",
        "category": "Alternatives",
        "title": "Cryptocurrency: Risks and Considerations",
        "content": (
            "Cryptocurrency (Bitcoin, Ethereum) offers potential for high returns but with extreme "
            "volatility (Bitcoin has lost 80%+ multiple times). It is NOT a replacement for traditional "
            "investments for most people. If including crypto, most advisors suggest no more than "
            "1-5% of portfolio. Bitcoin ETFs now available in US (approved Jan 2024). Understand "
            "tax implications: crypto gains taxed as property."
        ),
        "source": "SEC",
    },
    # ── INTERNATIONAL ─────────────────────────────────────────────────────────
    {
        "id": "int_001",
        "category": "International",
        "title": "International Diversification",
        "content": (
            "US stocks represent ~60% of global market cap but 100% of many US investor portfolios. "
            "Developed market ETFs (VEA, EFA) cover Europe, Japan, Australia. Emerging market ETFs "
            "(VWO, EEM) cover China, India, Brazil. Historical data shows international diversification "
            "reduces volatility. Currency risk and political risk are additional considerations. "
            "Common allocation: 20-40% of equity in international."
        ),
        "source": "Vanguard",
    },
    # ── INSURANCE ─────────────────────────────────────────────────────────────
    {
        "id": "ins_001",
        "category": "Insurance",
        "title": "Life Insurance Basics",
        "content": (
            "Term life insurance provides coverage for a set period (10, 20, 30 years) at lower cost. "
            "Whole life/Universal life combine insurance with investment component at much higher cost. "
            "Most financial advisors recommend: buy term, invest the difference. Coverage rule of thumb: "
            "10-12x annual income. Essential if you have dependents relying on your income."
        ),
        "source": "CFP Board",
    },
    # ── DEBT MANAGEMENT ───────────────────────────────────────────────────────
    {
        "id": "dbt_001",
        "category": "Debt Management",
        "title": "Good Debt vs Bad Debt",
        "content": (
            "Good debt funds appreciating assets or builds human capital: mortgages (real estate "
            "appreciation + tax deduction), student loans for high-ROI careers. Bad debt is for "
            "depreciating items at high rates: credit cards (20-30% APR), auto loans for luxury cars, "
            "payday loans. Priority order: 1) High-interest debt payoff, 2) Emergency fund, "
            "3) Employer 401k match, 4) Other investing."
        ),
        "source": "Personal Finance",
    },
    {
        "id": "dbt_002",
        "category": "Debt Management",
        "title": "Debt Avalanche vs Snowball Method",
        "content": (
            "Debt Avalanche: pay minimum on all debts, extra to highest-interest first. Mathematically "
            "optimal – saves most money. Debt Snowball: pay minimum on all, extra to smallest balance "
            "first. Provides psychological wins. Research shows snowball works better for people who "
            "struggle with motivation, despite being less efficient. Choose based on your psychology, "
            "not just math."
        ),
        "source": "Dave Ramsey / Financial Research",
    },
    # ── ESTATE PLANNING ───────────────────────────────────────────────────────
    {
        "id": "est_001",
        "category": "Estate Planning",
        "title": "Basic Estate Planning",
        "content": (
            "Essential documents: Will (distributes assets, names guardian for children), Durable Power "
            "of Attorney (financial decisions if incapacitated), Healthcare Proxy/Living Will (medical "
            "decisions), Beneficiary Designations (override will for retirement accounts, life insurance). "
            "Trusts avoid probate and provide more control. Estate tax (federal) applies to estates "
            ">$13.6M (2024)."
        ),
        "source": "CFP Board",
    },
    # ── FINANCIAL RATIOS ──────────────────────────────────────────────────────
    {
        "id": "rat_001",
        "category": "Analysis",
        "title": "Key Financial Ratios for Beginners",
        "content": (
            "P/E Ratio (Price/Earnings): how much you pay per dollar of earnings. S&P 500 average ~21x. "
            "PEG Ratio: P/E divided by earnings growth rate; PEG <1 may indicate undervaluation. "
            "Price-to-Book (P/B): price vs book value per share; <1 potentially undervalued. "
            "Debt-to-Equity (D/E): total debt / shareholders equity; higher = more leveraged. "
            "Return on Equity (ROE): net income / equity; >15% generally considered strong."
        ),
        "source": "CFA Institute",
    },
    # ── ADDITIONAL TOPICS ─────────────────────────────────────────────────────
    {
        "id": "add_001",
        "category": "Fundamentals",
        "title": "Understanding Stock Market Orders",
        "content": (
            "Market order: executes immediately at current market price. Limit order: executes only "
            "at specified price or better. Stop-loss order: sells when price falls to stop price, "
            "protecting downside. Stop-limit: combines stop and limit. For most long-term investors, "
            "limit orders are preferred to control execution price. Avoid market orders for thinly "
            "traded securities to prevent slippage."
        ),
        "source": "FINRA",
    },
    {
        "id": "add_002",
        "category": "Stocks",
        "title": "Stock Splits and What They Mean",
        "content": (
            "A stock split increases share count while proportionally reducing price (e.g., 2-for-1 "
            "split: $200 stock becomes 2 shares at $100 each). Your total investment value doesn't "
            "change. Companies split to make shares more accessible to retail investors. Reverse "
            "splits reduce share count, increase price – often done to avoid exchange delisting. "
            "Historical splits don't predict future performance."
        ),
        "source": "Investopedia",
    },
    {
        "id": "add_003",
        "category": "Market Concepts",
        "title": "Sector Rotation Strategy",
        "content": (
            "Different sectors outperform at different economic cycle stages. Early recovery: "
            "Consumer discretionary, financials. Mid-cycle expansion: Technology, industrials. "
            "Late cycle: Energy, materials, healthcare. Recession: Utilities, consumer staples, "
            "healthcare (defensive sectors). Understanding the cycle helps position portfolios, "
            "though timing is difficult even for professionals."
        ),
        "source": "Fidelity Investments",
    },
    {
        "id": "add_004",
        "category": "ETFs",
        "title": "Factor Investing: Smart Beta ETFs",
        "content": (
            "Factor investing targets specific return drivers: Value (low P/B, P/E), Momentum "
            "(recent price trends), Quality (high ROE, stable earnings), Low Volatility (less "
            "price swings), Size (small caps). Smart Beta ETFs offer factor exposure at lower cost "
            "than active funds. Academic research (Fama-French) shows value and small-cap factors "
            "have historically generated excess returns over time."
        ),
        "source": "BlackRock",
    },
]


def get_knowledge_base() -> list[dict]:
    """Return the full knowledge base."""
    return FINANCIAL_KNOWLEDGE


def get_by_category(category: str) -> list[dict]:
    """Filter articles by category."""
    return [a for a in FINANCIAL_KNOWLEDGE if a["category"].lower() == category.lower()]


def get_all_categories() -> list[str]:
    """Return unique categories."""
    seen: set[str] = set()
    return [
        a["category"]
        for a in FINANCIAL_KNOWLEDGE
        if not (a["category"] in seen or seen.add(a["category"]))
    ]
