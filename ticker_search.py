"""
Local ticker search — fuzzy match on symbol and company name.
Covers ~220 of the most-traded US equities, ETFs, and indices.
"""

TICKER_DB = [
    # Mega-cap tech
    ("AAPL",  "Apple Inc."),
    ("MSFT",  "Microsoft Corporation"),
    ("GOOGL", "Alphabet Inc. (Google) Class A"),
    ("GOOG",  "Alphabet Inc. (Google) Class C"),
    ("AMZN",  "Amazon.com Inc."),
    ("NVDA",  "NVIDIA Corporation"),
    ("META",  "Meta Platforms Inc. (Facebook)"),
    ("TSLA",  "Tesla Inc."),
    ("AVGO",  "Broadcom Inc."),
    ("ORCL",  "Oracle Corporation"),
    ("ADBE",  "Adobe Inc."),
    ("CRM",   "Salesforce Inc."),
    ("AMD",   "Advanced Micro Devices Inc."),
    ("INTC",  "Intel Corporation"),
    ("QCOM",  "Qualcomm Inc."),
    ("TXN",   "Texas Instruments Inc."),
    ("AMAT",  "Applied Materials Inc."),
    ("MU",    "Micron Technology Inc."),
    ("LRCX",  "Lam Research Corporation"),
    ("KLAC",  "KLA Corporation"),
    ("ASML",  "ASML Holding N.V."),
    ("TSM",   "Taiwan Semiconductor Manufacturing"),
    ("NFLX",  "Netflix Inc."),
    ("SNOW",  "Snowflake Inc."),
    ("PLTR",  "Palantir Technologies Inc."),
    ("UBER",  "Uber Technologies Inc."),
    ("LYFT",  "Lyft Inc."),
    ("SNAP",  "Snap Inc."),
    ("PINS",  "Pinterest Inc."),
    ("SPOT",  "Spotify Technology S.A."),
    ("ABNB",  "Airbnb Inc."),
    ("COIN",  "Coinbase Global Inc."),
    ("HOOD",  "Robinhood Markets Inc."),
    ("RBLX",  "Roblox Corporation"),
    ("U",     "Unity Software Inc."),
    ("PATH",  "UiPath Inc."),
    ("AI",    "C3.ai Inc."),
    ("CRWD",  "CrowdStrike Holdings Inc."),
    ("PANW",  "Palo Alto Networks Inc."),
    ("FTNT",  "Fortinet Inc."),
    ("ZS",    "Zscaler Inc."),
    ("OKTA",  "Okta Inc."),
    ("NET",   "Cloudflare Inc."),
    ("DDOG",  "Datadog Inc."),
    ("MDB",   "MongoDB Inc."),
    ("TEAM",  "Atlassian Corporation"),
    ("NOW",   "ServiceNow Inc."),
    ("WDAY",  "Workday Inc."),
    ("ZM",    "Zoom Video Communications"),
    ("DOCU",  "DocuSign Inc."),
    # Finance
    ("JPM",   "JPMorgan Chase & Co."),
    ("BAC",   "Bank of America Corporation"),
    ("WFC",   "Wells Fargo & Company"),
    ("GS",    "Goldman Sachs Group Inc."),
    ("MS",    "Morgan Stanley"),
    ("C",     "Citigroup Inc."),
    ("BLK",   "BlackRock Inc."),
    ("SCHW",  "Charles Schwab Corporation"),
    ("AXP",   "American Express Company"),
    ("V",     "Visa Inc."),
    ("MA",    "Mastercard Inc."),
    ("PYPL",  "PayPal Holdings Inc."),
    ("SQ",    "Block Inc. (Square)"),
    ("AFRM",  "Affirm Holdings Inc."),
    ("BRK.B", "Berkshire Hathaway Inc. Class B"),
    ("BRK.A", "Berkshire Hathaway Inc. Class A"),
    ("CB",    "Chubb Limited"),
    ("MMC",   "Marsh & McLennan Companies"),
    # Healthcare
    ("JNJ",   "Johnson & Johnson"),
    ("UNH",   "UnitedHealth Group Inc."),
    ("LLY",   "Eli Lilly and Company"),
    ("PFE",   "Pfizer Inc."),
    ("ABBV",  "AbbVie Inc."),
    ("MRK",   "Merck & Co. Inc."),
    ("TMO",   "Thermo Fisher Scientific Inc."),
    ("ABT",   "Abbott Laboratories"),
    ("DHR",   "Danaher Corporation"),
    ("BMY",   "Bristol-Myers Squibb Company"),
    ("AMGN",  "Amgen Inc."),
    ("GILD",  "Gilead Sciences Inc."),
    ("REGN",  "Regeneron Pharmaceuticals Inc."),
    ("MRNA",  "Moderna Inc."),
    ("BNTX",  "BioNTech SE"),
    ("ISRG",  "Intuitive Surgical Inc."),
    ("SYK",   "Stryker Corporation"),
    ("BSX",   "Boston Scientific Corporation"),
    ("CVS",   "CVS Health Corporation"),
    ("CI",    "The Cigna Group"),
    ("HUM",   "Humana Inc."),
    # Consumer
    ("AMZN",  "Amazon.com Inc."),
    ("WMT",   "Walmart Inc."),
    ("COST",  "Costco Wholesale Corporation"),
    ("TGT",   "Target Corporation"),
    ("HD",    "The Home Depot Inc."),
    ("LOW",   "Lowe's Companies Inc."),
    ("MCD",   "McDonald's Corporation"),
    ("SBUX",  "Starbucks Corporation"),
    ("NKE",   "Nike Inc."),
    ("TJX",   "TJX Companies Inc."),
    ("ROST",  "Ross Stores Inc."),
    ("DG",    "Dollar General Corporation"),
    ("KO",    "The Coca-Cola Company"),
    ("PEP",   "PepsiCo Inc."),
    ("PM",    "Philip Morris International Inc."),
    ("MO",    "Altria Group Inc."),
    ("PG",    "Procter & Gamble Co."),
    ("CL",    "Colgate-Palmolive Company"),
    ("KMB",   "Kimberly-Clark Corporation"),
    ("GIS",   "General Mills Inc."),
    ("K",     "Kellanova (Kellogg)"),
    ("MDLZ",  "Mondelez International Inc."),
    ("HSY",   "The Hershey Company"),
    ("TSCO",  "Tractor Supply Company"),
    ("LULU",  "lululemon athletica inc."),
    ("RH",    "RH (Restoration Hardware)"),
    # Energy
    ("XOM",   "Exxon Mobil Corporation"),
    ("CVX",   "Chevron Corporation"),
    ("COP",   "ConocoPhillips"),
    ("EOG",   "EOG Resources Inc."),
    ("SLB",   "Schlumberger N.V. (SLB)"),
    ("OXY",   "Occidental Petroleum Corporation"),
    ("PSX",   "Phillips 66"),
    ("VLO",   "Valero Energy Corporation"),
    ("MPC",   "Marathon Petroleum Corporation"),
    ("DVN",   "Devon Energy Corporation"),
    ("HAL",   "Halliburton Company"),
    ("BKR",   "Baker Hughes Company"),
    # Industrials
    ("GE",    "GE Aerospace"),
    ("HON",   "Honeywell International Inc."),
    ("MMM",   "3M Company"),
    ("CAT",   "Caterpillar Inc."),
    ("DE",    "Deere & Company (John Deere)"),
    ("RTX",   "RTX Corporation (Raytheon)"),
    ("LMT",   "Lockheed Martin Corporation"),
    ("BA",    "The Boeing Company"),
    ("NOC",   "Northrop Grumman Corporation"),
    ("GD",    "General Dynamics Corporation"),
    ("UPS",   "United Parcel Service Inc."),
    ("FDX",   "FedEx Corporation"),
    ("CSX",   "CSX Corporation"),
    ("UNP",   "Union Pacific Corporation"),
    ("NSC",   "Norfolk Southern Corporation"),
    # Real Estate / Utilities
    ("AMT",   "American Tower Corporation"),
    ("PLD",   "Prologis Inc."),
    ("EQIX",  "Equinix Inc."),
    ("SPG",   "Simon Property Group Inc."),
    ("O",     "Realty Income Corporation"),
    ("NEE",   "NextEra Energy Inc."),
    ("DUK",   "Duke Energy Corporation"),
    ("SO",    "The Southern Company"),
    ("D",     "Dominion Energy Inc."),
    # ETFs — broad market
    ("SPY",   "SPDR S&P 500 ETF Trust"),
    ("QQQ",   "Invesco QQQ Trust (Nasdaq 100)"),
    ("IWM",   "iShares Russell 2000 ETF"),
    ("DIA",   "SPDR Dow Jones Industrial Average ETF"),
    ("VOO",   "Vanguard S&P 500 ETF"),
    ("VTI",   "Vanguard Total Stock Market ETF"),
    ("VEA",   "Vanguard FTSE Developed Markets ETF"),
    ("VWO",   "Vanguard FTSE Emerging Markets ETF"),
    ("EEM",   "iShares MSCI Emerging Markets ETF"),
    ("EFA",   "iShares MSCI EAFE ETF"),
    # ETFs — sector
    ("XLK",   "Technology Select Sector SPDR Fund"),
    ("XLF",   "Financial Select Sector SPDR Fund"),
    ("XLV",   "Health Care Select Sector SPDR Fund"),
    ("XLE",   "Energy Select Sector SPDR Fund"),
    ("XLI",   "Industrial Select Sector SPDR Fund"),
    ("XLC",   "Communication Services Select Sector SPDR"),
    ("ARKK",  "ARK Innovation ETF"),
    ("SOXX",  "iShares Semiconductor ETF"),
    ("SMH",   "VanEck Semiconductor ETF"),
    ("GLD",   "SPDR Gold Shares"),
    ("SLV",   "iShares Silver Trust"),
    ("TLT",   "iShares 20+ Year Treasury Bond ETF"),
    ("HYG",   "iShares iBoxx High Yield Corporate Bond ETF"),
    ("LQD",   "iShares iBoxx Investment Grade Corporate Bond ETF"),
    # Indices
    ("^SPX",  "S&P 500 Index"),
    ("^NDX",  "Nasdaq 100 Index"),
    ("^DJI",  "Dow Jones Industrial Average"),
    ("^VIX",  "CBOE Volatility Index (VIX)"),
    ("^RUT",  "Russell 2000 Index"),
    # International / ADRs
    ("BABA",  "Alibaba Group Holding Limited"),
    ("JD",    "JD.com Inc."),
    ("PDD",   "PDD Holdings Inc. (Pinduoduo / Temu)"),
    ("BIDU",  "Baidu Inc."),
    ("NIO",   "NIO Inc."),
    ("XPEV",  "XPeng Inc."),
    ("LI",    "Li Auto Inc."),
    ("SAP",   "SAP SE"),
    ("SHOP",  "Shopify Inc."),
    ("RY",    "Royal Bank of Canada"),
    ("TD",    "Toronto-Dominion Bank"),
    ("TM",    "Toyota Motor Corporation"),
    ("SONY",  "Sony Group Corporation"),
    ("NVS",   "Novartis AG"),
    ("HSBC",  "HSBC Holdings plc"),
    ("BP",    "BP p.l.c."),
    ("RIO",   "Rio Tinto Group"),
    ("BHP",   "BHP Group Limited"),
    ("ARM",   "Arm Holdings plc"),
    ("SMCI",  "Super Micro Computer Inc."),
    ("DELL",  "Dell Technologies Inc."),
    ("HPQ",   "HP Inc."),
    ("HPE",   "Hewlett Packard Enterprise"),
    ("IBM",   "International Business Machines"),
    ("ACN",   "Accenture plc"),
    ("INFY",  "Infosys Limited"),
    ("WIT",   "Wipro Limited"),
]

def search_tickers(query: str, max_results: int = 6) -> list[dict]:
    """
    Fuzzy search tickers by symbol or company name.
    Returns list of {symbol, name, match_type} dicts, ranked by relevance.
    """
    if not query or len(query.strip()) < 1:
        return []

    q = query.strip().upper()
    q_lower = query.strip().lower()

    exact_sym, starts_sym, starts_name, contains_name = [], [], [], []

    for symbol, name in TICKER_DB:
        sym_upper = symbol.upper()
        name_lower = name.lower()

        if sym_upper == q:
            exact_sym.append({"symbol": symbol, "name": name, "match": "symbol"})
        elif sym_upper.startswith(q):
            starts_sym.append({"symbol": symbol, "name": name, "match": "symbol"})
        elif name_lower.startswith(q_lower):
            starts_name.append({"symbol": symbol, "name": name, "match": "name"})
        elif q_lower in name_lower:
            contains_name.append({"symbol": symbol, "name": name, "match": "name"})

    results = exact_sym + starts_sym + starts_name + contains_name
    # Deduplicate by symbol
    seen, out = set(), []
    for r in results:
        if r["symbol"] not in seen:
            seen.add(r["symbol"])
            out.append(r)

    return out[:max_results]