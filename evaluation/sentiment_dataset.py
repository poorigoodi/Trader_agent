"""Hand-labelled crypto headlines for scoring the sentiment classifier.

PROVENANCE -- READ THIS BEFORE TRUSTING ANY SCORE FROM IT
---------------------------------------------------------
These headlines were WRITTEN BY HAND for this evaluation. They are not
scraped, and they are not real published articles. The news collector is
roadmap Phase 6 and does not exist yet, so there is no real feed to sample.

What that means for the numbers:
  * They are representative in FORM (typical crypto-desk headline phrasing,
    the event types that actually move these five assets) but they are not a
    random sample of real news.
  * A classifier's accuracy here is a sanity check, not a production
    estimate. Real headlines are longer, noisier, more ambiguous, and often
    carry sarcasm or hedging that these do not.
  * Because a human wrote both the headlines and the labels, there is an
    obvious risk of writing headlines that are easier to classify than
    reality. Ambiguous and mixed-signal cases are included deliberately to
    push back on that, and they are marked ``hard=True``.

Replace this with sampled, independently-labelled real headlines when Phase 6
lands. Until then, treat every score computed from it as provisional.

Labels are from the perspective of a trader HOLDING the asset:
  POSITIVE - would plausibly support the price
  NEGATIVE - would plausibly pressure the price
  NEUTRAL  - informational, or genuinely two-sided
"""

from __future__ import annotations

from dataclasses import dataclass

from app.news.sentiment import Sentiment


@dataclass(frozen=True)
class LabelledHeadline:
    text: str
    label: Sentiment
    hard: bool = False  # ambiguous / mixed-signal, included on purpose


POS = Sentiment.POSITIVE
NEG = Sentiment.NEGATIVE
NEU = Sentiment.NEUTRAL


HEADLINES: tuple[LabelledHeadline, ...] = (
    # ---------------- clearly negative: security and legal ----------------
    LabelledHeadline("Major DeFi protocol hacked, $120M drained from liquidity pools", NEG),
    LabelledHeadline("Exchange halts all BTC withdrawals after suspected breach", NEG),
    LabelledHeadline("Bridge exploit sees 40,000 ETH stolen overnight", NEG),
    LabelledHeadline("SEC sues major exchange over unregistered securities offering", NEG),
    LabelledHeadline("Regulator announces crackdown on offshore crypto derivatives", NEG),
    LabelledHeadline("Binance delists XRP perpetual futures effective next week", NEG),
    LabelledHeadline("Lending platform files for bankruptcy, freezes customer funds", NEG),
    LabelledHeadline("Critical vulnerability disclosed in popular smart contract library", NEG),
    LabelledHeadline("Court rejects appeal, exchange ordered to cease US operations", NEG),
    LabelledHeadline("Validator outage halts network block production for six hours", NEG),

    # ---------------- clearly negative: market ----------------
    LabelledHeadline("Bitcoin plunges 12% as leveraged longs face mass liquidation", NEG),
    LabelledHeadline("Ethereum tumbles below key support amid heavy selloff", NEG),
    LabelledHeadline("Record ETF outflows as institutions cut crypto exposure", NEG),
    LabelledHeadline("SOL crashes after large token unlock hits the market", NEG),
    LabelledHeadline("Analysts turn bearish on BNB following weak on-chain activity", NEG),
    LabelledHeadline("Crypto market sheds $200B in worst week since 2022", NEG),
    LabelledHeadline("Whale dumps 50,000 BTC, triggering cascading liquidations", NEG),

    # ---------------- clearly positive: adoption and approval ----------------
    LabelledHeadline("Spot Bitcoin ETF approval clears final regulatory hurdle", POS),
    LabelledHeadline("Major payment processor adds XRP settlement for cross-border transfers", POS),
    LabelledHeadline("Sovereign wealth fund announces $2B allocation to digital assets", POS),
    LabelledHeadline("Solana lists on three additional tier-one exchanges", POS),
    LabelledHeadline("BNB Chain announces partnership with global retail network", POS),
    LabelledHeadline("Ethereum upgrade cuts transaction fees by 60% post-launch", POS),
    LabelledHeadline("Institutional inflows hit record high for fourth straight week", POS),
    LabelledHeadline("Regulator approves staking services for licensed custodians", POS),

    # ---------------- clearly positive: market ----------------
    LabelledHeadline("Bitcoin surges to new all-time high above $120,000", POS),
    LabelledHeadline("ETH rallies 15% on strong network activity and ETF speculation", POS),
    LabelledHeadline("SOL soars after major protocol migration completes successfully", POS),
    LabelledHeadline("XRP climbs on renewed optimism over regulatory clarity", POS),
    LabelledHeadline("Crypto market rebounds sharply, recovering weekly losses", POS),
    LabelledHeadline("BNB breaks out to six-month high on record burn", POS),

    # ---------------- clearly neutral ----------------
    LabelledHeadline("Bitcoin trades sideways as markets await Fed decision", NEU),
    LabelledHeadline("Ethereum holds steady near $2,500 in quiet weekend session", NEU),
    LabelledHeadline("Exchange publishes monthly proof-of-reserves attestation", NEU),
    LabelledHeadline("Developers outline roadmap for next protocol upgrade", NEU),
    LabelledHeadline("Crypto market little changed ahead of CPI print", NEU),
    LabelledHeadline("Weekly derivatives volume report shows mixed activity", NEU),
    LabelledHeadline("Conference announces speaker lineup for annual summit", NEU),
    LabelledHeadline("Network completes scheduled maintenance window as planned", NEU),
    LabelledHeadline("SOL range-bound between $95 and $105 for third session", NEU),
    LabelledHeadline("Analyst note reviews historical volatility across major assets", NEU),

    # ---------------- HARD: negation ----------------
    LabelledHeadline("Exchange denies hack rumours, confirms all funds are safe", POS, hard=True),
    LabelledHeadline("No evidence of exploit found after routine security audit", NEU, hard=True),
    LabelledHeadline("Regulator dismisses case against exchange, no penalty imposed", POS, hard=True),
    LabelledHeadline("Reports of insolvency denied by company spokesperson", NEU, hard=True),

    # ---------------- HARD: mixed signals ----------------
    LabelledHeadline(
        "Bitcoin rallies 8% despite ongoing regulatory investigation", POS, hard=True
    ),
    LabelledHeadline(
        "Exchange wins lawsuit but announces delisting of twelve tokens", NEG, hard=True
    ),
    LabelledHeadline(
        "ETF approval granted, though inflows disappoint on first day", NEU, hard=True
    ),
    LabelledHeadline(
        "Protocol recovers stolen funds after exploit, users made whole", NEU, hard=True
    ),
    LabelledHeadline(
        "SOL jumps on listing news but faces large unlock next month", NEU, hard=True
    ),

    # ---------------- HARD: sarcasm / hedging / forward-looking ----------------
    LabelledHeadline(
        "Another 'sure thing' altcoin collapses, retail left holding the bag", NEG, hard=True
    ),
    LabelledHeadline(
        "Analysts warn rally may be running out of steam", NEG, hard=True
    ),
    LabelledHeadline(
        "Traders position cautiously ahead of major token unlock", NEG, hard=True
    ),
    LabelledHeadline(
        "Could this be the bottom? Analysts remain divided", NEU, hard=True
    ),
    LabelledHeadline(
        "Bitcoin dominance rises as altcoins underperform", NEU, hard=True
    ),

    # ---------------- HARD: crypto-specific vocabulary ----------------
    LabelledHeadline("Large token unlock scheduled for next Tuesday", NEG, hard=True),
    LabelledHeadline("Exchange announces reduced maker fees for high-volume traders", POS, hard=True),
    LabelledHeadline("Funding rates turn deeply negative across perpetual markets", NEG, hard=True),
    LabelledHeadline("Open interest hits record high amid rising volatility", NEU, hard=True),
    LabelledHeadline("Stablecoin briefly depegs to $0.94 before recovering", NEG, hard=True),
)


def as_rows() -> list[dict]:
    return [
        {"text": h.text, "label": h.label.value, "hard": h.hard} for h in HEADLINES
    ]


def label_counts() -> dict[str, int]:
    counts: dict[str, int] = {}
    for headline in HEADLINES:
        counts[headline.label.value] = counts.get(headline.label.value, 0) + 1
    return counts
