"""News layer (roadmap Phase 6). Only the sentiment classifier exists so far."""

from app.news.sentiment import (
    LexiconSentiment,
    FinBertSentiment,
    OllamaSentiment,
    Sentiment,
    SentimentClassifier,
    SentimentResult,
    build_sentiment_candidates,
)

__all__ = [
    "Sentiment",
    "SentimentResult",
    "SentimentClassifier",
    "LexiconSentiment",
    "FinBertSentiment",
    "OllamaSentiment",
    "build_sentiment_candidates",
]
