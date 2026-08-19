"""
feature_extraction.py

THIS FILE ANSWERS SIR'S ORIGINAL QUESTION:
"Dataset has 54 columns, but when a real user uses the model, they only give
1 URL. How do you get 54 columns from 1 URL?"

This module takes a raw URL (just text, like "http://fake-bank.com") and
automatically calculates a set of numeric features from it — the same KIND
of features that exist in the training dataset — so the model can make a
prediction on a brand new URL it has never seen before.

IMPORTANT / HONEST LIMITATION:
The PhiUSIIL dataset's 54 columns include features from TWO sources:
  1. The URL string itself (e.g. URLLength, IsHTTPS, NoOfDigitsInURL) — we
     CAN calculate these from just the URL text, no internet needed.
  2. The webpage's actual source code (e.g. NoOfImage, NoOfJS, LineOfCode,
     HasSocialNet) — these require actually FETCHING the live webpage and
     reading its HTML, which needs an internet connection and is a bigger
     step (web scraping).

For this project, we implement the URL-based features fully (option 1), and
explain to sir that page-content features (option 2) would need an extra
scraping step (e.g. using the `requests` and `BeautifulSoup` libraries) as
a further extension.
"""

import re
from difflib import SequenceMatcher
from urllib.parse import urlparse


# A small reference list of well-known, popular domains. In the real PhiUSIIL
# dataset, URLSimilarityIndex is computed against a much larger reference
# database - this is a simplified version for demo purposes.
POPULAR_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "instagram.com", "twitter.com",
    "wikipedia.org", "amazon.com", "yahoo.com", "whatsapp.com", "netflix.com",
    "microsoft.com", "apple.com", "linkedin.com", "paypal.com", "ebay.com",
    "reddit.com", "tiktok.com", "bing.com", "office.com", "github.com",
    "gmail.com", "outlook.com", "dropbox.com", "spotify.com", "adobe.com",
]


def calculate_url_similarity_index(domain: str) -> float:
    """Estimate how similar this domain is to a known popular domain.
    Returns a score from 0-100 (100 = matches a popular domain exactly,
    lower = less similar to anything well-known).

    This mimics the intuition behind PhiUSIIL's URLSimilarityIndex: phishing
    sites often use domains that look ALMOST like a real brand (e.g.
    "paypa1.com" vs "paypal.com"), so we measure the best-match similarity
    against our reference list.
    """
    domain = domain.lower().replace("www.", "")
    best_score = 0.0
    for known in POPULAR_DOMAINS:
        ratio = SequenceMatcher(None, domain, known).ratio()
        best_score = max(best_score, ratio)
    return round(best_score * 100, 2)


def extract_url_features(url: str) -> dict:
    """Calculate URL-based numeric features from a raw URL string.
    Returns a dictionary of feature_name -> value.
    """
    parsed = urlparse(url)
    domain = parsed.netloc
    features = {}

    # Basic length-based features
    features["URLLength"] = len(url)
    features["DomainLength"] = len(domain)

    # Character composition
    digits = sum(c.isdigit() for c in url)
    letters = sum(c.isalpha() for c in url)
    special_chars = len(url) - digits - letters

    features["NoOfDegitsInURL"] = digits
    features["DegitRatioInURL"] = digits / len(url) if len(url) > 0 else 0
    features["NoOfLettersInURL"] = letters
    features["LetterRatioInURL"] = letters / len(url) if len(url) > 0 else 0
    features["NoOfOtherSpecialCharsInURL"] = special_chars
    features["SpacialCharRatioInURL"] = special_chars / len(url) if len(url) > 0 else 0

    # Security indicator
    features["IsHTTPS"] = 1 if parsed.scheme == "https" else 0

    # Structural indicators known to matter for phishing detection
    features["NoOfSubDomain"] = max(domain.count(".") - 1, 0)
    features["HasObfuscation"] = 1 if "%" in url else 0
    features["NoOfObfuscatedChar"] = url.count("%")
    features["NoOfEqualsInURL"] = url.count("=")
    features["NoOfQMarkInURL"] = url.count("?")
    features["NoOfAmpersandInURL"] = url.count("&")

    # Suspicious patterns
    features["HasIPAddress"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain) else 0
    features["HasAtSymbol"] = 1 if "@" in url else 0

    tld = domain.split(".")[-1] if "." in domain else ""
    features["TLDLength"] = len(tld)

    # New: similarity to known popular domains
    features["URLSimilarityIndex"] = calculate_url_similarity_index(domain)

    return features


def build_feature_vector(url: str, reference_columns: list) -> dict:
    """Build a feature dictionary that matches the training data's column
    structure. Any column we cannot calculate from the URL alone (i.e. the
    webpage-content-based features) is filled with a neutral default (0),
    with a printed note explaining why — this keeps the demo honest rather
    than pretending we have data we don't.
    """
    url_features = extract_url_features(url)

    full_vector = {}
    missing_cols = []
    for col in reference_columns:
        if col in url_features:
            full_vector[col] = url_features[col]
        else:
            full_vector[col] = 0  # neutral default — see limitation note above
            missing_cols.append(col)

    if missing_cols:
        print(f"Note: {len(missing_cols)} columns require webpage source-code "
              f"scraping (not just the URL) and were defaulted to 0: {missing_cols[:5]}...")

    return full_vector
