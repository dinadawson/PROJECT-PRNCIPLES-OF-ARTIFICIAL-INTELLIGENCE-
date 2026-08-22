"""
feature_extraction.py

Converts a raw URL into the numeric features used by the new
"Web Page Phishing Detection" dataset (Hannousse & Yahiouche, 2020),
so the trained model can predict on a brand-new URL a real user provides.

HONEST LIMITATION:
This dataset's 89 columns come from THREE sources:
  1. URL string itself (e.g. length_url, nb_dots, https_token) — we CAN
     calculate these from just the URL text.
  2. Webpage content (e.g. nb_hyperlinks, login_form, iframe, popup_window)
     — these require actually fetching and parsing the live webpage HTML.
  3. External services (e.g. web_traffic, google_index, page_rank,
     whois_registered_domain, domain_age) — these require querying
     external APIs (WHOIS, search engines, ranking services).

We implement URL-based features (source 1) fully. Content-based and
external-service features (sources 2 and 3) are defaulted to 0, with a
printed note - this mirrors the same honest gap we highlighted for the
PhiUSIIL dataset, and is a natural discussion point about what a full
production system would need to add (web scraping + external API calls).
"""

import re
from urllib.parse import urlparse

# Common brand names used to check for brand impersonation in domain/path
KNOWN_BRANDS = [
    "google", "facebook", "paypal", "amazon", "apple", "microsoft",
    "netflix", "instagram", "twitter", "bank", "ebay", "yahoo"
]

SHORTENING_SERVICES = ["bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd"]


def extract_url_features(url: str) -> dict:
    """Calculate URL-based numeric features from a raw URL string.
    Returns a dictionary of feature_name -> value, using this dataset's
    exact column naming convention.
    """
    parsed = urlparse(url)
    hostname = parsed.netloc.replace("www.", "")
    path = parsed.path or ""
    features = {}

    # --- Basic length features ---
    features["length_url"] = len(url)
    features["length_hostname"] = len(hostname)

    # --- IP address check ---
    features["ip"] = 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", hostname) else 0

    # --- Character counts ---
    features["nb_dots"] = url.count(".")
    features["nb_hyphens"] = url.count("-")
    features["nb_at"] = url.count("@")
    features["nb_qm"] = url.count("?")
    features["nb_and"] = url.count("&")
    features["nb_or"] = url.count("|")
    features["nb_eq"] = url.count("=")
    features["nb_underscore"] = url.count("_")
    features["nb_tilde"] = url.count("~")
    features["nb_percent"] = url.count("%")
    features["nb_slash"] = url.count("/")
    features["nb_star"] = url.count("*")
    features["nb_colon"] = url.count(":")
    features["nb_comma"] = url.count(",")
    features["nb_semicolumn"] = url.count(";")
    features["nb_dollar"] = url.count("$")
    features["nb_space"] = url.count(" ") + url.count("%20")
    features["nb_www"] = url.lower().count("www")
    features["nb_com"] = url.lower().count(".com")
    features["nb_dslash"] = url.count("//") - 1 if url.count("//") > 0 else 0

    # --- Structural indicators ---
    features["http_in_path"] = 1 if "http" in path.lower() else 0
    features["https_token"] = 1 if parsed.scheme == "https" else 0
    digits_in_url = sum(c.isdigit() for c in url)
    digits_in_host = sum(c.isdigit() for c in hostname)
    features["ratio_digits_url"] = digits_in_url / len(url) if len(url) > 0 else 0
    features["ratio_digits_host"] = digits_in_host / len(hostname) if len(hostname) > 0 else 0
    features["punycode"] = 1 if "xn--" in hostname else 0
    features["port"] = 1 if parsed.port else 0
    features["tld_in_path"] = 1 if re.search(r"\.(com|net|org|info)", path.lower()) else 0
    features["tld_in_subdomain"] = 0  # requires subdomain parsing, kept simple
    features["abnormal_subdomain"] = 0
    features["nb_subdomains"] = max(hostname.count(".") - 1, 0)
    features["prefix_suffix"] = 1 if "-" in hostname else 0
    features["random_domain"] = 0  # would require a dictionary-word check
    features["shortening_service"] = 1 if any(s in url for s in SHORTENING_SERVICES) else 0
    features["path_extension"] = 1 if "." in path.split("/")[-1] else 0
    features["nb_redirection"] = url.count("//") - 1 if url.count("//") > 1 else 0
    features["nb_external_redirection"] = 0

    # --- Word-based features ---
    words = re.split(r"[/\-_.?=&]", url)
    words = [w for w in words if w]
    if words:
        features["length_words_raw"] = len(words)
        features["shortest_words_raw"] = min(len(w) for w in words)
        features["longest_words_raw"] = max(len(w) for w in words)
        features["avg_words_raw"] = sum(len(w) for w in words) / len(words)
    else:
        features["length_words_raw"] = 0
        features["shortest_words_raw"] = 0
        features["longest_words_raw"] = 0
        features["avg_words_raw"] = 0

    features["char_repeat"] = max(
        [len(m.group()) for m in re.finditer(r"(.)\1+", url)], default=0
    )
    features["shortest_word_host"] = min([len(p) for p in hostname.split(".") if p], default=0)
    features["longest_word_host"] = max([len(p) for p in hostname.split(".") if p], default=0)
    features["avg_word_host"] = (
        sum(len(p) for p in hostname.split(".")) / len(hostname.split(".")) if hostname else 0
    )
    path_parts = [p for p in path.split("/") if p]
    features["shortest_word_path"] = min([len(p) for p in path_parts], default=0)
    features["longest_word_path"] = max([len(p) for p in path_parts], default=0)
    features["avg_word_path"] = (
        sum(len(p) for p in path_parts) / len(path_parts) if path_parts else 0
    )

    # --- Phishing hint keywords ---
    phish_keywords = ["login", "verify", "account", "secure", "update", "confirm", "banking"]
    features["phish_hints"] = sum(1 for kw in phish_keywords if kw in url.lower())

    # --- Brand impersonation checks ---
    features["domain_in_brand"] = 1 if any(b in hostname.lower() for b in KNOWN_BRANDS) else 0
    features["brand_in_subdomain"] = 0
    features["brand_in_path"] = 1 if any(b in path.lower() for b in KNOWN_BRANDS) else 0
    features["suspicious_tld"] = 1 if hostname.split(".")[-1] in ["xyz", "top", "club", "info"] else 0
    features["statistical_report"] = 0  # requires external blacklist lookup

    return features


def build_feature_vector(url: str, reference_columns: list) -> dict:
    """Build a feature dictionary matching the training data's column
    structure. Columns we cannot calculate from the URL alone (content-based
    and external-service features) are defaulted to 0, with a printed note.
    """
    url_features = extract_url_features(url)

    full_vector = {}
    missing_cols = []
    for col in reference_columns:
        if col in url_features:
            full_vector[col] = url_features[col]
        else:
            full_vector[col] = 0  # content/external-service feature - not computed here
            missing_cols.append(col)

    if missing_cols:
        print(f"Note: {len(missing_cols)} columns require webpage content or "
              f"external services (not just the URL) and were defaulted to 0: "
              f"{missing_cols[:5]}...")

    return full_vector
