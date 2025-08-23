import re
import unicodedata

def basic_normalize(text: str) -> str:
    # NFKC unicode normalize, lowercase, collapse spaces
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text

def strip_punct(text: str) -> str:
    return re.sub(r"[^\w\s']", "", text)
