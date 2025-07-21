from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from typing import List

def tfidf_match(query: str, candidates: List[str], threshold: float = 0.3) -> List[int]:
    """
    Return the indices in `candidates` whose TF-IDF cosine similarity against `query` exceeds `threshold`.
    """
    vectorizer = TfidfVectorizer()
    
    # First row is the query, the rest are candidate texts
    matrix = vectorizer.fit_transform([query] + candidates)
    sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    
    # Pick those above threshold, sorted descending
    idxs = [i for i, s in enumerate(sims) if s > threshold]
    
    return sorted(idxs, key=lambda i: -sims[i])
