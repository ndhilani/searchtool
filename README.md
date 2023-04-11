# searchtool
 A basic machine learning model to rank documents based on their relevance to a query.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string

class SimpleSearchEngine:
    def __init__(self, documents, preprocess=False):
        self.preprocess = preprocess
        if preprocess:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            documents = [self.preprocess_text(doc) for doc in documents]
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = self.vectorizer.fit_transform(documents)

    def preprocess_text(self, text):
        # Tokenize the text
        tokens = word_tokenize(text)
        # Convert to lowercase and remove punctuation
        tokens = [token.lower() for token in tokens if token not in string.punctuation]
        # Remove stopwords
        tokens = [token for token in tokens if token not in self.stop_words]
        # Stem the words
        tokens = [self.stemmer.stem(token) for token in tokens]
        return ' '.join(tokens)

    def search(self, query, top_k=5, min_similarity=0.0):
        if self.preprocess:
            query = self.preprocess_text(query)
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        # Filter results based on minimum similarity threshold
        results = [(idx, similarity) for idx, similarity in enumerate(similarities) if similarity >= min_similarity]
        # Sort results by similarity score
        results.sort(key=lambda x: x[1], reverse=True)
        # Return the top k results
        return results[:top_k]

# Sample documents
documents = [
    "The quick brown fox jumps over the lazy dog.",
    "The quick brown fox is fast and agile.",
    "The lazy dog is slow and sleepy.",
    "The cat is curious and playful.",
    "The dog and the cat are friends.",
    "The fox, the dog, and the cat are playing together."
]

# Initialize the search engine with the documents
search_engine = SimpleSearchEngine(documents, preprocess=True)

# Perform a search
query = "quick fox"
top_k = 3
min_similarity = 0.1
results = search_engine.search(query, top_k, min_similarity)

# Display the search results
print("Search results for query:", query)
for i, (idx, similarity) in enumerate(results):
    print(f"Rank {i+1} (Similarity: {similarity:.2f}): {documents[idx]}")
