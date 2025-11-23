from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse

class TextMetaTransformer:
    def __init__(self, max_features=5000, ngram_range=(1,2), min_df=2, stop_words='english'):
        self.vect = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df, stop_words=stop_words)

    def fit(self, X, y=None):
        texts = X['email_content'] if hasattr(X, 'get') else X
        self.vect.fit(texts)
        return self

    def transform(self, X):
        texts = X['email_content'] if hasattr(X, 'get') else X
        Xb = self.vect.transform(texts)
        if hasattr(X, 'get'):
            meta = X['domain_flag'].astype(int).values.reshape(-1,1)
            meta_sparse = sparse.csr_matrix(meta)
            return sparse.hstack([Xb, meta_sparse], format='csr')
        return Xb

    def fit_transform(self, X, y=None):
        self.fit(X,y)
        return self.transform(X)
