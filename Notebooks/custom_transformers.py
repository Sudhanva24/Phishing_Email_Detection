import pandas as pd
import re
import unicodedata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
def clean_text(text):

    # Import stop words
    # Convert text to lowercase and split into words
    words = text.lower().split()
    
    # Remove stop words
    words = [word for word in words if word not in stop_words]
    
    # Join words back together
    text = ' '.join(words)
    
    # Convert to string
    text = str(text)
    # Remove punctuation except ! and $
    text = re.sub(r'[^\w\s!$]', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '', text)
    
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    
    return text
# Step 2: Custom Transformer to Add New Features
class FeatureAdder(BaseEstimator, TransformerMixin):
    """Custom Transformer to add new numeric features based on text analysis"""
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()  # Preserve original DataFrame
        
        # Apply text cleaning and create 'clean_text' column
        X_transformed['clean_text'] = X_transformed['mail'].apply(clean_text)
        
        # Count occurrences of '$' and '!'
        X_transformed['dollar_count'] = X_transformed['mail'].str.count(r'\$')
        X_transformed['exclamation_count'] = X_transformed['mail'].str.count(r'!')
        
        # Count diacritics
        def count_diacritics(text):
            return sum(1 for char in text if unicodedata.combining(char) or 
                       (unicodedata.category(char).startswith('L') and 
                        unicodedata.decomposition(char)))
        
        X_transformed['diacritic_count'] = X_transformed['clean_text'].apply(count_diacritics)
        
        return X_transformed

# Step 3: NMF Transformer 
class NMFTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.nmf = NMF(n_components=n_components, random_state=random_state)
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')

    def fit(self, X, y=None):
        text_matrix = self.tfidf.fit_transform(X)
        self.nmf.fit(text_matrix)
        return self

    def transform(self, X):
        text_matrix = self.tfidf.transform(X)
        nmf_output = self.nmf.transform(text_matrix)
        return nmf_output.argmax(axis=1).reshape(-1, 1)
# Step 5: Create Column Transformer
column_transformer = ColumnTransformer(
    transformers=[
        ('nmf', NMFTransformer(), 'clean_text'),  # Apply NMF on clean_text
        ('dollar', 'passthrough', ['dollar_count']),  # Keep dollar_count
        ('exclamation', 'passthrough', ['exclamation_count']),  # Keep exclamation_count
        ('clean_text', 'passthrough', ['clean_text']),
        ('diacritic', 'passthrough', ['diacritic_count']) # Keep diacritic_count
    ],
    remainder='passthrough'  # Keep other columns unchanged
)

# Step 6: Create Full Pipeline
pipeline = Pipeline([
    ('feature_adder', FeatureAdder()),  # Add new features and clean text
    ('feature_extraction', column_transformer)  # Apply NMF and keep extracted features
])

# Step 7: Apply Pipeline on a Sample DataFrame
df = pd.DataFrame({
    'mail': ["Win $1000 now!!! Click here", "Hello, let's meet tomorrow!", "Urgent! Your account is at risk!"],
    'type': ['Phishing Email', 'Safe Email', 'Phishing Email']
})

df_transformed = pipeline.fit_transform(df)
df_transformed
# Convert back to DataFrame with proper column names
