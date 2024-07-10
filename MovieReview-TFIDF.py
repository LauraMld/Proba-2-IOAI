import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import time

# Start timing
start_time = time.time()

# Load data
train_df = pd.read_csv('train.tsv', sep='\t')
test_df = pd.read_csv('test.tsv', sep='\t')

# Clean data
def clean_text(text):
    return str(text) if pd.notna(text) else ''

train_df['Phrase'] = train_df['Phrase'].apply(clean_text)
test_df['Phrase'] = test_df['Phrase'].apply(clean_text)

# Prepare data
X = train_df['Phrase']
y = train_df['Sentiment']

# Split the data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Create a pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=100000, ngram_range=(1, 3))),
    ('clf', SGDClassifier(loss='modified_huber', penalty='l2', alpha=1e-4, random_state=42, max_iter=100, tol=1e-3))
])

# Train the model
model.fit(X_train, y_train)

# Validate the model
val_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_accuracy:.4f}")

# Predict on test set
test_pred = model.predict(test_df['Phrase'])

# Prepare submission
submission = pd.DataFrame({'PhraseId': test_df['PhraseId'], 'Sentiment': test_pred})
submission.to_csv('submission.csv', index=False)

# Print total time taken
print(f"Total time: {time.time() - start_time:.2f} seconds")Mo
