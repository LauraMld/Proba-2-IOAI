import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from gensim.models import FastText
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

# Încărcăm datele
train_df = pd.read_csv('nitro/train.csv')
test_df = pd.read_csv('nitro/test.csv')

# Combinăm titlul și conținutul
train_df['text'] = train_df['title'].fillna('') + ' ' + train_df['content'].fillna('')
test_df['text'] = test_df['title'].fillna('') + ' ' + test_df['content'].fillna('')

# Împărțim datele de antrenament
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['class'])

# Pregătim datele pentru FastText
sentences = [text.split() for text in train_df['text']]

# Antrenăm modelul FastText
model = FastText(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=1)

# Funcție pentru a obține vectorul mediu al unui text
def get_mean_vector(text):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) > 0:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Pregătim datele pentru clasificator
X_train = np.array([get_mean_vector(text) for text in tqdm(train_df['text'])])
X_val = np.array([get_mean_vector(text) for text in tqdm(val_df['text'])])
X_test = np.array([get_mean_vector(text) for text in tqdm(test_df['text'])])

y_train = train_df['class']
y_val = val_df['class']

# Antrenăm clasificatorul
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

# Evaluăm pe setul de validare
y_pred = clf.predict(X_val)
print("Balanced Accuracy pe setul de validare:", balanced_accuracy_score(y_val, y_pred))
print("\nRaport de clasificare:")
print(classification_report(y_val, y_pred))

# Facem predicții pe setul de test
test_predictions = clf.predict(X_test)

# Creăm fișierul de submisie
submission = pd.DataFrame({
    'id': test_df['id'],
    'class': test_predictions
})

# Salvăm fișierul de submisie
submission.to_csv('nitro/submission.csv', index=False)

print("Fișierul de submisie a fost creat cu succes.")
