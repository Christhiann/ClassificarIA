import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import joblib
import numpy as np

# Carregue seu dataset (substitua pelo caminho correto)
df = pd.read_csv('data/gpt_dataset.csv')

# Função para limpar texto
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)      # Remove números
        text = re.sub(r'\W+', ' ', text)     # Remove caracteres especiais
        text = text.strip()
        return text
    else:
        return ""

# Aplicar limpeza no texto
df['clean_resume'] = df['Resume'].apply(clean_text)

# Gerar embeddings com SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(df['clean_resume'].tolist())

# Converter labels para numpy array (recomendado)
labels = np.array(df['Category'])

# Dividir treino/teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, labels, test_size=0.2, random_state=42, stratify=labels
)

# Treinar modelo de regressão logística
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = clf.predict(X_test)

# Avaliar resultados
print(classification_report(y_test, y_pred, digits=4))

# Salvar o modelo treinado
joblib.dump(clf, 'modelo2_resume_logreg.pkl')
print("Modelo salvo em 'modelo_resume_logreg.pkl'")
