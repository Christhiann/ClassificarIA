import os
import re
import joblib
from sentence_transformers import SentenceTransformer
import PyPDF2
import docx

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\W+', ' ', text)
    text = text.strip()
    return text

def extract_text_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def extract_text_docx(docx_path):
    doc = docx.Document(docx_path)
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return ' '.join(fullText)


embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
clf = joblib.load('modelo2_resume_logreg.pkl')

def predict_resume_category_from_text(text):
    cleaned = clean_text(text)
    emb = embedding_model.encode([cleaned])
    pred = clf.predict(emb)
    return pred[0]

def processar_curriculos(pasta):
    for arquivo in os.listdir(pasta):
        caminho = os.path.join(pasta, arquivo)

        if arquivo.lower().endswith('.pdf'):
            texto = extract_text_pdf(caminho)
        elif arquivo.lower().endswith('.docx'):
            texto = extract_text_docx(caminho)
        else:
            continue 

        try:
            categoria = predict_resume_category_from_text(texto)
            print(f"{arquivo} -> Categoria prevista: {categoria}")
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

if __name__ == "__main__":
    pasta_curriculos = 'curriculos'  
    if os.path.exists(pasta_curriculos):
        processar_curriculos(pasta_curriculos)
    else:
        print(f"Pasta '{pasta_curriculos}' não encontrada.")
