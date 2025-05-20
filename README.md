# ClassificarIA
# 🧠 Classificador Inteligente de Currículos com IA

Este projeto implementa um sistema de **Inteligência Artificial** capaz de **ler, interpretar e classificar automaticamente currículos** em diferentes áreas da tecnologia, como Backend Developer, Data Scientist, Full Stack Developer, entre outros.

## 💡 Objetivo
Automatizar o processo de análise de currículos por meio de um modelo de machine learning que entende o conteúdo textual de arquivos em **PDF** ou **Word**, e retorna a **área profissional** do candidato com alta precisão.

## ⚙️ Tecnologias Utilizadas
- **Python**
- **SentenceTransformers** (`all-MiniLM-L6-v2`)
- **Scikit-learn** (Logistic Regression)
- **Pandas**
- **Joblib** (salvar/carregar o modelo)
- **PyPDF2** (leitura de PDFs)
- **python-docx** (leitura de .docx)

## 🧪 Resultados
- **Acurácia no teste externo**: 90,91% (10 acertos em 11 currículos não vistos)
- O modelo foi capaz de interpretar corretamente o conteúdo sem depender apenas de palavras-chave fixas.
- Suporte a arquivos `.pdf` e `.docx`.

## 📁 Como usar
# Coloque o curriculo entro da pasta curriculos
### 1. Treinamento do modelo
```bash
python treinar.py
