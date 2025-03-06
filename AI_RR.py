import os
import re
import pandas as pd
import nltk
import torch
import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from sklearn.preprocessing import MinMaxScaler
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Load pre-trained Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_file):
    text = ""
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text.strip()

# Streamlit App
st.title("AI Resume Ranking System")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", accept_multiple_files=True, type=['pdf'])
if uploaded_files:
    resume_data = []
    
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        resume_data.append({"Filename": file.name, "Text": resume_text})
    
    df = pd.DataFrame(resume_data)
    
    # Process resumes
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = text.lower().strip()
        return text
    
    df['Cleaned_Resume'] = df['Text'].apply(clean_text)
    
    # Compute similarity scores
    resume_embeddings = sbert_model.encode(df['Cleaned_Resume'], convert_to_tensor=True)
    job_description = st.text_area("Enter Job Description for Ranking", placeholder="Type job description here...")
    
    if job_description:
        job_description_cleaned = clean_text(job_description)
        job_embedding = sbert_model.encode(job_description_cleaned, convert_to_tensor=True)
        similarity_scores = [util.pytorch_cos_sim(res, job_embedding).item() for res in resume_embeddings]
    
        df['Similarity_Score'] = similarity_scores
        
        # Adjust relevance threshold for better filtering
        relevance_threshold = 0.3  # Lowered threshold to allow more resumes to be ranked
        df['Relevant'] = df['Similarity_Score'].apply(lambda x: 'Relevant' if x >= relevance_threshold else 'Not Relevant')
        
        # Generate ranking scores for relevant resumes only
        scaler = MinMaxScaler(feature_range=(0, 100))
        df['Rank_Score'] = scaler.fit_transform(df[['Similarity_Score']])
        df.loc[df['Similarity_Score'] < relevance_threshold, 'Rank_Score'] = float('nan')
        
        # Display ranked resumes
        st.write("### Ranked Resumes")
        st.dataframe(df[['Filename', 'Rank_Score', 'Relevant']].sort_values(by='Rank_Score', ascending=False, na_position='last'))
