from pdfminer.high_level import extract_text
import openai
from openai import OpenAI

import os
from dotenv import load_dotenv
import pandas as pd
import pickle
from scipy.spatial.distance import cosine
import requests
import json

# Load environment variables from .env file
load_dotenv()

# Access the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
print(openai_api_key)

# Set the OpenAI API key

# Define the file path with spaces
file_path = "/Users/oscarmoxon/Desktop/AI Projects/hackathon/SmartRedBox/data_files"
directory_path = "/Users/oscarmoxon/Desktop/AI Projects/hackathon/SmartRedBox/processed_files"

client = openai.Client()
print(client)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    # Implement your PDF text extraction logic here
    return extract_text(pdf_path)

# Function to get embeddings from text
def get_embeddings(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(model=model,
    input=[text])
    return response['data'][0]['embedding']

def extract_and_store_text_from_pdfs(directory, storage_directory):
    text_store = {}
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            path = os.path.join(directory, filename)
            try:
                text = extract_text(path)
                text_store[filename] = text
                # Store each document's text in a separate file
                with open(os.path.join(storage_directory, f"{filename}.txt"), 'w', encoding='utf-8') as f:
                    f.write(text)
            except Exception as e:
                print(f"Failed to process {filename}: {str(e)}")

output = extract_and_store_text_from_pdfs(file_path, directory_path)

embeddings_store = {}

def get_embeddings(text, model="text-embedding-ada-002"):
    """
    Fetches embeddings for the provided text using the specified model.
    This function is adjusted to comply with openai API version >= 1.0.0.
    """
    # Making sure to handle texts longer than the model's max token limit
    # The token limit for the embedding model can be checked via OpenAI's API documentation
    # Ada model typically supports up to 8192 tokens
    response = client.embeddings.create(model=model,
    input=text[:2048])
    return response['data'][0]['embedding']

def embed_and_store_texts(storage_directory, embedding_storage_path):
    embeddings_store = {}
    for filename in os.listdir(storage_directory):
        if filename.endswith(".txt"):
            with open(os.path.join(storage_directory, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                try:
                    embedding = get_embeddings(text)
                    embeddings_store[filename] = embedding
                except Exception as e:
                    print(f"Failed to generate embedding for {filename}: {str(e)}")
    
    # Save embeddings to disk
    with open(embedding_storage_path, 'wb') as f:
        pickle.dump(embeddings_store, f)

# Example usage
embeddings = embed_and_store_texts(directory_path, 'embeddings_store.pkl')