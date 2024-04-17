import json
import os
import pickle
import random
import time
import pandas as pd
from dotenv import load_dotenv
import numpy as np

# PDF processing
# from pdfminer.high_level import extract_text
import PyPDF2

# Machine Learning and Mathematics
from scipy.spatial.distance import cosine

# Langchain imports
import langchain
# from langchain.llms import OpenAI
# from langchain.chains import LLMChain, MapReduceChain, summarize as load_summarize_chain
# from langchain.prompts import PromptTemplate
# from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
# from langchain.docstore.document import Document

from openai import OpenAI

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

def get_embeddings(texts, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=texts, model=model)
        return [item['embedding'] for item in response['data']]
    except Exception as e:
        print(f"Error fetching embeddings: {e}")
        return []

api_key = os.getenv("OPENAI_API_KEY")


results = [
        {"ID": "REF992034", "From": "Edward H. Scissor", "Date": "16-04-2024 18:33", "Topic": "Banning Hands Export.", "Priority": "10", "Dept": "DBT", "Suggested Respondent": "🔘  Greg Hands", "Action": "None"},
        {"ID": "REF110293", "From": "Linda Sun", "Date": "16-04-2024 13:30", "Topic": "Will YC Move to UK?", "Priority": "9", "Dept": "DoT", "Suggested Respondent": "🔘  Michelle Donelan", "Action": "None"},
        {"ID": "REF485720", "From": "Clara Hughes", "Date": "16-04-2024 18:22", "Topic": "Economic Recovery Plans.", "Priority": "9", "Dept": "DoF", "Suggested Respondent": "🔘  Nigel Huddleston", "Action": "None"},
        {"ID": "REF326781", "From": "Sophia Myles", "Date": "16-04-2024 17:57", "Topic": "Remote Learning Infrastructure.", "Priority": "8", "Dept": "DoE", "Suggested Respondent": "🔘  Gillian Keegan", "Action": "!"},
        {"ID": "REF546372", "From": "James O'Connor", "Date": "16-04-2024 17:29", "Topic": "Cybersecurity Measures.", "Priority": "7", "Dept": "DoT", "Suggested Respondent": "🔘  Michelle Donelan", "Action": "None"},
        {"ID": "REF765389", "From": "Fiona Gallagher", "Date": "15-04-2024 23:20", "Topic": "Plastic Waste Reduction.", "Priority": "6", "Dept": "Defra", "Suggested Respondent": "🔘  Steve Barclay", "Action": "None"},
        {"ID": "REF882014", "From": "George Karim", "Date": "16-04-2024 14:23", "Topic": "Arts Funding.", "Priority": "5", "Dept": "DCMS", "Suggested Respondent": "🔘  Lucy Frazer", "Action": "None"},
        {"ID": "REF213899", "From": "Omar Jensen", "Date": "16-04-2024 15:44", "Topic": "Public Transport Clean Energy Transition.", "Priority": "3", "Dept": "DfT", "Suggested Respondent": "🔘  Nick Harper", "Action": "None"},
    ]

def create_dataframe(results):
    # Convert results into a DataFrame
    df = pd.DataFrame(results)
    # Convert 'Date' to datetime format for better handling
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    df['Priority'] = pd.to_numeric(df['Priority'])
    return df

def summarise(path_to_pdf):
    """Extract text from a PDF and summarise it using OpenAI."""
    text = extract_text(path_to_pdf)
    response = client.completions.create(engine="text-davinci-003",
    prompt=f"Summarize this document: {text}",
    max_tokens=150)
    return response.choices[0].text.strip()

def get_position(text):
    """Determine the position from the text summary using OpenAI."""
    system_prompt = """
    You are an AI assistant that reads summaries of documents and determines the opinions 
    of the minister referred to in the documents about the topics discussed.
    """
    response = client.completions.create(engine="text-davinci-003",
    prompt=f"{system_prompt} Document summary: {text} What is the minister's position?",
    max_tokens=100)
    return response.choices[0].text.strip()

def doc_to_json(path):
    """Convert document text to a structured JSON format."""
    text = extract_text(path)
    system_prompt = """
    Convert this document text into JSON format with the following fields:
    Type, Title, Topic, Summary, Content, Relevant People, Director, Team, Sentiment,
    Deadline, Actions, Notes, Date.
    """
    response = client.completions.create(engine="text-davinci-003",
    prompt=f"{system_prompt} Document text: {text}",
    max_tokens=500)
    return json.loads(response.choices[0].text)

def fake_data():
    """Generate fake data for testing and placeholders."""
    data = {
    'Headline': ['AI regulation: pro-innovation, responsible growth.', 'Event 1', 'Event 2', 'Event 3'],
    'Sentiment': [1, 1, -1, 3],
    'Date': ['2020-01-01', '2021-03-01', '2022-04-01', '2022-05-01'],
    'Relevance': [0.7, 0.4, 0.8, 0.95],
    'Color': ['blue', 'blue', 'red', 'green'],
    'Topic': ['Submissions', 'Submissions', 'Internal sources', 'Public'],
    'Summary': [
        'The UK government has proposed a new AI regulatory framework that includes five values-focused cross-sectoral principles to guide regulator responses to AI risks and opportunities. This framework is designed to provide clarity and coherence to the AI regulatory landscape, build the evidence base, and ensure risks are identified and addressed while also supporting innovation. It includes a set of functions to support implementation of the framework, such as monitoring, assessment and feedback, and cross-sectoral risk assessment. The government has opened a consultation to receive feedback from stakeholders, and plans to establish a regulatory sandbox for AI and engage with the public to build trust in the technology.', 
        'Event 1 summary', 'Event 2 summary', 'Event 3 summary']
    }
    df = pd.DataFrame(data)

    data_2 = {
    'Department': ['HMPO', 'DBT', 'DfE', 'FCO', 'HMRC', 'Defra', 'DCLG', 'HMT', 'DfT', 'HO', 'BIS', 'DECC', 'MoD', 'DCMS', 'MoJ', 'CO', 'DfID', 'NIO', 'Scot', 'Wal'],
    'Volume': [20000, 15000, 12000, 10000, 9500, 9000, 8500, 8000, 7500, 7000, 6500, 6000, 5500, 5000, 4500, 4000, 3500, 3000, 2500, 2000]
    }
    dfa = pd.DataFrame(data_2)

    return df, dfa
