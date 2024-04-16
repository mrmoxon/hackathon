import streamlit as st
import PyPDF2
import io
import openai
from openai import OpenAI


client = OpenAI(

  api_key=""
)



def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    for page in pdf_reader.pages:
        if page.extract_text():
            text += page.extract_text() + '\n'
    return text

def summarize_text(text):
    
    # Create a chat completion to summarize the document
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarize this document and highlight any questions asked:\n\n{text}"}
        ]
    )
    
    return completion.choices[0].message.content

def assign_priority(text):
    high_priority_titles = ["Shadow Minister", "CEO", "President", "Chairman"]
    for title in high_priority_titles:
        if title.lower() in text.lower():
            return "High Priority"
    return "Normal Priority"

def main():
    st.title("PDF Content Analyzer")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        extracted_text = extract_text_from_pdf(pdf_file)
        st.session_state['extracted_text'] = extracted_text

    if 'extracted_text' in st.session_state and st.button("Analyze Text"):
        try:
            summary = summarize_text(st.session_state['extracted_text'])
            priority = assign_priority(st.session_state['extracted_text'])
            st.subheader("Summary")
            st.write(summary)
            st.subheader("Priority")
            st.write(priority)
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == "__main__":
    main()

