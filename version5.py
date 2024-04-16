import streamlit as st
import PyPDF2
import io
import openai
from openai import OpenAI
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

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
    # Create a chat completion to analyze the document more deeply
    department_info = """
    Home Office: immigration, security, law and order.
    Department for Business and Trade: business growth, industrial strategy, scientific research, innovation, international trade.
    Treasury (HM Treasury): economy, public finances, financial services.
    Department of Health and Social Care: health and social care services, NHS.
    Department for Education: child protection, education, apprenticeships, skills.
    Ministry of Defence: British Armed Forces, defense policy.
    Department for Work and Pensions (DWP): welfare, pensions, child maintenance policy.
    Ministry of Justice: courts, prisons, probation services, parole boards.
    Department for Transport: transport networks.
    Department for Environment, Food and Rural Affairs (DEFRA): environment, food production, agriculture, fisheries, rural communities.
    Department for Levelling Up, Housing and Communities: local government, housing, urban regeneration, building safety, community cohesion.
    Foreign, Commonwealth & Development Office: foreign policy, overseas aid, diplomacy, international development.
    """
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant. Here is information about UK ministerial departments: {department_info}"},
            {"role": "user", "content": f"Summarize the main question asked in the correspondence between the constituent and the MP, analyze the tone to determine urgency or significance of the sender, and identify which UK ministerial department the enquiry pertains to. Here is the text:\n\n{text}"}
        ]
    )
    return completion.choices[0].message.content

def draft_response(text):
    # Create a chat completion to draft a response to the document
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Draft a high-quality, clear, accurate, and helpful reply that answers all points raised in the correspondence, quotes any reference numbers and include the date the initial correspondence was sent. Address the response to the MP who sent the enquiry."},
            {"role": "user", "content": f"Draft a substantive reply for this correspondence:\n\n{text}"}
        ]
    )
    return completion.choices[0].message.content

def refine_response(original_text, style_preferences):
    # OpenAI API call to refine the response based on user stylistic preferences
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Revise the following text according to the user's stylistic preferences: {style_preferences}."},
            {"role": "user", "content": original_text}
        ]
    )
    return completion.choices[0].message.content


   

def main():
    st.title("PDF Content Analyzer")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_file = io.BytesIO(uploaded_file.getvalue())
        extracted_text = extract_text_from_pdf(pdf_file)
        st.session_state['extracted_text'] = extracted_text

        if st.button("Analyze Text"):
            summary = summarize_text(st.session_state['extracted_text'])
            st.subheader("Summary and Analysis")
            st.write(summary)

        if 'extracted_text' in st.session_state:
            if st.button("Draft Response"):
                response = draft_response(st.session_state['extracted_text'])
                st.session_state['drafted_response'] = response
                st.subheader("Drafted Response")
                st.write(response)

            # User input for stylistic preferences
            style_preferences = st.text_input("Input your stylistic preferences (e.g., formal, informal, use of specific terms):")
            
            if st.button("Update Response"):
                if 'drafted_response' in st.session_state and style_preferences:
                    updated_response = refine_response(st.session_state['extracted_text'], style_preferences)
                    st.subheader("Updated Response")
                    st.write(updated_response)

if __name__ == "__main__":
    main()