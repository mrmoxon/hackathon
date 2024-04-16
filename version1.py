import streamlit as st
import PyPDF2
import io

def extract_text_from_pdf(pdf_file):
    # Initialize a PDF file reader
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ''
    
    # Iterate over each page and extract text
    for page in pdf_reader.pages:
        text += page.extract_text() + '\n'  # Extract text and add a newline for separation
    
    return text

def main():
    st.title("PDF Content Extractor")
    # File uploader widget
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        # Convert uploaded file to a readable file-like object
        pdf_file = io.BytesIO(uploaded_file.read())
        
        # Extract text from the PDF
        extracted_text = extract_text_from_pdf(pdf_file)
        
        # Display the extracted text
        st.text_area("Extracted Text", extracted_text, height=300)

if __name__ == "__main__":
    main()