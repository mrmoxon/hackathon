import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import openai
# import pdf2image
from PIL import Image
from streamlit_option_menu import option_menu
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import base64

from util import *

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(layout="wide")

# Function to get embeddings
# def get_embeddings(text, model="text-embedding-ada-002"):
#     response = openai.Embedding.create(
#         input=text,
#         model=model
#     )
#     return response['data'][0]['embedding']

def get_embeddings(text, model="text-embedding-ada-002"):
    """
    Fetches embeddings for the provided text using the specified model.
    This function is adjusted to comply with openai API version >= 1.0.0.
    """
    try:
        # Ensure that only the first 2048 characters are sent if the model's limit is 2048 tokens
        response = client.embeddings.create(model=model, input=text[:2048])
        
        if response.data and len(response.data) > 0:
            embedding_vector = response.data[0].embedding  # Accessing the first embedding in the data array
            return embedding_vector
        else:
            print("No embedding data found in the response.")
            return None

    except Exception as e:
        print(f"Error while fetching embeddings: {e}")
        return None

# # Load stored embeddings
# def load_embeddings():
#     with open('embeddings_store.pkl', 'rb') as f:
#         return pickle.load(f)

def load_embeddings(embeddings_directory):
    embeddings_store = {}
    for file in os.listdir(embeddings_directory):
        if file.endswith('.pkl'):  # Assuming all embedding files have the '.emb' extension
            file_path = os.path.join(embeddings_directory, file)
            with open(file_path, 'rb') as f:
                embeddings_store[file] = pickle.load(f)
    return embeddings_store

embeddings_directory = '/Users/oscarmoxon/Desktop/AI Projects/hackathon/embeddings'
embeddings_store = load_embeddings(embeddings_directory)

# Function to find most similar documents
# def find_most_similar(input_embedding, embeddings_store, top_k=5):
#     similarities = {filename: 1 - cosine(input_embedding, embedding)
#                     if embedding is not None else -1
#                     for filename, embedding in embeddings_store.items()}
#     sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
#     return sorted_similarities[:top_k]

# def find_most_similar(input_embedding, embeddings_store, top_k=5):
#     similarities = {}
#     for filename, embedding in embeddings_store.items():
#         if embedding is not None:
#             try:
#                 similarity_score = 1 - cosine(input_embedding, embedding)
#                 similarities[filename] = similarity_score
#             except Exception as e:
#                 print(f"Error computing similarity for {filename}: {e}")
#     sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
#     return sorted_similarities[:top_k]

def find_most_similar(input_embedding, embeddings_store, top_k=5):
    similarities = {}
    for filename, embedding in embeddings_store.items():
        if embedding is not None:
            try:
                # Ensure both embeddings are flattened to 1-D
                input_emb_flat = input_embedding.flatten()
                embedding_flat = embedding.flatten()
                similarity_score = 1 - cosine(input_emb_flat, embedding_flat)
                similarities[filename] = similarity_score
            except Exception as e:
                print(f"Error computing similarity for {filename}: {e}")
    sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return sorted_similarities[:top_k]

#########

def main():
    st.sidebar.title('MiDAS')

    app_mode = st.sidebar.selectbox("Choose the page", ["Dashboard", "OpenQ", "Drafts"])

    selected = option_menu(
        menu_title="Navigation", 
        options=['Dashboard', 'OpenQ', 'Drafts'],
        icons=['house-fill', 'file-earmark-text', 'inbox-fill'],
        menu_icon="cast", 
        default_index=0,
        orientation="vertical",
    )

    if selected == 'Dashboard':
        show_dashboard()
    elif selected == 'OpenQ':
        show_openq()
    elif selected == 'Drafts':
        show_drafts()

def show_dashboard():
    st.title('Dashboard')
    df = fake_data()  # Assuming this fetches or generates the data to be displayed
    add_timeline(df)  # Function to plot data

def show_openq():
    st.title('MiDAS: Query Documents')

    # Set up columns for inputs and results
    col1, col2 = st.columns([2, 3])  # Adjust the width ratio as needed

    with col1:
        # Dropdown for ministers
        ministers = ["Ruth Davidson", "Greg Hands", "Angela Rayner", "Jess Phillips", "Keir Starmer"]
        selected_minister = st.selectbox("Search by Minister", ministers + ["See More..."])

        # Dropdown for Department
        departments = ["Aid and Development", "Environment", "Health", "Education", "Business and Trade"]
        selected_departments = st.selectbox("Search by Department", departments + ["See More..."])

        # Text area for input
        user_input = st.text_area("Open a query by typing below...", height=150)

        # Search button
        search_clicked = st.button('Search')

    with col2:

        if search_clicked and user_input:
            # Convert input text to embeddings
            input_embedding = get_embeddings(user_input)
            
            # Find the most similar documents
            similar_docs = find_most_similar(input_embedding, embeddings_store)
            
            # Display the results
            st.subheader('Top similar documents:')
            for filename, similarity in similar_docs:
                st.write(f"**{filename}** - Similarity: {similarity:.4f}")

                filename_pdf = os.path.splitext(filename)[0]

                # Assume the file path needs to be adjusted if not in the same directory
                file_path = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/MiDAS/processed_files', filename)
                file_path_pdf = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/MiDAS/data_files', filename_pdf)
                
                # Displaying excerpts
                try:
                    with open(file_path, 'r') as file:
                        text = file.read(1000)  # Read first 1000 characters
                        st.text_area("Excerpt", text, height=100)
                except Exception as e:
                    st.error(f"Failed to read file: {str(e)}")

                # Displaying specific PDF page (e.g., page 10)
                if file_path_pdf.endswith('.pdf'):
                    try:
                        # images = pdf2image.convert_from_path(file_path_pdf, first_page=1, last_page=1)
                        # st.image(images[0], caption='First Page Preview')
                        print("displaying image")
                    except Exception as e:
                        st.error(f"Error displaying PDF page: {str(e)}")
       
        else:
            # Show most recent documents
            st.subheader('Most Recent Documents:')
            for filename in list(embeddings_store.keys())[:5]:
                st.write(f"**{filename}**")

def show_drafts():
    st.title('Drafts')
    st.write("Inbox:")

    # Display drafts or placeholders here
    # Potentially load and show drafts from a data source

def add_timeline(df):
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row['Date']], 
            y=[row['Sentiment']],
            mode='markers',
            marker=dict(size=row['Relevance']*50, color=row['Color']),
            text=f"<b>Date:</b> {row['Date']}<br><b>Headline:</b> {row['Headline']}<br><b>Summary:</b> {row['Summary']}",
            hoverinfo='text',
            name=row['Topic']
        ))
    st.plotly_chart(fig)

def display_submission(submission, related_data):
    st.image('./data/profile.jpeg', width=100)  # Adjust path and size as necessary

    st.title('Submission Summary')
    st.json(submission['summary'])  # Displaying JSON for clarity or use a markdown for HTML styling

    st.title('Current Position')
    st.json(submission['position'])

    st.title('Related Data Timeline')
    add_timeline(related_data)

if __name__ == "__main__":
    main()


# def add_linebreaks(string, every=64):
#     lines = []
#     for i in range(0, len(string), every):
#         lines.append(string[i:i+every])
#     return '<br>'.join(lines)

# def load_submission(path):
#     submission = {'summary': None, 'position': None}
#     submission['summary'] = summarise(path)
#     submission['position'] = get_position(path)
#     return submission

# def find_min_max_date(df):
#     # Convert the dates to datetime objects
#     df['Date'] = pd.to_datetime(df['Date'])
#     return df['Date'].min(), df['Date'].max()

# def add_timeline(df):
#     fig = go.Figure()
#     for i, row in df.iterrows():
#         fig.add_trace(go.Scatter(
#             x=[row['Date']], 
#             y=[row['Sentiment']],
#             mode='markers',
#             marker=dict(size=row['Relevance']*50, color=row['Color']),
#             text=f"<b>Date:</b> {row['Date']}<br><b>Headline:</b> {row['Headline']}<br><b>Summary:</b> {row['Summary']}",
#             hoverinfo='text',
#             name=row['Topic']
#         ))
#     st.plotly_chart(fig)

# def add_timeline(df):
#     min_date, max_date = find_min_max_date(df)

#     # add timeline
#     fig.add_shape(
#         type="line",
#         x0=min_date,
#         y0=0,
#         x1=max_date,
#         y1=0,
#         line=dict(
#             color="black",
#             width=2,
#         ),
#     )

#     # {
#     #     "Type": <type of document, this could be an email, meeting minutes, submission>,
#     #     "Title": <title of the document>,
#     #     "Topic": <overall topic discussed in the document>,
#     #     "Summary": <summary of the document content>,
#     #     "Content": null,
#     #     "Relevant People": <list of people mentioned in the document, inlcuding sender and recipients, format as: Forname Surname>,
#     #     "Director": <director mentioned in the document>,
#     #     "Team": <teams mentioned in the document that need to give clearance, where a name is provided>,
#     #     "Sentiment": <sentiment analysis of the document, should be a floating point number between -1.00 and 1.00, with -1.00 being negative and 1.00 being positive and 0.00 being neutral, must be filled>,
#     #     "Deadline": <deadline of any actions mentioned in the document, format as exactly as shown in the document>,
#     #     "Actions": <any follow-up actions mentioned in the document, with added deadline as per "Deadline" field>,
#     #     "Notes": <any notes attached to the document>,
#     #     "Date": <date of the document in the format: YYYY-MM-DD, if no date is found then null>
#     # }

#     for i, row in df.iterrows():
#         fig.add_shape(
#             type="line",
#             x0=row['Date'],
#             y0=0,
#             x1=row['Date'],
#             y1=row['Sentiment'],
#             yref='y',
#             line=dict(
#                 color="black",
#                 width=1,
#             ),
#         )

#         summary_with_breaks = add_linebreaks(str(row['Summary']))
        
#         fig.add_trace(go.Scatter(
#         x=[row['Date']], 
#         y=[row['Sentiment']],
#         mode='markers',
#         marker=dict(size=row['Relevance']*50, color=row['Color']),
#         # marker=dict(size=10, color=row['Color']),
#         text=['<b>Date:</b> '+str(row['Date']) + '<br><b>Headline:</b> '+str(row['Headline']) + '<br><b>Summary:</b> '+ summary_with_breaks],
#         hoverinfo='text',
#         name= str(row['Topic'])
#     ))

#     # Display the figure with Streamlit
#     st.plotly_chart(fig)


# def display_submission(submission, related_data):
#     st.image('./data/profile.jpeg')

#     ###### Submission summary
#     st.title('Submission summary')

#     summary = submission['summary']
#     bordered_text = f'<div style="border:2px solid black; padding:10px">{summary}</div>'

#     st.markdown(bordered_text, unsafe_allow_html=True)

#     ###### Current position
#     st.title('Current position')
#     position = submission['position']
#     bordered_text = f'<div style="border:2px solid black; padding:10px">{position}</div>'

#     st.markdown(bordered_text, unsafe_allow_html=True)

#     st.title('Timeline')
#     add_timeline(related_data)


# def upload():
#     uploaded_file = st.file_uploader("Upload your submission: ", type="pdf")
#     show_file = st.empty()
#     if uploaded_file is not None:
#         show_file.info("File received!")
#         submission = load_submission(uploaded_file)
#         related_data = get_related_data(uploaded_file)
#         display_submission(submission, related_data)

# def toolbar():
    
#     with st.sidebar:
#         st.title('MiDAS')
#         selected = option_menu(
#             menu_title= '',
#             options=['Dashboard', 'OpenQ', 'Drafts'],
#             icons=['house', 'tree', '100'],
#             menu_icon='arrow-through-heart',
#             default_index= 0,
#             orientation='vertical',
#         )
#     if selected == 'Dashboard':
#         st.title(f'Dashboard')


# Create a Plotly figure for the timeline
fig = go.Figure()

# st.title('REDUX')
# toolbar()
# upload()