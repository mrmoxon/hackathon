import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import openai
from PIL import Image
from streamlit_option_menu import option_menu
from scipy.spatial.distance import cosine
from dotenv import load_dotenv
import anthropic
from anthropic import Anthropic
import base64
import matplotlib.pyplot as plt

# For embedding and similarity functions
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from util import *

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
ant_api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure OpenAI client
openai_client = openai.Client(api_key=openai_api_key)

# Configure Anthropic client
anthropic_client = anthropic.Client(api_key=ant_api_key)

st.set_page_config(layout="wide")

embeddings_directory = '/Users/oscarmoxon/Desktop/AI Projects/hackathon/embeddings'

def load_embeddings(file_path):
    with open(file_path, 'rb') as f:
        embeddings = pickle.load(f)
    return embeddings

def find_most_similar(new_embedding, stored_embeddings, top_k=5):
    filenames = list(stored_embeddings.keys())
    embeddings = np.array(list(stored_embeddings.values()))
    similarities = cosine_similarity([new_embedding], embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(filenames[i], similarities[i]) for i in top_indices]

def get_embeddings(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(input = [text], model=model).data[0].embedding

embeddings_file = "/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/embeddings.pkl"
stored_embeddings = load_embeddings(embeddings_file)

# new_text = "Input your new text here to find similar images. I'd love to know about asylum employment."
# # Get embedding for the new text
# new_embedding = get_embeddings(new_text)

# # Find the top 5 most similar images
# similar_images = find_most_similar(new_embedding, stored_embeddings)

# # Print the results
# print("Top 5 similar images:")
# for filename, similarity in similar_images:
#     print(f"{filename}: {similarity:.4f}")

# Generate embeddings for the input correspondence
# def get_embeddings(text, model="text-embedding-ada-002"):
#     """
#     Fetches embeddings for the provided text using the specified model.
#     This function is adjusted to comply with openai API version >= 1.0.0.
#     """
#     try:
#         # Ensure that only the first 2048 characters are sent if the model's limit is 2048 tokens
#         response = openai_client.embeddings.create(model=model, input=text[:2048])
        
#         if response.data and len(response.data) > 0:
#             embedding_vector = response.data[0].embedding  # Accessing the first embedding in the data array
#             return embedding_vector
#         else:
#             print("No embedding data found in the response.")
#             return None

#     except Exception as e:
#         print(f"Error while fetching embeddings: {e}")
#         return None

# Load embeddings from the embeddings directory 
# def load_embeddings(embeddings_directory):
#     embeddings_store = {}
#     for file in os.listdir(embeddings_directory):
#         if file.endswith('.pkl'):  # Assuming all embedding files have the '.emb' extension
#             file_path = os.path.join(embeddings_directory, file)
#             with open(file_path, 'rb') as f:
#                 embeddings_store[file] = pickle.load(f)
#     return embeddings_store

# embeddings_store = load_embeddings(embeddings_directory)

# def find_most_similar(input_embedding, embeddings_store, top_k=5):
#     similarities = {}
#     for filename, embedding in embeddings_store.items():
#         if embedding is not None:
#             try:
#                 # Ensure both embeddings are flattened to 1-D
#                 input_emb_flat = input_embedding.flatten()
#                 embedding_flat = embedding.flatten()
#                 similarity_score = 1 - cosine(input_emb_flat, embedding_flat)
#                 similarities[filename] = similarity_score
#             except Exception as e:
#                 print(f"Error computing similarity for {filename}: {e}")
#     sorted_similarities = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
#     return sorted_similarities[:top_k]

fields = [  
  {'title': "Department", 'tag': 'department', 'description': 'This is always capitalised. The government department which has written the response to the query.'},
  {'title': "Subject", 'tag': 'subject', 'description': 'The specifics of the nature of the contact e.g. Arms Trade: Israel, otherwise infer this yourself from the context. Usually a colon separates the subject from a short description. Include this.'},
  {'title': "Questioner", 'tag': 'questioner', 'description': 'The name of the person asking the question. There may be multiple names of questioners , they are identified by not being indented.'},
  {'title': "ID", 'tag': 'id', 'description': 'The ID of the response in square brackets. Also return the URL associated with the hyperlink.'},
  {'title': "Question", 'tag': 'question', 'description': 'The question being asked. There may be multiple questions being asked, they are identified by not being indented.'},
  {'title': "Respondent", 'tag': 'respondent', 'description': "Identify this by the indent. The name of the Member of Parliament who has responded to written the question"},
  {'title': "Answer", 'tag': 'answer', 'description': "Identify this by the indent. The answer provided by Parliament to the question."}
]

def prompt():
    fields_string = "\n".join(f"- {field['title']}: {field['description']}. Wrap this field in <{field['tag']}> tags." for field in fields)
    return f"""
Transcribe the text in this image in full, wrapped in <transcription> tags.
Please also extract the following fields:
{fields_string}
    """.strip()

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

        # Image upload
        uploaded_image = st.file_uploader("Upload an image for transcription", type=['png', 'jpg'])
        if uploaded_image is not None:
            # Display the uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Process the image if a button is clicked
            # if st.button('Process Image'):
            if image:
                image_data = base64.b64encode(uploaded_image.getvalue()).decode("utf-8")
                image_media_type = 'image/png' if uploaded_image.type == 'image/png' else 'image/jpeg'
                
                # Create the message payload for the API
                response = anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": image_media_type,
                                        "data": image_data,
                                    },
                                },
                                {"type": "text", "text": prompt()}
                            ],
                        }
                    ],
                )
                parsed_text = response.content[0].text  # Adjust based on actual API response
                st.text_area("Transcribed Text", parsed_text, height=300)
                user_input = parsed_text

        # Send Correspondence
        if st.button("Send Correspondence"):
            st.write("Correspondence sent!")        
            
        if user_input:
            new_embedding = get_embeddings(user_input)
            similar_images = find_most_similar(new_embedding, stored_embeddings)
            st.session_state['similar_images'] = similar_images

    with col2:

        if 'similar_images' in st.session_state:
            st.subheader('Top similar documents:')
            for filename, similarity in st.session_state['similar_images']:
                st.write(f"**{filename}** - Similarity: {similarity:.4f}")

                print("displaying title:", filename)
                if filename.endswith('.png.txt'):
                    print("removing .txt from", filename)
                    filename_png = os.path.splitext(filename)[0]
                elif filename.endswith('.png'):
                    print("filename is already a png", filename)
                    filename_png = filename
                else:
                    print("filename is txt", filename)
                    filename_png = os.path.splitext(filename)[0]
                    filename_png = filename_png + ".png"
                print("filename_png:", filename_png)
                file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                
                # Displaying specific png page (e.g., page 10)
                # if filename_png.endswith('.png'):
                try:
                    print("displaying image")
                    image = Image.open(file_path_png)
                    st.image(image, caption=filename, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {filename}: {str(e)}")
       
        else:
            # Show most recent documents
            st.subheader('Recently Published:')
            new_embedding = get_embeddings("Home Office")
            recent_developments = find_most_similar(new_embedding, stored_embeddings)
            st.session_state['recent_developments'] = recent_developments

            for filename, similarity in recent_developments:
                if filename.endswith('.png.txt'):
                    print("removing .txt from", filename)
                    filename_png = os.path.splitext(filename)[0]
                elif filename.endswith('.png'):
                    print("filename is already a png", filename)
                    filename_png = filename
                else:
                    print("filename is txt", filename)
                    filename_png = os.path.splitext(filename)[0]
                    filename_png = filename_png + ".png"
                print("filename_png:", filename_png)
                file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                
                # Displaying specific png page (e.g., page 10)
                # if filename_png.endswith('.png'):
                try:
                    print("displaying image")
                    image = Image.open(file_path_png)
                    st.image(image, caption=filename, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {filename}: {str(e)}")


            # for filename in list(embeddings_store.keys())[:5]:
            #     st.write(f"**{filename}**")


                

        # if search_clicked and user_input:
        #     # Convert input text to embeddings
        #     input_embedding = get_embeddings(user_input)
            
        #     # Find the most similar documents
        #     similar_docs = find_most_similar(input_embedding, embeddings_store)
            
        #     # Display the results
        #     st.subheader('Top similar documents:')
        #     for filename, similarity in similar_docs:
        #         st.write(f"**{filename}** - Similarity: {similarity:.4f}")

        #         filename_pdf = os.path.splitext(filename)[0]

        #         # Assume the file path needs to be adjusted if not in the same directory
        #         file_path = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/MiDAS/processed_files', filename)
        #         file_path_pdf = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/MiDAS/data_files', filename_pdf)
                
        #         # Displaying excerpts
        #         try:
        #             with open(file_path, 'r') as file:
        #                 text = file.read(1000)  # Read first 1000 characters
        #                 st.text_area("Excerpt", text, height=100)
        #         except Exception as e:
        #             st.error(f"Failed to read file: {str(e)}")

        #         # Displaying specific PDF page (e.g., page 10)
        #         if file_path_pdf.endswith('.pdf'):
        #             try:
        #                 # images = pdf2image.convert_from_path(file_path_pdf, first_page=1, last_page=1)
        #                 # st.image(images[0], caption='First Page Preview')
        #                 print("displaying image")
        #             except Exception as e:
        #                 st.error(f"Error displaying PDF page: {str(e)}")
       
        # else:
        #     # Show most recent documents
        #     st.subheader('Most Recent Documents:')
        #     for filename in list(embeddings_store.keys())[:5]:
        #         st.write(f"**{filename}**")

#########

def generate_fade_red_color(value, max_value):
    """Generate a faded red color based on the value and the max value."""
    # Normalize value
    normalized = value / max_value
    # Convert to color
    color = f"rgba(255, {int(255 * (1 - normalized))}, {int(255 * (1 - normalized))}, {normalized})"
    return color

def add_bar_chart(df):
    # Find the max value for the normalization of colors
    max_value = df['Volume'].max()
    
    # Create a figure with Plotly
    fig = go.Figure()

    # Add bar for each department
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['Department']],
            y=[row['Volume']],
            marker=dict(color=generate_fade_red_color(row['Volume'], max_value)),
            name=row['Department']
        ))
    
    # Update the layout to match the style of the uploaded chart
    fig.update_layout(
        title='Volume of correspondence with MPs and Peers by government department, 2014',
        xaxis_title="Department",
        yaxis_title="Volume",
        legend_title="Departments",
        font=dict(size=10),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Add the source text
    # fig.add_annotation(
    #     # text='Source: Institute for Government analysis of annual ministerial correspondence reports, 2015',
    #     xref='paper', yref='paper',
    #     x=0, y=-0.1,
    #     showarrow=False,
    #     font=dict(size=10, color="blue")
    # )
    
    # Show the figure in the Streamlit app
    st.plotly_chart(fig)

def get_priority_color(priority):
    # Scale the priority to a 0-255 range for red color intensity
    if priority > 10:
        priority = 10  # Cap priority to 10 for calculation purposes
    green_blue_component = 255 - int((priority / 15) * 255)
    return f"#{255:02x}{green_blue_component:02x}{green_blue_component:02x}"

def create_dataframe(results):
    # Convert results into a DataFrame
    df = pd.DataFrame(results)
    # Convert 'Date' to datetime format for better handling
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y %H:%M')
    df['Priority'] = pd.to_numeric(df['Priority'])
    # Map priority to a 'Sentiment' for visualization purposes
    df['Sentiment'] = df['Priority']
    df['Relevance'] = df['Priority'] / 10  # Normalize for plotting
    return df

def show_dashboard():
    st.title('Triaging Dashboard')
    df, dfa = fake_data()  # Assuming this fetches or generates the data to be displayed
    col1, col2 = st.columns([2, 2])  # Adjust the width ratio as needed

    results = [
        {"ID": "REF992034", "From": "Edward H. Scissor", "Date": "16-04-2024 18:33", "Topic": "Banning Hands Export.", "Priority": "10", "Dept": "DBT", "Suggested Respondent": "🔘  Greg Hands", "Action": "None"},
        {"ID": "REF110293", "From": "Linda Sun", "Date": "16-04-2024 13:30", "Topic": "Will YC Move to UK?", "Priority": "9", "Dept": "DoT", "Suggested Respondent": "🔘  Michelle Donelan", "Action": "None"},
        {"ID": "REF485720", "From": "Clara Hughes", "Date": "16-04-2024 18:22", "Topic": "Economic Recovery Plans.", "Priority": "9", "Dept": "DoF", "Suggested Respondent": "🔘  Nigel Huddleston", "Action": "None"},
        {"ID": "REF326781", "From": "Sophia Myles", "Date": "16-04-2024 17:57", "Topic": "Remote Learning Infrastructure.", "Priority": "8", "Dept": "DoE", "Suggested Respondent": "🔘  Gillian Keegan", "Action": "!"},
        {"ID": "REF546372", "From": "James O'Connor", "Date": "16-04-2024 17:29", "Topic": "Cybersecurity Measures.", "Priority": "7", "Dept": "DoT", "Suggested Respondent": "🔘  Michelle Donelan", "Action": "None"},
        {"ID": "REF765389", "From": "Fiona Gallagher", "Date": "16-04-2024 12:20", "Topic": "Plastic Waste Reduction.", "Priority": "6", "Dept": "Defra", "Suggested Respondent": "🔘  Steve Barclay", "Action": "None"},
        {"ID": "REF882014", "From": "George Karim", "Date": "16-04-2024 14:23", "Topic": "Arts Funding.", "Priority": "5", "Dept": "DCMS", "Suggested Respondent": "🔘  Lucy Frazer", "Action": "None"},
        {"ID": "REF213899", "From": "Omar Jensen", "Date": "16-04-2024 15:44", "Topic": "Public Transport Clean Energy Transition.", "Priority": "3", "Dept": "DfT", "Suggested Respondent": "🔘  Nick Harper", "Action": "None"},
    ]

    with col1:
        df = create_dataframe(results)
        add_timeline(df)  # Function to plot data
        # df = create_dataframe(results)
        # plot_data(df)

    with col2:
        add_bar_chart(dfa)
    
    if 'hide_rows' not in st.session_state:
        st.session_state.hide_rows = []

    # Define column titles
    col_layout = ([1, 3, 2, 4, 1, 1, 1, 2, 1, 1])
    column_titles = ["ID", "From", "Date", "Topic", "", "Priority", "Dept", "Suggested Respondent", "Action"]
    cols = st.columns(len(column_titles))
    # Column layout
    for col, title in zip(cols, column_titles):
        col.write(f"**{title}**")
    st.markdown("<hr style='height:2px;border:none;color:#333;background-color:#333;' />", unsafe_allow_html=True)

    # Header row
    # for col, title in zip(cols, column_titles):
    #     col.write(f"**{title}**")

    for result in results:
        if result['ID'] not in st.session_state.hide_rows:
            priority_color = get_priority_color(int(result['Priority']))
            cols = st.columns([1, 2, 2, 3, 2, 1, 2, 1, 1])  # Adding an extra column for the 'Send' button

            # Data rows
            with st.container():
                for idx, (title, value) in enumerate(result.items()):
                    if title == 'Priority':
                        cols[idx].markdown(f"<span style='display: block; background-color: {priority_color}; padding: 8px; border-radius: 5px;'>{value}</span>", unsafe_allow_html=True)
                    elif title == "Action" and value == "!":
                        cols[idx].markdown(f"<span style='color: red; font-weight: bold;'>⚠️ Flagged</span>", unsafe_allow_html=True)
                    else:
                        cols[idx].write(value)

                # Send button
                send_col = cols[-1]
                if send_col.button("Send", key=f"send_{result['ID']}"):
                    st.session_state.hide_rows.append(result['ID'])

                with st.expander("Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Suggested Respondent: {result['Suggested Respondent']}")
                        st.write(f"Action Required: {result['Action']}")
                        st.write(f"Department: {result['Dept']}")
                        st.write(f"Summary: {result['Topic']}")
                        st.write(f"Date: {result['Date']}")
                        st.write(f"From: {result['From']}")
                        st.write(f"ID: {result['ID']}")
                        if result['ID'] == 'REF110293':
                            st.write("YC needs to move to the UK, non-controversial opinion.")
                        else:
                            st.write("Summary: \nSerious concerns raised about the export of Greg's Hands, and a suggestion to ban their export entirely as they represent a valuable UK asset and should be protected from foreign interference. From MP Mr Edward Scissorhands, representative of constutients in Somerset.")

                    with col2:
                        if result['ID'] == 'REF110293':
                            image_path = "/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/YC.png"
                            image = st.image(image_path, caption="YC", use_column_width=True)
                        else:
                            image_path = "/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/Correspondence.jpeg"
                            image = st.image(image_path, caption="Business and Trade", use_column_width=True)


















def draft_response(text, style_preferences):
    print("Text to respond to:", text)
    print("Style preferences:", style_preferences)
    print("Feeding into model...")
    response = client.chat.completions.create(
        model="gpt-4-turbo",  # Assuming you're using gpt-4-turbo based on your previous example
        messages=[
            {"role": "system", "content": f"You are a proxy for Greg Hands, MP for Chelsea and Fulham and Secretary of State for Business and Trade. Draft a high-quality, clear, accurate, and helpful reply that answers all points raised in the correspondence, quotes any reference numbers and include the date the initial correspondence was sent. Address the response to the MP who sent the enquiry. Furthermore, apply the user's stylistic preferences to tailor the response... {style_preferences}."},
            {"role": "user", "content": text}
        ]
    )
    print("\n")
    # print("Response generated:", response)
    # print("Responses generate.choices:", response.choices)
    # print("\n")
    # print("Response.choices[0]:", response.choices[0])
    # print("\n")
    # print("Response.choices[0].message:", response.choices[0].message)
    # print("\n")
    # print("Response.choices[0].message.content:", response.choices[0].message.content)
    # print("\n")
    # print("Response.choices[0].message.content.strip():", response.choices[0].message.content.strip())
    # print("\n")

    if response is not None and response.choices:
        draft_response_text = response.choices[0].message.content.strip()

        print("Draft response:", draft_response_text)
        return draft_response_text
    else:
        print("Failed to get a valid response.")
        return "No response generated."
    # return response.choices[0].message['content'].strip()

def show_drafts():
    st.title('Drafts')
    st.write("Inbox:")

    # Display drafts or placeholders here
    # Potentially load and show drafts from a data source

    results = [
        {"ID": "REF992034", "From": "Edward H. Scissor", "Date": "16-04-2024 18:33", "Topic": "Banning Hands Export.", "Priority": "10", "Dept": "DBT", "To": "Greg Hands", "Action": "None"},
        {"ID": "REF110283", "From": "Mary Manafred", "Date": "16-04-2024 13:30", "Topic": "Arms Trade: Israel", "Priority": "9", "Dept": "DBT", "To": "Greg Hands", "Action": "None"},
        {"ID": "REF485120", "From": "Debbie Hughes", "Date": "16-04-2024 18:22", "Topic": "Exports: Hamas.", "Priority": "9", "Dept": "DBT", "To": "Greg Hands", "Action": "None"},
        {"ID": "REF326782", "From": "Layla Margoyles", "Date": "16-04-2024 17:57", "Topic": "Food: Japan.", "Priority": "8", "Dept": "DBT", "To": "Greg Hands", "Action": "None"},
    ]

    drafted = """
    Dear Mr. Scissorhands,

    Thank you for your correspondence regarding the export of Greg's Hands.  I understand your concerns about preserving valuable U.K. assets, and I appreciate you bringing this matter to my attention.  

    I would like to clarify that the matter in question seems to involve a misunderstanding.  "Greg's Hands" appears to reference a metaphorical or symbolic expression rather than a physical commodity or asset that can be subjected to export control or trading restrictions.  As such, there is no direct action that can be taken to ban such exports since it does not represent a tangible or legal entity.  

    If, however, your enquiry pertains to a specific product or cultural property under my ministerial purview that you believe requires protection, I encourage you to provide further details.  My office is committed to ensuring that our national interests are safeguarded and that any export controls in place promote our national economic and cultural security effectively.

    Should you have further concerns or need assistance with different matters, please do not hesitate to contact my office.

    Thank you once again for your engagement on this issue.

    Sincerely,

    Greg Hands, MP
    Secretary of State for Business and Trade
    2024-04-17
    """

    if 'hide_rows' not in st.session_state:
        st.session_state.hide_rows = []
        for result in results:
            st.session_state[f'draft_text_{result["ID"]}'] = drafted  # Initialize draft text for each entry

    # Define column titles
    col_layout = ([1, 3, 2, 4, 1, 1, 1, 2, 1, 1])
    column_titles = ["ID", "From", "Date", "Topic", "", "Priority", "Dept", "To", "Action"]
    cols = st.columns(len(column_titles))
    # Column layout
    for col, title in zip(cols, column_titles):
        col.write(f"**{title}**")
    st.markdown("<hr style='height:2px;border:none;color:#333;background-color:#333;' />", unsafe_allow_html=True)

    for result in results:
        if result['ID'] not in st.session_state.hide_rows:
            priority_color = get_priority_color_ii(int(result['Priority']))
            cols = st.columns([1, 2, 2, 3, 2, 1, 2, 1, 1])  # Adding an extra column for the 'Send' button

            # Data rows
            with st.container():
                for idx, (title, value) in enumerate(result.items()):
                    if title == 'Priority':
                        cols[idx].markdown(f"<span style='display: block; background-color: {priority_color}; padding: 8px; border-radius: 5px;'>{value}</span>", unsafe_allow_html=True)
                    elif title == "Action" and value == "!":
                        cols[idx].markdown(f"<span style='color: red; font-weight: bold;'>⚠️ Flagged</span>", unsafe_allow_html=True)
                    else:
                        cols[idx].write(value)

                # Send button
                # send_col = cols[-1]
                # if send_col.button("Send", key=f"send_{result['ID']}"):
                #     st.session_state.hide_rows.append(result['ID'])

                with st.expander("Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Summary: {result['Topic']}")
                        st.write(f"Date: {result['Date']}")
                        st.write(f"From: {result['From']}")
                        st.write("Message: \n\nTo ask the Secretary of State for Business and Trade to ban the export of Gregs Hands. They are a valuable U.K asset and should be protected from foreign interference.")
                        
                        draft_key = f'draft_text_{result["ID"]}'
                        user_input = st.text_area("Draft Message", value=drafted, height=300, key=draft_key)

                        # draft_key = f'draft_text_{result["ID"]}'
                        # print("Session state key 1:", draft_key)

                        # st.session_state[draft_key] = 'Drafted response will appear here.'
                        # print("Session state key 2:", draft_key)

                        # user_input = st.text_area("Draft Message", value=st.session_state[draft_key], height=300, key=f'drafted_text_{result["ID"]}')
                        # print("Session state key 3:", draft_key)

                        if user_input:
                            new_embedding = get_embeddings(user_input)
                            similar_images = find_most_similar(new_embedding, stored_embeddings)
                            st.session_state['similar_images'] = similar_images

                        # send = st.button("Send", key=f"send_{result['ID']}")
                        # if st.button("Send", key=f"send_{result['ID']}"):
                        #     st.session_state.hide_rows.append(result['ID'])
                            # Also compress the expander

                        send_btn_key = f"send_{result['ID']}"
                        if st.button("Send", key=send_btn_key):
                            st.session_state.hide_rows.append(result['ID'])
                            if result['ID'] in st.session_state.get('expanded_rows', []):
                                st.session_state['expanded_rows'].remove(result['ID'])

                    with col2:
                        extra_instructions_key = f'extra_instructions_{result["ID"]}'
                        print("Session state key 4:", draft_key)
                        # extra_instructions = st.text_area("Add Extra Instructions Here:", value="Double space after all full stops. \n\nNever include the words estoppel, catastrophe, or any unnecessary epithets. \n\nAlways be polite and use formal language.", height=150, key=f'extra_instructions_{result["ID"]}')
                        extra_instructions = st.text_area("Add Extra Instructions Here:", value="Double space after all full stops. \n\nNever include the words estoppel, catastrophe, or any unnecessary epithets. \n\nAlways be polite and use formal language.", height=150, key=extra_instructions_key)
                        
                        rewrite_btn_key = f'rewrite_{result["ID"]}'
                        if st.button('Rewrite', key=rewrite_btn_key):
                            message = "From: Mr Edward Scissorhands. To ask the Secretary of State for Business and Trade to ban the export of Greg's Hands. They are a valuable U.K asset and should be protected from foreign interference."
                            new_draft = draft_response(message, extra_instructions)
                            print("New draft response GENERATED:", new_draft)
                            st.session_state[draft_key] = new_draft  # Update the draft in the session state
                            print("Experimental rerun...")
                            st.experimental_rerun()

                        # if st.button('Rewrite', key=f"rewrite_{result['ID']}"):
                        #     message = "From: Mr Edward Scissorhands. To ask the Secretary of State for Business and Trade to ban the export of Greg's Hands. They are a valuable U.K asset and should be protected from foreign interference."
                        #     prompt = message

                        #     new_draft = draft_response(prompt, extra_instructions)
                        #     print("Updating draft response...")
                        #     print("Session state key 5:", draft_key)
                        #     st.session_state[draft_key] = new_draft
                        #     print("Session state key 6:", draft_key)



                        if 'similar_images' in st.session_state:
                            st.subheader('Similar Historic Statements:')
                            for filename, similarity in st.session_state['similar_images']:
                                st.write(f"**{filename}** - Similarity: {similarity:.4f}")

                                print("\n\n New search \n\n")

                                print("displaying title:", filename)
                                if filename.endswith('.png.txt'):
                                    print("removing .txt from", filename)
                                    filename_png = os.path.splitext(filename)[0]
                                elif filename.endswith('.png'):
                                    print("filename is already a png", filename)
                                    filename_png = filename
                                else:
                                    print("filename is txt", filename)
                                    filename_png = os.path.splitext(filename)[0]
                                    filename_png = filename_png + ".png"
                                print("filename_png:", filename_png)
                                file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                                
                                # Displaying specific png page (e.g., page 10)
                                # if filename_png.endswith('.png'):
                                try:
                                    print("displaying image")
                                    image = Image.open(file_path_png)
                                    st.image(image, caption=filename, use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image {filename}: {str(e)}")
                    
                        else:
                            # Show most recent documents
                            st.subheader('Recently Published:')
                            new_embedding = get_embeddings("Business and Trade")
                            recent_developments = find_most_similar(new_embedding, stored_embeddings)
                            st.session_state['recent_developments'] = recent_developments

                            for filename, similarity in recent_developments:
                                if filename.endswith('.png.txt'):
                                    print("removing .txt from", filename)
                                    filename_png = os.path.splitext(filename)[0]
                                elif filename.endswith('.png'):
                                    print("filename is already a png", filename)
                                    filename_png = filename
                                else:
                                    print("filename is txt", filename)
                                    filename_png = os.path.splitext(filename)[0]
                                    filename_png = filename_png + ".png"
                                print("filename_png:", filename_png)
                                file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                                
                                # Displaying specific png page (e.g., page 10)
                                # if filename_png.endswith('.png'):
                                try:
                                    print("displaying image")
                                    image = Image.open(file_path_png)
                                    st.image(image, caption=filename, use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image {filename}: {str(e)}")


    #     # Send Correspondence
    #     if st.button("Send Correspondence"):
    #         st.write("Correspondence sent!")        
            
    #     if user_input:
    #         new_embedding = get_embeddings(user_input)
    #         similar_images = find_most_similar(new_embedding, stored_embeddings)
    #         st.session_state['similar_images'] = similar_images

    # with col2:

        # if 'similar_images' in st.session_state:
        #     st.subheader('Top similar documents:')
        #     for filename, similarity in st.session_state['similar_images']:
        #         st.write(f"**{filename}** - Similarity: {similarity:.4f}")

        #         print("displaying title:", filename)
        #         if filename.endswith('.png.txt'):
        #             print("removing .txt from", filename)
        #             filename_png = os.path.splitext(filename)[0]
        #         elif filename.endswith('.png'):
        #             print("filename is already a png", filename)
        #             filename_png = filename
        #         else:
        #             print("filename is txt", filename)
        #             filename_png = os.path.splitext(filename)[0]
        #             filename_png = filename_png + ".png"
        #         print("filename_png:", filename_png)
        #         file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                
        #         # Displaying specific png page (e.g., page 10)
        #         # if filename_png.endswith('.png'):
        #         try:
        #             print("displaying image")
        #             image = Image.open(file_path_png)
        #             st.image(image, caption=filename, use_column_width=True)
        #         except Exception as e:
        #             st.error(f"Error displaying image {filename}: {str(e)}")
       
        # else:
        #     # Show most recent documents
        #     st.subheader('Recently Published:')
        #     new_embedding = get_embeddings("Business and Trade")
        #     recent_developments = find_most_similar(new_embedding, stored_embeddings)
        #     st.session_state['recent_developments'] = recent_developments

        #     for filename, similarity in recent_developments:
        #         if filename.endswith('.png.txt'):
        #             print("removing .txt from", filename)
        #             filename_png = os.path.splitext(filename)[0]
        #         elif filename.endswith('.png'):
        #             print("filename is already a png", filename)
        #             filename_png = filename
        #         else:
        #             print("filename is txt", filename)
        #             filename_png = os.path.splitext(filename)[0]
        #             filename_png = filename_png + ".png"
        #         print("filename_png:", filename_png)
        #         file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                
        #         # Displaying specific png page (e.g., page 10)
        #         # if filename_png.endswith('.png'):
        #         try:
        #             print("displaying image")
        #             image = Image.open(file_path_png)
        #             st.image(image, caption=filename, use_column_width=True)
        #         except Exception as e:
        #             st.error(f"Error displaying image {filename}: {str(e)}")


    # Column titles at the top
    # header_cols = st.columns([1, 3, 1, 2, 2, 3, 1])
    # header_cols[0].write("**ID**")
    # header_cols[1].write("**Name**")
    # header_cols[2].write("**Priority**")
    # header_cols[3].write("**Department**")
    # header_cols[4].write("**Concept**")
    # header_cols[5].write("**Summary**")
    # header_cols[6].write("**Action**")

    # results = [
    #     {"ID": 1, "Name": "Result A", "Detail": "More details about Result A including potential drafts and internal notes.", "Priority": "High", "Department": "Health", "Concept": "Innovation", "Summary": "Summary of A"},
    #     {"ID": 2, "Name": "Result B", "Detail": "More details about Result B including amendments and suggestions.", "Priority": "Medium", "Department": "Finance", "Concept": "Regulation", "Summary": "Summary of B"},
    # ]

    # for result in results:
    #     if result['ID'] not in st.session_state.hide_rows:
    #         with st.container():
    #             cols = st.columns([1, 3, 1, 2, 2, 3, 1])
    #             cols[0].markdown(f"**{result['ID']}**")
    #             cols[1].markdown(f"{result['Name']}")
    #             cols[2].markdown(f"{result['Priority']}")
    #             cols[3].markdown(f"{result['Dept']}")
    #             cols[4].markdown(f"{result['Concept']}")
    #             cols[5].markdown(f"{result['Summary']}")

    #             # Button to send/hide the row
    #             if cols[6].button("Send", key=f"send_{result['ID']}"):
    #                 st.session_state.hide_rows.append(result['ID'])

    #             # Expander for more details with two-column layout inside
    #             with st.expander("Show Details"):
    #                 left_col, right_col = st.columns(2)
    #                 with left_col:
    #                     st.write("Query:")
    #                     st.write("To ask the Secretary of State for Business and Trade, whether her Department plans to lower animal welfare standards and allow live animal transport as part of the UK-India free trade agreement.")
    #                     st.text_area("Draft Notes", value=result['Detail'], height=100)
    #                 with right_col:
    #                     st.write("Other Actions:")
    #                     st.button("Review", key=f"review_{result['ID']}")
    #                     st.button("Approve", key=f"approve_{result['ID']}")

    #             # Adding a subtle separation for aesthetic purposes
    #             st.markdown("---")

    # # CSS to style the rows and expander for a consistent and distinct look
    # st.markdown("""
    # <style>
    # .stContainer {
    #     background-color: #ffcccc; /* Light red color for the whole container */
    #     border-radius: 10px;
    #     margin-bottom: 5px; /* Smaller gap between rows */
    #     padding: 10px; /* Padding inside the container */
    # }
    # .stExpander {
    #     background-color: #f8f9fa; /* Lighter grey for expander to distinguish from the main row container */
    # }
    # .stMarkdown {
    #     background-color: transparent; /* Ensure text within markdown does not have a different background */
    # }
    # </style>
    # """, unsafe_allow_html=True)

def get_priority_color_ii(priority):
    if priority > 10:
        priority = 10  # Cap priority to 10 for calculation purposes
    blue_component = int((priority / 10) * 200)  # Scale to 0-255 range
    return f"#{255-blue_component:02x}{255-blue_component:02x}{255:02x}"

def add_timeline(df):
    fig = go.Figure()
    for _, row in df.iterrows():
        color = 'red' if row['Priority'] >= 8 else 'orange' if row['Priority'] >= 5 else 'green'
        fig.add_trace(go.Scatter(
            x=[row['Date']],
            y=[row['Sentiment']],
            mode='markers',
            marker=dict(size=row['Relevance']*50, color=color),
            # text=f"<b>Date:</b> {row['Date']}<br><b>Topic:</b> {row['Topic']}<br><b>Summary:</b> {row['Summary']}",
            # hoverinfo='text',
            name=row['Dept']
        ))
    fig.update_layout(
        title='Timeline of Document Prioritization',
        xaxis_title='Date',
        yaxis_title='Priority Level',
        showlegend=True
    )
    st.plotly_chart(fig)

def display_submission(submission, related_data):
    st.image('./data/profile.jpeg', width=100)  # Adjust path and size as necessary

    st.title('Submission Summary')
    st.json(submission['summary'])  # Displaying JSON for clarity or use a markdown for HTML styling

    st.title('Current Position')
    st.json(submission['position'])

    st.title('Related Data Timeline')
    add_timeline(related_data)

def main():
    # Using option_menu for navigation in the sidebar
    selected = option_menu(
        menu_title="Navigation",  # Optional title for the menu
        options=['OpenMiDAS', 'Triage', 'Draft'],  # Updated option names
        icons=['house-fill', 'arrow-left-right', 'inbox-fill'],  # Icons for visual appeal
        menu_icon="cast",  # Icon for the menu itself
        default_index=0,  # Default to the first item
        orientation="vertical"  # Vertical layout for the menu
    )

    if selected == 'OpenMiDAS':
        show_openq()
    elif selected == 'Triage':
        show_dashboard()
    elif selected == 'Draft':
        show_drafts()

if __name__ == "__main__":
    main()

# Create a Plotly figure for the timeline
fig = go.Figure()