import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from PIL import Image
from scipy.spatial.distance import cosine

# For embedding and similarity functions
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

from util import *

from dotenv import load_dotenv
load_dotenv()

import openai
import anthropic
from anthropic import Anthropic
import base64

# Configure clients
openai_api_key = os.getenv("OPENAI_API_KEY")
ant_api_key = os.getenv("ANTHROPIC_API_KEY")
openai_client = openai.Client(api_key=openai_api_key)
anthropic_client = anthropic.Client(api_key=ant_api_key)

st.set_page_config(layout="wide")

fields = [  
  {'title': "Department", 'tag': 'department', 'description': 'This is always capitalised. The government department which has written the response to the query.'},
  {'title': "Subject", 'tag': 'subject', 'description': 'The specifics of the nature of the contact e.g. Arms Trade: Israel, otherwise infer this yourself from the context. Usually a colon separates the subject from a short description. Include this.'},
  {'title': "Questioner", 'tag': 'questioner', 'description': 'The name of the person asking the question. There may be multiple names of questioners , they are identified by not being indented.'},
  {'title': "ID", 'tag': 'id', 'description': 'The ID of the response in square brackets. Also return the URL associated with the hyperlink.'},
  {'title': "Question", 'tag': 'question', 'description': 'The question being asked. There may be multiple questions being asked, they are identified by not being indented.'},
  {'title': "Respondent", 'tag': 'respondent', 'description': "Identify this by the indent. The name of the Member of Parliament who has responded to written the question"},
  {'title': "Answer", 'tag': 'answer', 'description': "Identify this by the indent. The answer provided by Parliament to the question."}
]

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

def prompt():
    fields_string = "\n".join(f"- {field['title']}: {field['description']}. Wrap this field in <{field['tag']}> tags." for field in fields)
    return f"""
Transcribe the text in this image in full, wrapped in <transcription> tags.
Please also extract the following fields:
{fields_string}
    """.strip()

embeddings_file = "/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/embeddings.pkl"
stored_embeddings = load_embeddings(embeddings_file)

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

        # Image upload
        uploaded_image = st.file_uploader("Upload an image for transcription", type=['png', 'jpg'])
        if uploaded_image is not None:

            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Process image
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
                parsed_text = response.content[0].text
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

                # Processing errors meant different MPs have different formats
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

                # Display appropriate images 
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
                
                # Display appropriate images 
                try:
                    print("displaying image")
                    image = Image.open(file_path_png)
                    st.image(image, caption=filename, use_column_width=True)
                except Exception as e:
                    st.error(f"Error displaying image {filename}: {str(e)}")



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
        
    # Show the figure in Streamlit app
    st.plotly_chart(fig)

def get_priority_color(priority):
    # Scale the priority to a 0-255 range for red color intensity
    if priority > 10:
        priority = 10 
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
    df['Relevance'] = df['Priority'] / 10
    return df

def show_dashboard():
    st.title('Triaging Dashboard')
    df, dfa = fake_data()
    col1, col2 = st.columns([2, 2])

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

    if response is not None and response.choices:
        draft_response_text = response.choices[0].message.content.strip()

        print("Draft response:", draft_response_text)
        return draft_response_text
    else:
        print("Failed to get a valid response.")
        return "No response generated."

def show_drafts():
    st.title('Drafts')
    st.write("Inbox:")

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

                with st.expander("Details"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Summary: {result['Topic']}")
                        st.write(f"Date: {result['Date']}")
                        st.write(f"From: {result['From']}")
                        st.write("Message: \n\nTo ask the Secretary of State for Business and Trade to ban the export of Gregs Hands. They are a valuable U.K asset and should be protected from foreign interference.")
                        
                        draft_key = f'draft_text_{result["ID"]}'
                        user_input = st.text_area("Draft Message", value=drafted, height=300, key=draft_key)

                        if user_input:
                            new_embedding = get_embeddings(user_input)
                            similar_images = find_most_similar(new_embedding, stored_embeddings)
                            st.session_state['similar_images'] = similar_images

                        send_btn_key = f"send_{result['ID']}"
                        if st.button("Send", key=send_btn_key):
                            st.session_state.hide_rows.append(result['ID'])
                            if result['ID'] in st.session_state.get('expanded_rows', []):
                                st.session_state['expanded_rows'].remove(result['ID'])

                    with col2:
                        extra_instructions_key = f'extra_instructions_{result["ID"]}'
                        print("Session state key 4:", draft_key)
                        extra_instructions = st.text_area("Add Extra Instructions Here:", value="Double space after all full stops. \n\nNever include the words estoppel, catastrophe, or any unnecessary epithets. \n\nAlways be polite and use formal language.", height=150, key=extra_instructions_key)
                        
                        rewrite_btn_key = f'rewrite_{result["ID"]}'
                        if st.button('Rewrite', key=rewrite_btn_key):
                            message = "From: Mr Edward Scissorhands. To ask the Secretary of State for Business and Trade to ban the export of Greg's Hands. They are a valuable U.K asset and should be protected from foreign interference."
                            new_draft = draft_response(message, extra_instructions)
                            print("New draft response GENERATED:", new_draft)
                            st.session_state[draft_key] = new_draft  # Update the draft in the session state
                            print("Experimental rerun...")
                            st.experimental_rerun()

                        if 'similar_images' in st.session_state:
                            st.subheader('Similar Historic Statements:')
                            for filename, similarity in st.session_state['similar_images']:
                                st.write(f"**{filename}** - Similarity: {similarity:.4f}")

                                print("\n\n New search \n\n")

                                # print("displaying title:", filename)
                                if filename.endswith('.png.txt'):
                                    # print("removing .txt from", filename)
                                    filename_png = os.path.splitext(filename)[0]
                                elif filename.endswith('.png'):
                                    # print("filename is already a png", filename)
                                    filename_png = filename
                                else:
                                    # print("filename is txt", filename)
                                    filename_png = os.path.splitext(filename)[0]
                                    filename_png = filename_png + ".png"
                                # print("filename_png:", filename_png)
                                file_path_png = os.path.join('/Users/oscarmoxon/Desktop/AI Projects/hackathon/data_files/comparisons', filename_png)
                                
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
                                
                                try:
                                    print("displaying image")
                                    image = Image.open(file_path_png)
                                    st.image(image, caption=filename, use_column_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying image {filename}: {str(e)}")

def get_priority_color_ii(priority):
    if priority > 10:
        priority = 10 
    blue_component = int((priority / 10) * 200)
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
    st.image('./data/profile.jpeg', width=100) 

    st.title('Submission Summary')
    st.json(submission['summary']) 

    st.title('Current Position')
    st.json(submission['position'])

    st.title('Related Data Timeline')
    add_timeline(related_data)

def main():
    # Using option_menu for navigation in the sidebar
    selected = option_menu(
        menu_title="Navigation", 
        options=['OpenMiDAS', 'Triage', 'Draft'], 
        icons=['house-fill', 'arrow-left-right', 'inbox-fill'], 
        menu_icon="cast",
        default_index=0,
        orientation="vertical" 
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