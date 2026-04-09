import pandas as pd
import streamlit as st
import os
import google.generativeai as genai
import PIL.Image

# सेट API Key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBXlj6BPE6kJnafHryktvtEvn0RmRODmFw"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("models/gemini-2.5-flash")

# Function for analyzing attributes
def face_attribute(image):
    prompt = """
    You are an AI trained to analyze human attributes from images with high accuracy. 
    Carefully analyze the given image and return the following structured details:
    You have to return all results as you have the image, don't want any apologize or empty results.
    
    - Gender (Male/Female/Non-binary)
    - Age Estimate (e.g., 25 years)
    - Mood (e.g., Happy, Sad, Neutral, Excited)
    - Facial Expression (e.g., Smiling, Frowning, Neutral, etc.)
    - Glasses (Yes/No)
    - Beard (Yes/No)
    - Hair Color (e.g., Black, Blonde, Brown)
    - Eye Color (e.g., Blue, Green, Brown)
    - Headwear (Yes/No, specify type if applicable)
    - Emotions Detected (e.g., Joyful, Focused, Angry, etc.)
    - Confidence Level (Accuracy of prediction in percentage)
    """
    
    response = model.generate_content([prompt, image])
    return response.text.strip()

# Streamlit App
st.title("🧑‍💻 Human Face Attribute Analysis")
st.write("Upload an image to detect human attributes")

# Upload image
uploaded_image = st.file_uploader(
    "Upload an image of a human face", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_image is not None:
    image = PIL.Image.open(uploaded_image)

    # Create two columns
    col1, col2 = st.columns(2)

    # LEFT SIDE → Image
    with col1:
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # RIGHT SIDE → Results
    with col2:
        st.subheader("🧾 Detected Attributes")

        with st.spinner("🔍 Analyzing attributes..."):
            attributes = face_attribute(image)

        st.write(attributes)