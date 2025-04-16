import os
import base64
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st
from PIL import Image
from io import BytesIO

# Set your API Key
GOOGLE_API_KEY = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize the Medical Agent
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# AI Query Prompt
query = """
You are a highly skilled medical imaging expert...
(keep your existing detailed prompt here)
"""

# Analyze image function
def analyze_medical_image(image_path):
    image = PILImage.open(image_path)
    aspect_ratio = image.width / image.height
    new_height = int(500 / aspect_ratio)
    resized_image = image.resize((500, new_height))

    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)

    agno_image = AgnoImage(filepath=temp_path)

    try:
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"⚠️ Analysis error: {e}"
    finally:
        os.remove(temp_path)

# Streamlit UI setup
st.set_page_config(page_title="Medical Image Analysis Agent", layout="centered")

# Add Google Font for Poppins
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        
        body {
            font-family: 'Poppins', sans-serif;
        }

        [data-testid="stSidebar"] {
            width: 320px !important;
            background: linear-gradient(to bottom right, #003D62, #3497BA);
            color: white;
            padding: 20px;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Poppins', sans-serif;
        }

        p {
            font-family: 'Poppins', sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# -- 👨‍⚕️ Title & Introduction
st.markdown("""
    <h1 style='font-size: 30px; color: #2f729b; margin-bottom: 20px;'>
        Medical Image Analysis Agent
    </h1>
""", unsafe_allow_html=True)
st.markdown("""
    Welcome to the **Medical Image Analysis Agent**!  
    Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.), and our AI-powered system will analyze it, providing detailed findings, diagnosis, and research insights.  
    Let's get started!
""")

# -- 🖼 Logo with Rounded Corners in Sidebar
def get_base64_logo(img_path):
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

logo_base64 = get_base64_logo("static/Hoonartek-V25-White-Color.png")  # Update path as needed

st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 220px; border-radius: 0px; margin-bottom: 20px;" />
    </div>
    """,
    unsafe_allow_html=True
)

# -- 🧠 Use Case Description
st.sidebar.title("Use Case Details")
st.sidebar.markdown(
    """
    This module uses Gemini 2.0 Flash and Agno Agents to analyze radiology images, delivering expert AI insights for diagnostics and screening.
    """
)

st.sidebar.header("Model Used:")
st.sidebar.markdown("Gemini 2.0 Flash")

# -- 📤 Upload Image
uploaded_file = st.sidebar.file_uploader("Choose a medical image file", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.sidebar.button("Analyze Image"):
        with st.spinner("🔍 Analyzing the image... Please wait."):
            image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            report = analyze_medical_image(image_path)
            st.subheader("📋 Analysis Report")
            st.markdown(report, unsafe_allow_html=True)
            os.remove(image_path)
else:
    st.warning("⚠️ Please upload a medical image to begin analysis.")
