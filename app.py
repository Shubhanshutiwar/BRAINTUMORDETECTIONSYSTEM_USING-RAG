import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import random
from PIL import Image

# LangChain & Vector DB Imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# -------------------------------------------------------------------------
# 1. SETUP KNOWLEDGE BASE FILE (If not exists)
# -------------------------------------------------------------------------
if not os.path.exists("tumor_guidelines.txt"):
    medical_text = """
    GLIOMA PROTOCOL:
    Gliomas are tumors that start in the glial cells of the brain. Patients diagnosed with Glioma typically require an urgent referral to neuro-oncology. Standard procedures involve high-resolution contrast-enhanced MRIs to track infiltration boundaries, and planning for surgical resection or biopsy depending on the tumor grade (I to IV).

    MENINGIOMA PROTOCOL:
    Meningiomas arise from the meninges—the membranes that surround the brain and spinal cord. Many meningiomas are benign and grow slowly. Clinical guidelines suggest monitoring small, asymptomatic tumors with active surveillance (serial scans every 6–12 months). Large or symptomatic meningiomas usually require surgical removal or targeted radiation therapy.

    PITUITARY PROTOCOL:
    Pituitary tumors develop in the pituitary gland at the base of the brain. They often affect hormone levels, leading to endocrine symptoms or vision issues due to pressure on the optic chiasm. Management guidelines include hormone level blood panels, visual field mapping, and treatment using specialized medication (like dopamine agonists) or transsphenoidal surgery.
    """
    with open("tumor_guidelines.txt", "w") as f:
        f.write(medical_text.strip())

# -------------------------------------------------------------------------
# 2. INITIALIZE RAG PIPELINE
# -------------------------------------------------------------------------
@st.cache_resource
def load_rag_system():
    loader = TextLoader("tumor_guidelines.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # Pulls the API key securely from local environment or Streamlit Secrets
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=api_key)
    
    system_prompt = (
        "You are an expert clinical AI assistant. Use the following pieces of retrieved context "
        "to answer the question. If you don't know the answer, use your medical knowledge to provide a "
        "safe, general guideline but state clearly that it was not in the immediate source documents.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# -------------------------------------------------------------------------
# 3. LOAD TRAINED CNN MODEL
# -------------------------------------------------------------------------
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('brain_tumor_model.h5')

# -------------------------------------------------------------------------
# 4. STREAMLIT UI LAYOUT
# -------------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor AI Assistant", layout="wide")

st.title("🧠 Medical AI: Brain Tumor Detection & RAG System")
st.write("Upload an MRI scan to perform classification and generate automated clinical insights.")

# Ensure Google API key is available before processing
api_key_check = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
if not api_key_check:
    st.warning("⚠️ GOOGLE_API_KEY not found in environment settings. RAG generation might fail. Please set it up before proceeding.")

# Sidebar Information / Placeholder Setup
st.sidebar.header("Reference Images Setup")
st.sidebar.info("To use the side-by-side comparison feature, make sure to add a healthy sample image named `healthy_placeholder.png` into your local project directory!")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image file...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and process image
    image = Image.open(uploaded_file)
    
    # Run Predictions
    model = load_cnn_model()
    labels = ['glioma', 'meningioma', 'notumor', 'pituitary']
    
    # Prepare image array for CNN model
    img_resized = image.resize((224, 224))
    img_array = np.array(img_resized)
    
    # Handle cases where image is grayscale
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4: # Handle PNG alpha channels
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    img_array = img_array / 255.0
    input_batch = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing scan characteristics..."):
        preds = model.predict(input_batch)
        class_idx = np.argmax(preds)
        detected_class = labels[class_idx]
        confidence = preds[0][class_idx] * 100

    # UI Visual Columns Setup
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🖼️ Uploaded Patient Scan")
        st.image(image, use_container_width=True)
        if detected_class == 'notumor':
            st.success(f"**AI Prediction: HEALTHY / NO TUMOR ({confidence:.1f}%)**")
        else:
            st.error(f"**AI Prediction: {detected_class.upper()} DETECTED ({confidence:.1f}%)**")

    with col2:
        st.subheader("🔬 Baseline Comparison Profile")
        if detected_class != 'notumor' and os.path.exists("healthy_placeholder.png"):
            healthy_ref = Image.open("healthy_placeholder.png")
            st.image(healthy_ref, caption="Healthy Reference Profile Baseline", use_container_width=True)
        elif detected_class != 'notumor':
            st.warning("Add a `healthy_placeholder.png` file to your folder to see side-by-side normal structure comparisons automatically.")
        else:
            st.info("The uploaded scan matches standard normal structural expectations.")

    # --- RAG Report Block ---
    st.markdown("---")
    if detected_class == 'notumor':
        st.subheader("📋 Clinical Report Summary")
        st.success("The computer vision pipeline confirms normal tissue distribution. No active structural abnormalities matching targeted primary tumor signatures were identified.")
    else:
        st.subheader("📋 Generated AI Clinical Insight Report (RAG)")
        with st.spinner("Accessing guideline database and compiling report summaries..."):
            try:
                rag_chain = load_rag_system()
                prompt_query = f"An AI model classified an MRI brain scan as a {detected_class} with {confidence:.1f}% confidence. Summarize the critical medical information regarding this condition and outline the necessary next diagnostic or treatment guidelines based exclusively on your source data."
                
                response = rag_chain.invoke({"input": prompt_query})
                st.info(response['answer'])
            except Exception as e:
                st.error(f"Failed to generate RAG report. Error information: {e}")