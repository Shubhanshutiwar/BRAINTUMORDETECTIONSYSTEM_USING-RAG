# 🧠 Medical AI: Brain Tumor Detection & RAG Clinical Report System

An advanced, end-to-end medical decision support application that combines **Computer Vision (Deep Learning)** and **Generative AI (Retrieval-Augmented Generation)** to assist clinical workflows. 

The system analyzes patient MRI scans to classify brain pathologies and automatically queries a vectorized local medical database to generate a structured, context-aware clinical report with official protocols and next diagnostic steps.

---

## 🚀 Key Features

* **CNN Medical Image Classification:** Leverages Transfer Learning via a tuned network architecture to classify raw brain MRI images into 4 distinct categories (`Glioma`, `Meningioma`, `Pituitary`, or `Healthy / No Tumor`).
* **Visual Anomaly Contrast Integration:** Automatically generates a side-by-side comparative layout matching the patient's anomaly scan against a baseline structural control scan for explicit visual auditing.
* **Modern Multimodal RAG Pipeline:** Employs a local vector store built on a clinical guideline knowledge base to extract matching healthcare guidelines using state-of-the-art sentence embeddings.
* **LLM Clinical Insights:** Integrates with the `Gemini-2.5-Flash` architecture via modern LangChain orchestration layers to build deterministic, evidence-grounded reports—minimizing standard LLM hallucination risks.
* **Production-Ready Web Dashboard:** A highly interactive, cloud-deployable user interface engineered using Streamlit.

---

## 🛠️ System Architecture

1. **Intake & Diagnostics:** Patient uploads a digital MRI scan (`.png`, `.jpg`, `.jpeg`).
2. **Feature Extraction:** The underlying TensorFlow model weights calculate feature matrices to predict clinical categorization.
3. **Knowledge Retrieval:** If an abnormality is flagged, the system maps the classification into a vector space, querying a `ChromaDB` instance initialized with `all-MiniLM-L6-v2` HuggingFace text embeddings.
4. **Context Augmentation & Generation:** High-priority document chunks are isolated and piped as context alongside the classification metrics into `ChatGoogleGenerativeAI` to build the final synthesis summary.

---

## 📂 Project Repository Structure

```text
├── .streamlit/
│   └── config.toml          # Streamlit UI configuration parameters
├── app.py                   # Main Streamlit web application dashboard logic
├── requirements.txt         # Production python dependencies manifest
├── runtime.txt              # Cloud environment python engine specification
├── brain_tumor_model.h5     # Pre-trained deep learning CNN classification weights
├── tumor_guidelines.txt     # Local clinical text knowledge base file
└── healthy_placeholder.png  # Healthy reference control scan for UI contrast


💻 Local Installation & Setup
Follow these steps to configure and run the decision system locally on your environment console:

1. Clone the Repository
Bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git)
cd YOUR_REPOSITORY_NAME
2. Configure Environment Environment (Python 3.11 Recommended)
Ensure you are running an environment compatible with stable machine learning wheel distributions:

Bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Inject Your Gemini Credentials
The system requires an active API token from Google AI Studio to process the text generation chain:

Windows (CMD): set GOOGLE_API_KEY="AIzaSyYourKeyHere"

Mac/Linux: export GOOGLE_API_KEY="AIzaSyYourKeyHere"

5. Launch the System
Bash
streamlit run app.py
📊 Evaluation & Metrics Summary
The underlying classification core was trained utilizing an automated data-augmented workflow. Performance was validated across stratified sets evaluating Precision, Sensitivity (Recall), and F1-Scores to optimize against false-negative clinical outcomes.
Real-time tracking of training progress was monitored using dynamic Loss Reduction Convergence Curves and Categorical Accuracy Trends.

📜 Disclaimer
This system is intended strictly as an educational proof-of-concept portfolio project. It is not approved for official diagnostic clinical use, nor should it replace professional medical imaging evaluation workflows.
