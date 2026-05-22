import streamlit as st
import tensorflow as tf
import numpy as np
import os
from pathlib import Path
from PIL import Image
import io
import gdown
import zipfile
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(
    page_title="COVID-19 X-Ray Detector",
    page_icon="🫁",
    layout="wide"
)

# Global styles for improved look-and-feel
st.markdown("""
<style>
:root{
    --bg-1: #071029; /* deep navy */
    --bg-2: #0b1220; /* very dark */
    --card: #081428;
    --text: #e6f6ff;
    --muted: #9fb7c8;
    --accent-start: #06b6d4; /* vivid cyan */
    --accent-end: #ff7a18;   /* warm orange */
    --uploader-bg: #9be7ff; /* light blue for uploader */
    --uploader-text: #000000;
    --sidebar-bg: rgba(200, 255, 220, 0.06); /* subtle light green */
    --sidebar-border: rgba(200, 255, 220, 0.12);
}
body, .stApp {
    background: linear-gradient(180deg,var(--bg-1),var(--bg-2));
    color: var(--text);
    font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
.block-container{padding:2rem 1.5rem; max-width:1100px; margin:0 auto}
.sidebar .sidebar-content{
    background: var(--sidebar-bg);
    padding:1rem;
    border-radius:10px;
    border:1px solid var(--sidebar-border);
}
.result-card{
    background:var(--card);
    padding:18px;
    border-radius:12px;
    box-shadow:0 12px 40px rgba(2,6,23,0.6);
    text-align:center;
}
.result-title{margin:0; color:var(--text); font-size:20px; font-weight:700}
.result-confidence{color:var(--muted); font-size:16px; margin-top:8px}
.stButton>button{
    background: linear-gradient(90deg,var(--accent-start),var(--accent-end));
    color:#031026;
    border-radius:10px;
    padding:10px 14px;
    font-weight:700;
    box-shadow:0 10px 30px rgba(6,182,212,0.12);
}
.stButton>button:hover{filter:brightness(1.06); transform:translateY(-1px)}
img{border-radius:8px; box-shadow:0 16px 48px rgba(2,6,23,0.6)}
.prediction-card{background:linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)); padding:14px; border-radius:12px; border:1px solid rgba(255,255,255,0.03)}

/* File uploader visibility on dark background (high-contrast) */
.stFileUploader, input[type="file"] {
    color: var(--text) !important;
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px;
    padding: 8px;
    border: 1px solid rgba(255,255,255,0.08) !important;
}
.stFileUploader .stButton>button{ color: #ffffff !important; }
/* Force uploader text visibility */
.stFileUploader * { color: var(--text) !important; }
/* Primary file selector button styling */
input[type=file]::file-selector-button,
input[type=file]::-webkit-file-upload-button,
input[type=file]::-moz-file-upload-button {
    color: var(--uploader-text) !important;
    background: var(--uploader-bg) !important;
    border: none !important;
    padding: 10px 16px !important;
    border-radius: 10px !important;
    font-weight: 800 !important;
    cursor: pointer !important;
    box-shadow: 0 8px 28px rgba(6,182,212,0.14) !important;
    z-index: 10000 !important;
}
/* WebKit / Blink fallback */
input[type=file]::-webkit-file-upload-button {
    color: var(--uploader-text) !important;
    background: var(--uploader-bg) !important;
    border: none !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer;
}
/* Firefox fallback */
input[type=file]::-moz-file-upload-button {
    color: var(--uploader-text) !important;
    background: var(--uploader-bg) !important;
    border: none !important;
    padding: 8px 12px !important;
    border-radius: 8px !important;
    cursor: pointer;
}
/* Force a highly-visible white button with black text as a last-resort override */
input[type=file]::file-selector-button,
input[type=file]::-webkit-file-upload-button,
input[type=file]::-moz-file-upload-button,
input[type=file] + label,
.stFileUploader button,
.stFileUploader .stButton>button {
    background: var(--uploader-bg) !important;
    color: var(--uploader-text) !important;
    -webkit-text-fill-color: var(--uploader-text) !important;
    opacity: 1 !important;
    font-weight: 900 !important;
    border: 1px solid rgba(0,0,0,0.08) !important;
    box-shadow: 0 6px 18px rgba(2,6,23,0.12) !important;
}
/* Ensure text is not lightened by filters or blend modes */
input[type=file]::file-selector-button,
input[type=file]::-webkit-file-upload-button {
    filter: none !important;
    mix-blend-mode: normal !important;
}
/* Make sure uploader controls are on top and clearly visible */
.stFileUploader, .stFileUploader * {
    z-index: 9999 !important;
    font-weight: 600 !important;
}
/* Ensure Streamlit's internal uploader button (when present) uses black text */
.stFileUploader .stButton>button{ color: #000000 !important; }

/* Recommendation boxes: use subtle dark background so light text is readable */
.recommendation-box {
    background-color: rgba(255,255,255,0.03);
    color: var(--text);
    padding: 10px;
    border-radius: 8px;
    margin: 6px 0;
    border-left: 4px solid transparent;
}
</style>
""", unsafe_allow_html=True)

# Inject JS to force uploader button visibility (runs in browser)
components.html("""
<script>
function styleUploaders(){
    try{
        const wrappers = document.querySelectorAll('.stFileUploader, [data-testid="stFileUploader"]');
        wrappers.forEach(w => {
            const buttons = w.querySelectorAll('button, [role="button"], label, input[type=file] + label');
            buttons.forEach(b => {
                try{
                    const bg = (getComputedStyle(document.documentElement).getPropertyValue('--uploader-bg') || '#ffd966').trim() || '#ffd966';
                    b.style.background = bg;
                }catch(e){ b.style.background = '#ffd966'; }
                b.style.color = '#000000';
                b.style.fontWeight = '900';
                b.style.border = '1px solid rgba(0,0,0,0.08)';
                b.style.boxShadow = '0 6px 18px rgba(2,6,23,0.12)';
                b.style.opacity = '1';
            });
            const inputs = w.querySelectorAll('input[type=file]');
            inputs.forEach(i => { i.style.color = '#000000'; i.style.opacity = '1'; });
        });
    } catch(e){console.log('uploader style script error', e);}  
}
styleUploaders();
new MutationObserver(styleUploaders).observe(document.body, {childList:true, subtree:true});
</script>
""", height=0)

# Model configuration
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_NAME = "model_2_vgg16.keras"

CLASS_NAMES = ["COVID-19", "Normal", "Viral Pneumonia"]
CLASS_COLORS = {"COVID-19": "#FF4444", "Normal": "#00AA00", "Viral Pneumonia": "#FFA500"}

@st.cache_resource
def download_model():
    """Download model from Google Drive if not present"""
    target = MODELS_DIR / DEFAULT_MODEL_NAME
    if target.exists():
        return True
    
    try:
        file_id = "17OucXrgs5Ovv1Q0W9xHg_Y4kYwuRLQ9E"
        zip_path = BASE_DIR / "models.zip"
        
        st.info("📥 Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, str(zip_path), quiet=False)
        
        st.info("📦 Extracting model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(BASE_DIR)
        
        if zip_path.exists():
            os.remove(zip_path)
        
        st.success("✅ Model ready!")
        return True
    except Exception as e:
        st.error(f"❌ Model download failed: {e}")
        return False

@st.cache_resource
def load_model(model_path: str):
    """Load the chosen Keras model from disk.

    If the default model is requested and missing, attempt auto-download.
    """
    path = Path(model_path)

    # If user chose the default model name and it's missing, try download
    if path.name == DEFAULT_MODEL_NAME and not path.exists():
        if not download_model():
            return None

    if not path.exists():
        st.error(f"Model file not found at {path}")
        return None

    try:
        model = tf.keras.models.load_model(str(path))
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image_data):
    """Preprocess image for VGG16 (RGB, 224x224, normalized)"""
    try:
        # Convert to PIL Image and ensure RGB
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize to 224x224
        img_resized = img.resize((224, 224))
        
        # Convert to numpy array
        img_array = np.asarray(img_resized, dtype=np.float32)
        
        # Normalize pixel values (0-1)
        img_normalized = img_array / 255.0
        
        # Add batch dimension
        img_input = np.expand_dims(img_normalized, axis=0)
        
        return img_input, np.array(img_resized)
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None, None

def main():
    # Title section
    st.title("🫁 COVID-19 Chest X-Ray Detector")
    st.markdown("Advanced Deep Learning Model for Medical Image Classification")
    st.markdown("---")
    
    # Sidebar information
    with st.sidebar:
        st.title("ℹ️ About")
        st.markdown("""
        ### Application Features
        - **Model**: VGG16 Transfer Learning
        - **Classes**: COVID-19, Normal, Viral Pneumonia
        - **Input Size**: 224 × 224 pixels
        - **Framework**: TensorFlow/Keras
        
        ### Instructions
        1. Upload a chest X-ray image
        2. Click "Predict" to analyze
        3. View results with confidence scores
        
        ### ⚠️ Disclaimer
        This tool is for **educational purposes only**. 
        **NOT a substitute for professional medical diagnosis**. 
        Always consult a healthcare professional.
        """)
    
    # Model selector in sidebar (allows choosing the notebook's best model)
    # Map on-disk model filenames to friendly display names
    filename_to_display = {
        "model_1a.keras": "Basic CNN (1a)",
        "model_1b.keras": "Deep CNN (1b)",
        "model_2_vgg16.keras": "VGG16 Transfer (2)",
        "model_3_resnet50_aug.keras": "ResNet50 + Aug (3)",
        "best_tuned_model.keras": "Best Tuned Model",
        DEFAULT_MODEL_NAME: "VGG16 Transfer (2)"
    }

    # Reverse map for lookup when user selects a display name
    display_to_filename = {v: k for k, v in filename_to_display.items()}

    available_files = []
    if MODELS_DIR.exists():
        available_files = [p.name for p in MODELS_DIR.glob("*.keras")]

    # Ensure default is present
    if DEFAULT_MODEL_NAME not in available_files:
        available_files.append(DEFAULT_MODEL_NAME)

    # Build display options preserving a sensible ordering
    preferred_order = [
        "model_1a.keras",
        "model_1b.keras",
        "model_2_vgg16.keras",
        "model_3_resnet50_aug.keras",
        "best_tuned_model.keras",
    ]

    options = []
    # Add preferred models first if available
    for fname in preferred_order:
        if fname in available_files:
            options.append(filename_to_display.get(fname, fname))

    # Add any remaining files (fallback to filename)
    for fname in sorted(set(available_files) - set(preferred_order)):
        options.append(filename_to_display.get(fname, fname))

    # Determine default selection index
    default_display = filename_to_display.get(DEFAULT_MODEL_NAME, DEFAULT_MODEL_NAME)
    default_index = options.index(default_display) if default_display in options else 0

    selected_display = st.sidebar.selectbox("Select model to use:", options=options, index=default_index)
    selected_model = display_to_filename.get(selected_display, selected_display)
    selected_model_path = MODELS_DIR / selected_model

    # Load model
    model = load_model(str(selected_model_path))
    if model is not None:
        st.sidebar.success(f"Loaded model: {selected_model}")
    
    if model is None:
        st.error("❌ Could not load the model. Please check your internet connection.")
        return
    
    # Upload section
    st.header("📤 Upload Chest X-Ray Image")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a chest X-ray image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"]
        )
    
    with col2:
        st.info("""
        **Requirements:**
        - Clear chest X-ray
        - Good contrast
        - 224×224 or larger
        """)
    
    # Handle file upload
    if uploaded_file is not None:
        image_data = uploaded_file.read()
        
        # Display uploaded image
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Uploaded Image")
            st.image(Image.open(io.BytesIO(image_data)), use_container_width=True)
        
        with col2:
            st.subheader("📊 Analysis Results")
            
            # Prediction button
            if st.button("🔬 Predict", use_container_width=True):
                with st.spinner("Analyzing image..."):
                    # Preprocess
                    img_input, img_display = preprocess_image(image_data)
                    
                    if img_input is None:
                        st.error("Failed to process image")
                        return
                    
                    # Make prediction
                    predictions = model.predict(img_input, verbose=0)[0]
                    
                    # Get class and confidence
                    pred_index = int(np.argmax(predictions))
                    pred_class = CLASS_NAMES[pred_index]
                    confidence = float(predictions[pred_index] * 100)
                    
                    # Display result card (styled)
                    color = CLASS_COLORS[pred_class]
                    st.markdown(f"""
                    <div class="result-card" style="border-left:6px solid {color};">
                        <h3 class="result-title">{pred_class}</h3>
                        <p class="result-confidence"><b>{confidence:.2f}%</b></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Store predictions for later display
                    st.session_state.predictions = predictions
                    st.session_state.pred_class = pred_class
                    st.session_state.confidence = confidence
        
        # Display detailed predictions if available
        if "predictions" in st.session_state:
            st.markdown("---")
            st.subheader("📈 Confidence Scores for All Classes")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Bar chart
                fig, ax = plt.subplots(figsize=(10, 5))
                colors_list = [CLASS_COLORS[cls] for cls in CLASS_NAMES]
                bars = ax.barh(CLASS_NAMES, st.session_state.predictions * 100, color=colors_list)
                ax.set_xlabel('Confidence (%)', fontsize=12)
                ax.set_title('Model Prediction Confidence', fontsize=14, fontweight='bold')
                ax.set_xlim([0, 100])
                
                for bar in bars:
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2,
                            f'{width:.1f}%', ha='left', va='center', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Detailed Breakdown:**")
                for i, (class_name, prob) in enumerate(zip(CLASS_NAMES, st.session_state.predictions)):
                    st.progress(float(prob), text=f"{class_name}: {prob*100:.2f}%")
            
            # Medical guidance
            st.markdown("---")
            st.subheader("💊 Medical Guidance")
            
            guidance = {
                "COVID-19": {
                    "icon": "🔴",
                    "color": "#FF4444",
                    "recommendations": [
                        "🏥 Seek immediate medical attention",
                        "🔬 Get RT-PCR test confirmation",
                        "🏠 Self-isolate to prevent transmission",
                        "📞 Contact healthcare provider",
                        "⚠️ Monitor symptoms closely"
                    ]
                },
                "Normal": {
                    "icon": "🟢",
                    "color": "#00AA00",
                    "recommendations": [
                        "✅ Keep up with regular health checkups",
                        "💪 Maintain healthy lifestyle",
                        "🫁 Avoid smoking and air pollution",
                        "🏃 Regular exercise",
                        "⌛ Monitor for any new symptoms"
                    ]
                },
                "Viral Pneumonia": {
                    "icon": "🟠",
                    "color": "#FFA500",
                    "recommendations": [
                        "👨‍⚕️ Consult a healthcare professional",
                        "💊 Follow prescribed treatment",
                        "🛏️ Rest and stay hydrated",
                        "🌡️ Monitor temperature and symptoms",
                        "👥 Avoid close contact with others"
                    ]
                }
            }
            
            pred = st.session_state.pred_class
            rec = guidance[pred]
            
            st.markdown(f"### {rec['icon']} {pred}")
            for recommendation in rec['recommendations']:
                st.markdown(f"<div class='recommendation-box' style='border-left:4px solid {rec['color']};'>{recommendation}</div>", unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 12px;'>
        <p>
        ⚕️ <b>Medical Disclaimer:</b> This application is for educational purposes only. 
        It should not be used as a substitute for professional medical advice, diagnosis, or treatment.
        Always consult with a qualified healthcare provider.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
