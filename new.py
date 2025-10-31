import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers, models
from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import cv2
import pandas as pd
from io import BytesIO

st.set_page_config(
    page_title="Kidney Health Analytics",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = 'KIDENY_MODEL.h5'
IMG_SIZE = 128
CLASS_NAMES = ['Cyst', 'Normal', 'Stone', 'Tumor']
NUM_CLASSES = len(CLASS_NAMES)

CLINICAL_INFO = {
    'Cyst': {
        'description': 'Fluid-filled sacs in the kidney',
        'severity': 'Low to Moderate',
        'treatment': 'Usually monitored; drainage if symptomatic',
        'urgency': 'Non-urgent',
        'color': '#17a2b8'
    },
    'Normal': {
        'description': 'Healthy kidney tissue',
        'severity': 'None',
        'treatment': 'No treatment required',
        'urgency': 'None',
        'color': '#28a745'
    },
    'Stone': {
        'description': 'Hard deposits formed in the kidney',
        'severity': 'Moderate to High',
        'treatment': 'Hydration, medication, or surgical removal',
        'urgency': 'Moderate to High',
        'color': '#ffc107'
    },
    'Tumor': {
        'description': 'Abnormal growth of tissue in the kidney',
        'severity': 'High',
        'treatment': 'Further imaging and oncological consultation required',
        'urgency': 'High - Immediate attention needed',
        'color': '#dc3545'
    }
}

@st.cache_resource
def load_model():
    inference_model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES)
    ])
    inference_model.load_weights(MODEL_PATH)
    return inference_model

def detect_image_modality(image):
    img_array = np.array(image.convert('L'))
    mean_intensity = np.mean(img_array)
    std_intensity = np.std(img_array)
    
    if std_intensity < 40 and mean_intensity < 100:
        return "Ultrasound"
    else:
        return "CT Scan"

def preprocess_image_advanced(image, modality):
    img = image.convert('RGB')
    img_array = np.array(img)
    
    if modality == "Ultrasound":
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.5)
        img_array = np.array(img)
        img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
        img = Image.fromarray(img_array)
        
    elif modality == "CT Scan":
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(1.2)
        img_array = np.array(img.convert('L'))
        img_array = cv2.equalizeHist(img_array)
        img = Image.fromarray(img_array).convert('RGB')
    
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    return img_array, img

def get_grad_cam(model, image, class_idx):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer
            break
    
    if last_conv_layer is None:
        return None
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [last_conv_layer.output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        loss = predictions[:, class_idx]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()

def predict_with_explanation(image, modality):
    processed_image, enhanced_img = preprocess_image_advanced(image, modality)
    
    predictions = model.predict(processed_image, verbose=0)
    score = tf.nn.softmax(predictions[0])
    
    class_idx = np.argmax(score)
    class_name = CLASS_NAMES[class_idx]
    confidence = 100 * np.max(score)
    
    all_probs = {CLASS_NAMES[i]: float(score[i] * 100) for i in range(len(CLASS_NAMES))}
    
    try:
        heatmap = get_grad_cam(model, processed_image, class_idx)
    except:
        heatmap = None
    
    return class_name, confidence, all_probs, heatmap, enhanced_img

def generate_clinical_report(class_name, confidence, all_probs, modality, patient_info=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
KIDNEY IMAGING ANALYSIS REPORT

Report Generated: {timestamp}
Analysis Method: AI-Assisted Diagnosis using Deep Learning CNN
Image Modality: {modality}

PATIENT INFORMATION
"""
    
    if patient_info and any(patient_info.values()):
        report += f"""
Patient ID: {patient_info.get('id', 'Not provided')}
Age: {patient_info.get('age', 'Not provided')}
Gender: {patient_info.get('gender', 'Not provided')}
"""
    else:
        report += "Patient information not provided\n"
    
    report += f"""
DIAGNOSIS SUMMARY

Primary Finding: {class_name}
Confidence Level: {confidence:.1f}%
Clinical Significance: {CLINICAL_INFO[class_name]['severity']}

DETAILED ANALYSIS

Condition Description: {CLINICAL_INFO[class_name]['description']}

Probability Distribution:
"""
    
    for condition, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        report += f"- {condition}: {prob:.1f}%\n"
    
    report += f"""
CLINICAL RECOMMENDATIONS

Treatment Approach: {CLINICAL_INFO[class_name]['treatment']}
Urgency Level: {CLINICAL_INFO[class_name]['urgency']}

IMPORTANT NOTES

- This AI analysis is intended as a diagnostic aid and should not replace clinical judgment
- Further clinical correlation and additional imaging may be required
- Results should be interpreted by a qualified radiologist or physician
- For urgent findings, immediate clinical attention is recommended

This report was generated by an AI system trained on kidney imaging data. Please consult with healthcare professionals for medical decisions.
"""
    
    return report

def create_visualization_dashboard(all_probs, heatmap, enhanced_img):
    plt.style.use('dark_background')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.patch.set_facecolor('#0E1117')
    fig.suptitle('Clinical Analysis Dashboard', fontsize=16, fontweight='bold', color='white')
    
    conditions = list(all_probs.keys())
    probabilities = list(all_probs.values())
    colors = [CLINICAL_INFO[cond]['color'] for cond in conditions]
    
    axes[0, 0].barh(conditions, probabilities, color=colors, alpha=0.8)
    axes[0, 0].set_xlabel('Probability (%)', fontweight='bold', color='white')
    axes[0, 0].set_title('Condition Probability Distribution', fontweight='bold', color='white')
    axes[0, 0].set_xlim(0, 100)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_facecolor('#0E1117')
    axes[0, 0].tick_params(colors='white')
    
    for i, (cond, prob) in enumerate(zip(conditions, probabilities)):
        axes[0, 0].text(prob + 1, i, f'{prob:.1f}%', va='center', fontweight='bold', color='white')
    
    axes[0, 1].imshow(enhanced_img)
    axes[0, 1].set_title('Enhanced Input Image', fontweight='bold', color='white')
    axes[0, 1].axis('off')
    
    if heatmap is not None:
        im = axes[1, 0].imshow(heatmap, cmap='hot', alpha=0.8)
        axes[1, 0].set_title('Model Attention Heatmap', fontweight='bold', color='white')
        axes[1, 0].axis('off')
        cbar = plt.colorbar(im, ax=axes[1, 0])
        cbar.set_label('Attention Intensity', fontweight='bold', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        cbar.ax.yaxis.label.set_color('white')
    else:
        axes[1, 0].text(0.5, 0.5, 'Attention Heatmap\nNot Available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes,
                       fontsize=12, fontweight='bold', color='white')
        axes[1, 0].set_title('Model Attention Heatmap', fontweight='bold', color='white')
        axes[1, 0].axis('off')
        axes[1, 0].set_facecolor('#0E1117')
    
    predicted_class = max(all_probs, key=all_probs.get)
    risk_level = CLINICAL_INFO[predicted_class]['severity']
    
    risk_levels = ['None', 'Low', 'Low to Moderate', 'Moderate', 'Moderate to High', 'High']
    risk_colors = ['#28a745', '#28a745', '#17a2b8', '#ffc107', '#fd7e14', '#dc3545']
    
    if risk_level in risk_levels:
        risk_index = risk_levels.index(risk_level)
    else:
        risk_index = 0
    
    axes[1, 1].pie([1], colors=[risk_colors[risk_index]], startangle=90, 
                  wedgeprops=dict(width=0.5))
    axes[1, 1].set_title(f'Risk Assessment\n{risk_level}', fontweight='bold', color='white')
    axes[1, 1].set_facecolor('#0E1117')
    
    plt.tight_layout()
    return fig

try:
    model = load_model()
    model_loaded = True
except:
    model_loaded = False

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #1a1a1a 0%, #2d2d2d 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
        border: 1px solid #333;
    }
    .metric-container {
        background: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
    }
    .sidebar .sidebar-content {
        background: #1a1a1a;
    }
    .stFileUploader {
        background: #1a1a1a;
        border: 2px dashed #333;
    }
    .stSelectbox > div > div {
        background-color: #1a1a1a;
        color: white;
    }
    .stTextInput > div > div > input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #333;
    }
    .stNumberInput > div > div > input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #333;
    }
    .stExpander {
        background-color: #1a1a1a;
        border: 1px solid #333;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    .stMarkdown {
        color: white;
    }
    .stInfo {
        background-color: #1a1a1a;
        border: 1px solid #17a2b8;
    }
    .stWarning {
        background-color: #1a1a1a;
        border: 1px solid #ffc107;
    }
    .stError {
        background-color: #1a1a1a;
        border: 1px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="color: white;">Kidney Health Analytics</h1>
    <h3 style="color: #cccccc;">Clinical Decision Support System</h3>
    <p style="color: #999999;">AI-Powered Kidney Anomaly Detection for Medical Imaging</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Patient Information")
st.sidebar.markdown("---")

patient_info = {}
patient_info['id'] = st.sidebar.text_input("Patient ID", placeholder="e.g., P12345")
patient_info['age'] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=None)
patient_info['gender'] = st.sidebar.selectbox("Gender", ["", "Male", "Female", "Other"])

st.sidebar.markdown("---")
st.sidebar.header("System Information")
st.sidebar.info(
    "This clinical decision support system uses deep learning to analyze kidney images "
    "and provide diagnostic assistance for radiologists and clinicians."
)

st.sidebar.markdown("---")
st.sidebar.header("Model Specifications")
st.sidebar.markdown(f"""
- Architecture: Convolutional Neural Network
- Input Size: {IMG_SIZE}x{IMG_SIZE} pixels
- Classes: {', '.join(CLASS_NAMES)}
- Modalities: Ultrasound, CT Scan
""")

if not model_loaded:
    st.error("Model could not be loaded. Please check the model file.")
    st.stop()

st.header("Image Upload and Analysis")
uploaded_file = st.file_uploader(
    "Upload Kidney Image (Ultrasound or CT Scan)", 
    type=["jpg", "jpeg", "png"],
    help="Upload a kidney ultrasound or CT scan image for analysis"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    modality = detect_image_modality(image)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, caption=f'Detected Modality: {modality}', use_container_width=True)
        
        st.markdown("**Image Information:**")
        st.write(f"- Dimensions: {image.size[0]} x {image.size[1]} pixels")
        st.write(f"- Modality: {modality}")
        st.write(f"- File Type: {uploaded_file.type}")
    
    with col2:
        st.subheader("Analysis Results")
        
        with st.spinner("Analyzing image..."):
            class_name, confidence, all_probs, heatmap, enhanced_img = predict_with_explanation(image, modality)
        
        color = CLINICAL_INFO[class_name]['color']
        st.markdown(f"""
        <div style="padding: 1.5rem; border-radius: 8px; border-left: 5px solid {color}; background-color: #1a1a1a; margin: 1rem 0; border: 1px solid #333;">
            <h3 style="color: {color}; margin: 0;">Primary Diagnosis: {class_name}</h3>
            <p style="margin: 0.5rem 0; font-size: 1.1rem; color: white;"><strong>Confidence Level:</strong> {confidence:.1f}%</p>
            <p style="margin: 0; font-size: 1rem; color: white;"><strong>Clinical Urgency:</strong> {CLINICAL_INFO[class_name]['urgency']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Detailed Probability Distribution:**")
        for condition, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
            st.write(f"- {condition}: {prob:.2f}%")
    
    st.header("Comprehensive Analysis Dashboard")
    
    fig = create_visualization_dashboard(all_probs, heatmap, enhanced_img)
    st.pyplot(fig)
    
    st.header("Clinical Report")
    
    report = generate_clinical_report(class_name, confidence, all_probs, modality, patient_info)
    
    with st.expander("View Complete Clinical Report", expanded=True):
        st.text(report)
    
    st.header("Export and Documentation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.download_button(
            label="Download Clinical Report",
            data=report,
            file_name=f"kidney_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    with col2:
        img_buffer = BytesIO()
        fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight', facecolor='#0E1117')
        img_buffer.seek(0)
        
        st.download_button(
            label="Download Analysis Charts",
            data=img_buffer.getvalue(),
            file_name=f"kidney_analysis_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True
        )
    
    with col3:
        summary_df = pd.DataFrame({
            'Condition': list(all_probs.keys()),
            'Probability (%)': [f"{prob:.2f}" for prob in all_probs.values()],
            'Predicted': [cond == class_name for cond in all_probs.keys()]
        })
        
        csv_buffer = BytesIO()
        summary_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        st.download_button(
            label="Download Analysis Data",
            data=csv_buffer.getvalue(),
            file_name=f"kidney_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

else:
    st.header("Welcome to the Kidney Health Analytics System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("System Capabilities")
        st.markdown("""
        - Automated Modality Detection: Distinguishes between ultrasound and CT scans
        - Advanced Preprocessing: Modality-specific image enhancement algorithms
        - Deep Learning Classification: CNN-based classification of kidney conditions
        - Model Interpretability: Gradient-based attention visualization
        - Clinical Integration: Professional diagnostic reports and documentation
        - Export Functionality: Multiple format support for clinical workflows
        """)
    
    with col2:
        st.subheader("Supported Conditions")
        for condition, info in CLINICAL_INFO.items():
            st.markdown(f"""
            **{condition}**  
            *{info['description']}*  
            Severity: {info['severity']}
            """)
    
    st.subheader("Usage Instructions")
    st.markdown("""
    1. Upload Image: Select a kidney ultrasound or CT scan image using the file uploader
    2. Automatic Analysis: The system will detect the imaging modality and process the image
    3. Review Results: Examine the diagnostic predictions and confidence levels
    4. Clinical Report: Generate a comprehensive clinical report for documentation
    5. Export Data: Download reports, visualizations, and analysis data as needed
    """)
    
    st.subheader("Important Medical Disclaimers")
    st.warning("""
    IMPORTANT: This system is intended as a diagnostic aid and should not replace clinical judgment. 
    All results should be interpreted by qualified medical professionals. For urgent findings, 
    immediate clinical attention is recommended.
    """)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666666; padding: 2rem 0;">
    <p><strong>Kidney Health Analytics System</strong></p>
    <p>12-Hour AI Hackathon | Clinical Decision Support System</p>
    <p>Built with Streamlit, TensorFlow, and OpenCV</p>
</div>
""", unsafe_allow_html=True)