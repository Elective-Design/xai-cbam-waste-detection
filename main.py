import streamlit as st
import av
import torch
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from ultralytics import YOLO

# ====================================================================
# PAGE CONFIGURATION
# ====================================================================
st.set_page_config(
    page_title="Waste Detection System",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================================================================
# OPTIMIZED DARK THEME - Fast & Professional
# ====================================================================
st.markdown("""
    <style>
    /* Base styling - Minimal for performance */
    * {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
        max-width: 95% !important;
    }
    
    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Dark Background */
    .stApp {
        background: #050810 !important;
        color: #F9FAFB !important;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F9FAFB !important;
        font-weight: 600 !important;
    }
    
    h1 {
        font-size: 2.5rem !important;
        margin-bottom: 1.25rem !important;
        font-weight: 700 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid #F5C453 !important;
    }
    
    h2 {
        font-size: 1.875rem !important;
        margin-bottom: 1rem !important;
        padding-bottom: 0.75rem !important;
        border-bottom: 2px solid #F5C453 !important;
    }
    
    h3 {
        font-size: 1.375rem !important;
        margin-bottom: 0.75rem !important;
        color: #F5C453 !important;
    }
    
    h4 {
        color: #F9FAFB !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Cards - Simplified hover */
    .dark-card {
        background: #0F172A !important;
        border-radius: 16px !important;
        padding: 1.75rem !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
        border: 1px solid #1F2933 !important;
        margin-bottom: 1.25rem !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease !important;
    }
    
    .dark-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 24px rgba(245, 196, 83, 0.15) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
    }
    
    /* Metrics */
    div[data-testid="stMetric"] {
        background: #0F172A !important;
        border-radius: 12px !important;
        border: 1px solid #1F2933 !important;
        padding: 1.25rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    div[data-testid="stMetric"] label {
        color: #9CA3AF !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #F5C453 !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
    }
    
    /* Buttons - Black & Gold */
    .stButton > button {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2416 100%) !important;
        color: #F5C453 !important;
        border: 1px solid rgba(245, 196, 83, 0.3) !important;
        padding: 0.75rem 2rem !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2d2416 0%, #3d3118 100%) !important;
        border-color: #F5C453 !important;
        box-shadow: 0 4px 15px rgba(245, 196, 83, 0.3) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0B1120 !important;
        border-right: 1px solid #1F2933 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label {
        background: transparent !important;
        border-radius: 8px !important;
        padding: 0.75rem 1rem !important;
        border: 1px solid #1F2933 !important;
        color: #E5E7EB !important;
        transition: all 0.2s ease !important;
        margin: 0.25rem 0 !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label:hover {
        background: rgba(245, 196, 83, 0.05) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
    }
    
    [data-testid="stSidebar"] .stRadio > div > label[data-baseweb="radio"]:has(input:checked) {
        background: rgba(245, 196, 83, 0.1) !important;
        border-left: 3px solid #F5C453 !important;
        border-color: #F5C453 !important;
        color: #F9FAFB !important;
        font-weight: 600 !important;
    }
    
    /* Video */
    video {
        border-radius: 12px !important;
        border: 2px solid #1F2933 !important;
    }
    
    /* Text */
    p {
        color: #D1D5DB !important;
        line-height: 1.6 !important;
        font-size: 1rem !important;
    }
    
    /* Lists */
    ul, ol {
        color: #D1D5DB !important;
        line-height: 1.7 !important;
        padding-left: 1.5rem !important;
    }
    
    li {
        margin-bottom: 0.5rem !important;
    }
    
    /* Badge */
    .tech-badge {
        display: inline-block;
        padding: 0.4rem 0.875rem;
        background: rgba(34, 197, 94, 0.1);
        color: #22C55E;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    /* Status */
    .status-active {
        color: #22C55E;
        font-weight: 600;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0B1120;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #F5C453;
        border-radius: 4px;
    }
    
    /* Team card - Simplified */
    .team-card {
        text-align: center;
        padding: 1.5rem;
        background: #0F172A;
        border: 1px solid #1F2933;
        border-radius: 12px;
        transition: all 0.2s ease !important;
        height: 100%;
    }
    
    .team-card:hover {
        transform: translateY(-4px) !important;
        box-shadow: 0 8px 20px rgba(245, 196, 83, 0.15) !important;
        border-color: rgba(245, 196, 83, 0.3) !important;
    }
    
    .team-card h4 {
        color: #F9FAFB;
        margin: 0.5rem 0 0.25rem 0;
        font-size: 1.05rem;
        font-weight: 600;
    }
    
    .team-card p {
        color: #F5C453 !important;
        margin: 0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .team-emoji {
        font-size: 2rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    
    /* Dataframes */
    .stDataFrame {
        background: #0F172A !important;
    }
    
    /* Info/warning boxes */
    .stAlert {
        background: #0F172A !important;
        border: 1px solid #1F2933 !important;
        border-radius: 8px !important;
    }
    
    /* Selection boxes */
    .stSelectbox [data-baseweb="select"] {
        background: #0F172A !important;
        border-color: #1F2933 !important;
        color: #F9FAFB !important;
    }
    
    /* Number input */
    .stNumberInput input {
        background: #0F172A !important;
        border-color: #1F2933 !important;
        color: #F9FAFB !important;
    }
    </style>
""", unsafe_allow_html=True)

# ====================================================================
# DEVICE CONFIGURATION
# ====================================================================
@st.cache_resource
def get_device():
    """Get device with fallback"""
    if torch.backends.mps.is_available():
        return "mps", "🤖 Apple Silicon (MPS)"
    elif torch.cuda.is_available():
        return "cuda", "🚀 NVIDIA GPU"
    else:
        return "cpu", "💻 CPU"

DEVICE, DEVICE_LABEL = get_device()

# ====================================================================
# MODEL LOADING
# ====================================================================
# Using Untrained YOLOv10 (will likely download yolov10n.pt)
MODEL_CONFIG = {
    "YOLOv10 Nano (Untrained)": "yolov10n.pt",
    "YOLOv8 Nano (Fallback)": "yolov8n.pt"
}

@st.cache_resource(show_spinner=False)
def load_model(model_path):
    """Load YOLO model efficiently"""
    try:
        model = YOLO(model_path)
        model.to(DEVICE)
        # model.fuse() # Optional for inference speedup
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

MODEL = None

# ====================================================================
# VIDEO PROCESSOR
# ====================================================================
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.model = MODEL
        self.conf_threshold = 0.5
        
    def recv(self, frame):
        if self.model is None:
            return frame
            
        try:
            img = frame.to_ndarray(format="bgr24")
            # Inference
            results = self.model(img, 
                               conf=self.conf_threshold,
                               verbose=False,
                               max_det=20, 
                               imgsz=480)
            processed_img = results[0].plot()
            return av.VideoFrame.from_ndarray(processed_img, format="bgr24")
        except Exception as e:
            print(f"Processing error: {e}")
            return frame

# ====================================================================
# SIDEBAR NAVIGATION
# ====================================================================
def render_sidebar():
    with st.sidebar:
        # Header
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>♻️</div>
                <h3 style='margin: 0; color: #F5C453;'>Recycle Vision</h3>
                <p style='color: #9CA3AF; margin: 0.5rem 0 0 0; font-size: 0.9rem;'>
                    Waste Detection System
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        options = ["🏠 Overview", "🎥 Live Detection", "📊 Analytics"]
        
        if "page_selection" not in st.session_state:
            st.session_state.page_selection = options[0]

        selection = st.radio(
            "Navigation", 
            options, 
            index=options.index(st.session_state.page_selection),
            label_visibility="collapsed"
        )
        
        if selection != st.session_state.page_selection:
            st.session_state.page_selection = selection
            st.rerun()

        st.divider()

        # Model Selection
        st.markdown(f"""
            <div style='margin-bottom: 0.5rem;'>
                <h4 style='color: #F5C453; margin: 0 0 0.5rem 0; font-size: 1rem;'>⚙️ Model Config</h4>
            </div>
        """, unsafe_allow_html=True)

        selected_model_name = st.selectbox(
            "Select AI Model",
            list(MODEL_CONFIG.keys()),
            index=0,
            label_visibility="collapsed",
            key="model_selection"
        )
        
        global MODEL
        MODEL = load_model(MODEL_CONFIG[selected_model_name])
        
        st.divider()
        
        # System Status
        st.markdown(f"""
            <div style='background: #0F172A; border: 1px solid #1F2933; border-radius: 12px; padding: 1.25rem; margin: 1rem 0;'>
                <h4 style='color: #F5C453; margin: 0 0 1rem 0; font-size: 1rem;'>System Status</h4>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span style='color: #9CA3AF; font-size: 0.9rem;'>Hardware</span>
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>{DEVICE_LABEL}</span>
                </div>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span style='color: #9CA3AF; font-size: 0.9rem;'>Model</span>
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>{selected_model_name.split()[0]}</span>
                </div>
                <div style='display: flex; justify-content: space-between;'>
                    <span style='color: #9CA3AF; font-size: 0.9rem;'>Status</span>
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>● Active</span>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Footer / Team
        st.markdown("""
            <div style='text-align: center; padding: 1rem 0;'>
                <p style='color: #F5C453; font-weight: 600; margin-bottom: 0.25rem;'>Project Team</p>
                <p style='color: #9CA3AF; font-size: 0.85rem;'>Ramazan YILDIZ</p>
                <p style='color: #9CA3AF; font-size: 0.85rem;'>Samin Feyzi</p>
            </div>
        """, unsafe_allow_html=True)

# ====================================================================
# PROJECT OVERVIEW PAGE
# ====================================================================
def render_project_overview():
    st.markdown("<h1>Waste Detection System</h1>", unsafe_allow_html=True)
    
    # Intoduction
    st.markdown("""
        <div class='dark-card'>
            <h3>🎯 Project Goal</h3>
            <p>
                This project aims to develop a real-time waste detection system using advanced YOLOv10 architecture.
                The system detects various waste objects to facilitate better waste management and recycling processes.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Architecture", value="YOLOv10")
    with col2:
        st.metric(label="Detection Speed", value="Real-Time")
    with col3:
        st.metric(label="Interface", value="Streamlit")
    
    # Team Section
    st.markdown("<h2>Development Team</h2>", unsafe_allow_html=True)
    
    team_col1, team_col2 = st.columns(2)
    
    with team_col1:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👨‍💻</span>
                <h4>Ramazan YILDIZ</h4>
                <p style="color: #F5C453; font-weight: bold;">Developer & Researcher</p>
            </div>
        """, unsafe_allow_html=True)
    
    with team_col2:
        st.markdown("""
            <div class='team-card'>
                <span class='team-emoji'>👨‍💻</span>
                <h4>Samin Feyzi</h4>
                <p style="color: #F5C453; font-weight: bold;">Developer & Researcher</p>
            </div>
        """, unsafe_allow_html=True)
    
    # CTA
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🎥 Start Waste Detection", width='stretch'):
        st.session_state.page_selection = "🎥 Live Detection"
        st.rerun()

# ====================================================================
# LIVE DETECTION PAGE
# ====================================================================
def render_live_detection():
    st.markdown("<h1>Live Waste Detection</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
            <div class='dark-card' style='margin-bottom: 1rem;'>
                 <strong>Active Model:</strong> YOLOv10 (Untrained/Pretrained)
            </div>
        """, unsafe_allow_html=True)
    with col2:
         st.metric(label="Hardware", value=DEVICE_LABEL)

    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        model_key = f"yolo_detection_{st.session_state.get('model_selection', 'default')}"
        
        webrtc_ctx = webrtc_streamer(
            key=model_key,
            video_processor_factory=YOLOVideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            mode=WebRtcMode.SENDRECV,
             rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                ]
            }
        )
        
        if not webrtc_ctx.state.playing:
            st.info("👆 Click **START** above to activate camera")
            
    except Exception as e:
        st.error(f"Camera Error: {str(e)}")

# ====================================================================
# ANALYTICS DASHBOARD
# ====================================================================
def render_analytics_dashboard():
    st.markdown("<h1>Analytics Dashboard</h1>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='dark-card'>
            <p>
                Real-time statistics of detected waste items. 
                <em>(Placeholder data for demonstration)</em>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sample Data
    data = pd.DataFrame({
        'Waste Type': ['Plastic Bottle', 'Can', 'Paper', 'Glass', 'Cardboard'],
        'Count': [45, 32, 28, 15, 20],
        'Confidence': ['88%', '85%', '82%', '90%', '86%']
    })
    
    st.table(data)
    
    st.bar_chart(data.set_index('Waste Type')['Count'])

# ====================================================================
# MAIN APP ROUTING
# ====================================================================
def main():
    render_sidebar()
    
    page = st.session_state.page_selection
    
    if page == "🏠 Overview":
        render_project_overview()
    elif page == "🎥 Live Detection":
        render_live_detection()
    elif page == "📊 Analytics":
        render_analytics_dashboard()

if __name__ == "__main__":
    main()
