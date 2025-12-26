import streamlit as st
import av
import torch
import pandas as pd
import glob
import os
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
        width: 100% !important; 
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
# MODEL LOADING - DYNAMIC
# ====================================================================
def get_all_models():
    """Scan directory recursively for .pt files"""
    # Search in current directory and subdirectories (up to depth 3)
    models = []
    
    # search current dir
    models.extend(glob.glob("*.pt"))
    
    # search known subdirs like YOLO_Runs
    models.extend(glob.glob("**/*.pt", recursive=True))
    
    # Filter out duplicates and non-model files if needed
    models = list(set(models))
    
    if not models:
        return ["yolov10n.pt"] # Fallback
    
    # Sort to make 'best.pt' first if it exists (prioritize 'best.pt' in filename)
    models.sort(key=lambda x: (not "best.pt" in x, x))
    
    return models

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
        # These will be updated from main thread
        self.conf_threshold = 0.25 
        self.iou_threshold = 0.45
        
    def recv(self, frame):
        if self.model is None:
            return frame
            
        try:
            img = frame.to_ndarray(format="bgr24")
            
            # 1. Update params dynamically from session state if available
            # Note: Webrtc runs in a separate thread, so we access session state safely
            # or rely on the values being pushed to the processor instance.
            
            # Inference
            results = self.model(img, 
                               conf=self.conf_threshold,
                               iou=self.iou_threshold,
                               verbose=False,
                               max_det=20, 
                               imgsz=640) # Changed to 640 to match training!
            
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
                <span style='background:rgba(34, 197, 94, 0.1); color:#22C55E; padding:2px 8px; border-radius:12px; font-size:0.8rem; border:1px solid rgba(34, 197, 94, 0.3);'>
                    YOLOv11 Powered
                </span>
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

        available_models = get_all_models()
        selected_model_name = st.selectbox(
            "Select AI Model",
            available_models,
            index=0,
            key="model_selection",
            help="Place your .pt files in the project folder to see them here."
        )
        
        # Load Global Model
        global MODEL
        MODEL = load_model(selected_model_name)
        
        
        # INFERENCE SETTINGS
        with st.expander("🛠️ Inference Settings", expanded=True):
            conf_val = st.slider("Confidence", min_value=0.0, max_value=1.0, value=0.25, step=0.05)
            iou_val = st.slider("IoU Threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.05)
            
            # Store interpretable values in session state for the processor to pick up (if we were passing it)
            # Since Processor is instantiated inside render_live_detection, we will pass these values there.
            st.session_state['conf_threshold'] = conf_val
            st.session_state['iou_threshold'] = iou_val

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
                    <span style='color: #22C55E; font-weight: 600; font-size: 0.9rem;'>{selected_model_name}</span>
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
                This project aims to develop a real-time waste detection system using advanced 
                <span style="color:#F5C453; font-weight:bold;">YOLOv11</span> architecture.
                The system detects various waste objects to facilitate better waste management and recycling processes.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Model Architecture", value="YOLOv11 Medium")
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
        current_model = st.session_state.get('model_selection', 'Unknown')
        st.markdown(f"""
            <div class='dark-card' style='margin-bottom: 1rem;'>
                 <strong>Active Model:</strong> {current_model}
            </div>
        """, unsafe_allow_html=True)
    with col2:
         st.metric(label="Hardware", value=DEVICE_LABEL)

    st.markdown("<br>", unsafe_allow_html=True)
    
    try:
        model_key = f"yolo_detection_{st.session_state.get('model_selection', 'default')}"
        
        # Factory needed to pass parameters dynamically? 
        # Streamlit Webrtc is tricky with dynamic params.
        # We'll use a class-based approach where we update the instance.
        
        def video_frame_callback(frame):
            if MODEL is None:
                return frame
            
            try:
                img = frame.to_ndarray(format="bgr24")
                
                # Get dynamic params from session state
                # Note: This callback runs in a different thread!
                # st.session_state might be thread-safe reading? 
                # Yes, in recent Streamlit versions usually okay for reading.
                # If not, use defaults.
                
                conf = 0.25
                iou = 0.45
                
                # We can't easily access st.session_state here in some contexts
                # But let's try standard way or defaults
                
                # Hacky fix: Since we can't reliably pass args to this callback easily 
                # without closure or class, we use the global variable 'MODEL' 
                # but 'conf' we have to be careful.
                # For now let's use the ones set in global scope or defaults.
                
                # Actually, class based processor is better for state.
                pass 
                
            except:
                pass

        # We stick to class based processor defined above
        # But we need to inject the params.
        
        ctx = webrtc_streamer(
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
        
        # DYNAMIC UPDATE OF PROCESSOR
        if ctx.video_processor:
             ctx.video_processor.conf_threshold = st.session_state.get('conf_threshold', 0.25)
             ctx.video_processor.iou_threshold = st.session_state.get('iou_threshold', 0.45)
             ctx.video_processor.model = MODEL # Ensure latest model is used
        
        if not ctx.state.playing:
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
    
    # Sample Data - UPDATED for 8 Classes
    data = pd.DataFrame({
        'Waste Type': ['Plastic', 'Paper', 'Glass', 'Metal', 'Cardboard', 'Organic', 'Bottle', 'Trash'],
        'Count': [45, 32, 12, 15, 20, 8, 10, 5],
        'Confidence': ['88%', '85%', '90%', '82%', '86%', '75%', '92%', '60%']
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
