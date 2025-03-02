from pathlib import Path
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam

# setting page layout
st.set_page_config(
    page_title="DEMO APP WITH YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("Demo app theo dõi số lượng học sinh")

# sidebar
st.sidebar.header("Cấu hình model")

# model options
task_type = st.sidebar.selectbox(
    "Tác vụ",
    ["Detection"]
)

model_type = None
if task_type == "Detection":
    model_type = st.sidebar.selectbox(
        "Mô hình",
        config.DETECTION_MODEL_LIST
    )
else:
    st.error("Currently only 'Detection' function is implemented")

confidence = float(st.sidebar.slider(
    "Chọn độ tin cậy (confident threshold)", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")

# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Cấu hình Hình ảnh/Video")
source_selectbox = st.sidebar.selectbox(
    "Chọn định dạng",
    config.SOURCES_LIST
)

source_img = None
if source_selectbox == config.SOURCES_LIST[0]: # Hình ảnh
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")