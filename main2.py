import cv2
import mediapipe as mp # Still needed for mp.solutions.drawing_utils
import numpy as np 
import streamlit as st
import av 

# Import the necessary components from your copied files
from utils import get_mediapipe_pose # For initializing the MediaPipe Pose object
from process_frame import ProcessFrame # For processing each video frame
from thresholds import get_thresholds_beginner # Or get_thresholds_pro, for squat thresholds

# Import Streamlit WebRTC components
from streamlit_webrtc import webrtc_streamer, VideoHTMLAttributes, WebRtcMode


# --- Global MediaPipe Pose and ProcessFrame setup ---
# Initialize the MediaPipe Pose object once
pose = get_mediapipe_pose()

# Initialize the ProcessFrame object with squat thresholds
# You can choose 'Beginner' or 'Pro' thresholds
thresholds = get_thresholds_beginner() 
live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)


# --- Video Frame Callback Function ---
# This function will be called for each frame by streamlit-webrtc
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    # Convert incoming av.VideoFrame to a NumPy array (RGB format, as process_frame expects)
    # The process_frame.py expects "rgb24", as seen in its video_frame_callback.
    # So, we convert from av.VideoFrame to ndarray with "rgb24" format.
    img_rgb = frame.to_ndarray(format="rgb24") 
    
    # Process the frame using the ProcessFrame instance and the pose object
    # The process method returns the annotated frame and a sound signal (which we ignore here)
    annotated_frame_rgb, _ = live_process_frame.process(img_rgb, pose) 
    
    # Convert the annotated NumPy array back to an av.VideoFrame object (RGB format, as process_frame returns RGB)
    return av.VideoFrame.from_ndarray(annotated_frame_rgb, format="rgb24")


# --- Streamlit App Layout ---
st.title("AI Fitness Trainer: Squats Analysis (Live Demo)") # Changed title to reflect the content
st.write("This app uses MediaPipe to analyze squat movements from your webcam.")

webrtc_streamer(
    key="squats-pose-analysis", # Key from the original working app
    mode=WebRtcMode.SENDRECV, 
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, 
    video_frame_callback=video_frame_callback, # Pass our processing function here
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True), 
)

st.write("Perform squats in front of the camera to see the analysis!")
st.write("Note: This demo is configured for squat analysis based on the provided helper files.")