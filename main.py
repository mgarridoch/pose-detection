import av
import os
import sys
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer
from aiortc.contrib.media import MediaRecorder # Necesario si MediaRecorder se usa internamente

# Ajustar el sys.path si los archivos utils.py, process_frame.py y thresholds.py
# no están en el mismo directorio principal. Si los copiaste al mismo directorio,
# estas líneas no son estrictamente necesarias pero no causan daño.
BASE_DIR = os.path.abspath(os.path.join(__file__, '../../'))
sys.path.append(BASE_DIR)

# Importar las funciones y clases necesarias de tus archivos auxiliares
from utils import get_mediapipe_pose
from process_frame import ProcessFrame
from thresholds import get_thresholds_beginner # Usaremos los umbrales de principiante para un ejemplo

# --- Modificaciones de texto y eliminación de opciones de modo ---
st.title('Detector de Poses Simple con IA') # Título principal en español
st.write('Con esta aplicación, puedes configurar fácilmente detecciones de poses a partir de tu webcam para diversas aplicaciones, como juegos de baile o análisis de movimientos.') # Texto introductorio en español

# --- Inicialización de los componentes de procesamiento de poses (parte de la app original de sentadillas) ---
# Usaremos los umbrales de principiante para el demo.
# La app de sentadillas ya define 'thresholds' en su lógica.
thresholds = get_thresholds_beginner() 

# Inicializar la clase de procesamiento de fotogramas.
# Esta clase contiene la lógica para procesar los fotogramas y dibujar sobre ellos.
live_process_frame = ProcessFrame(thresholds=thresholds, flip_frame=True)

# Inicializar el objeto de pose de MediaPipe.
# Esta función viene de tu archivo utils.py.
pose = get_mediapipe_pose()

# --- Callback de fotogramas de vídeo (como en tu 1_📷️_Live_Stream.py) ---
def video_frame_callback(frame: av.VideoFrame):
    # Convertir el fotograma AV a un array NumPy en formato RGB24
    frame_rgb = frame.to_ndarray(format="rgb24")
    
    # Procesar el fotograma usando tu lógica de ProcessFrame (análisis de sentadillas)
    # y el objeto de pose. La función .process() devuelve el fotograma anotado y una señal de sonido.
    annotated_frame_rgb, _ = live_process_frame.process(frame_rgb, pose)
    
    # Convertir el array NumPy anotado de nuevo a un av.VideoFrame en formato RGB24
    return av.VideoFrame.from_ndarray(annotated_frame_rgb, format="rgb24")

# --- Configuración de Streamlit WebRTC (como en tu 1_📷️_Live_Stream.py) ---
ctx = webrtc_streamer(
                        key="simple-pose-detector", # Una clave única para el streamer
                        video_frame_callback=video_frame_callback, # Tu función de procesamiento de fotogramas
                        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}, # Configuración estándar de WebRTC
                        media_stream_constraints={"video": {"width": {'min':480, 'ideal':480}}, "audio": False}, # Restricciones de medios (solo vídeo)
                        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=True), # Atributos HTML del vídeo (sin controles, silenciado)
                        # Eliminado out_recorder_factory y lógica de descarga para simplificar
                    )

# --- Texto final en la interfaz ---
st.write("Esta demostración muestra cómo es posible realizar un detector de poses en vivo usando tu cámara.") # Nuevo texto final