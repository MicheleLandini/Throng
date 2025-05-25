import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import threading
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurazione WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
})

class WorkoutProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1  # Ridotto per mobile
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Thread-safe variables
        self.lock = threading.Lock()
        self.current_angle = 0
        self.rep_count = 0
        self.stage = None
        self.last_update = 0
        
    def calculate_angle(self, a, b, c):
        """Calcola l'angolo tra tre punti"""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians * 180.0 / np.pi)
            
            if angle > 180.0:
                angle = 360 - angle
                
            return angle
        except Exception as e:
            logger.error(f"Error calculating angle: {e}")
            return 0
            
    def detect_reps(self, angle, threshold_down=70, threshold_up=160):
        """Rileva le ripetizioni basandosi sull'angolo"""
        with self.lock:
            if angle > threshold_up and self.stage != "up":
                self.stage = "up"
            elif angle < threshold_down and self.stage == "up":
                self.stage = "down"
                self.rep_count += 1
                
            return self.rep_count, self.stage
    
    def get_landmarks_for_exercise(self, landmarks, exercise_type):
        """Estrae i landmark appropriati per ogni esercizio"""
        try:
            if exercise_type == "Push-up":
                shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                return shoulder, elbow, wrist
                
            elif exercise_type == "Squat":
                hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                       landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                return hip, knee, ankle
                
            elif exercise_type == "Bicep Curl":
                shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wrist = [landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                return shoulder, elbow, wrist
                
            else:  # Shoulder Press
                elbow = [landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                        landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                shoulder = [landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                           landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                hip = [landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                      landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                return elbow, shoulder, hip
                
        except (IndexError, AttributeError) as e:
            logger.error(f"Error extracting landmarks: {e}")
            return None, None, None
    
    def process_frame(self, frame, exercise_type, workout_active):
        """Processa un singolo frame"""
        if not workout_active:
            return frame
            
        try:
            # Converti BGR a RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Processa con MediaPipe
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                
                # Estrai coordinate per l'esercizio
                point1, point2, point3 = self.get_landmarks_for_exercise(landmarks, exercise_type)
                
                if all(p is not None for p in [point1, point2, point3]):
                    # Calcola angolo
                    angle = self.calculate_angle(point1, point2, point3)
                    self.current_angle = angle
                    
                    # Rileva ripetizioni
                    reps, stage = self.detect_reps(angle)
                    
                    # Disegna skeleton
                    self.mp_drawing.draw_landmarks(
                        rgb_frame, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                    
                    # Aggiungi testo overlay
                    cv2.putText(rgb_frame, f'Reps: {reps}', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(rgb_frame, f'Stage: {stage if stage else "N/A"}', (15, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(rgb_frame, f'Angle: {int(angle)}°', (15, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    self.last_update = time.time()
            
            # Converti di nuovo a BGR per output
            return cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame

def initialize_session_state():
    """Inizializza lo stato della sessione"""
    defaults = {
        'rep_count': 0,
        'series_count': 0,
        'stage': None,
        'reps_data': [],
        'workout_started': False,
        'countdown_started': False,
        'workout_paused': False,
        'saved_workouts': [],
        'pause_time': 0,
        'total_pause_time': 0,
        'start_time': 0,
        'processor': None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_workout_charts(reps_data):
    """Crea grafici per i dati dell'allenamento"""
    if len(reps_data) < 2:
        return None
        
    try:
        df = pd.DataFrame(reps_data)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Angolo nel tempo', 'Ripetizioni cumulative'),
            vertical_spacing=0.15
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['angle'], 
                mode='lines', 
                name='Angolo',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['reps'], 
                mode='lines+markers', 
                name='Reps',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400, 
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating charts: {e}")
        return None

def video_frame_callback(frame, exercise_type, workout_active, processor):
    """Callback per processare i frame video"""
    img = frame.to_ndarray(format="bgr24")
    
    if processor:
        img = processor.process_frame(img, exercise_type, workout_active)
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

def mostra_pagina():
    st.set_page_config(
        page_title="Allenamento Assistito",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏋️ Allenamento Assistito Avanzato")
    st.markdown("*Compatibile con dispositivi mobili tramite WebRTC*")
    
    # Inizializza stato
    initialize_session_state()
    
    # Crea processor se non esiste
    if st.session_state.processor is None:
        st.session_state.processor = WorkoutProcessor()
    
    # Sidebar configurazioni
    st.sidebar.header("⚙️ Impostazioni Allenamento")
    
    target_reps = st.sidebar.number_input(
        "Ripetizioni per serie", 
        min_value=1, max_value=100, value=10
    )
    target_series = st.sidebar.number_input(
        "Numero di serie", 
        min_value=1, max_value=20, value=3
    )
    countdown_time = st.sidebar.number_input(
        "Countdown iniziale (secondi)", 
        min_value=3, max_value=10, value=5
    )
    
    exercise_type = st.sidebar.selectbox(
        "Tipo di esercizio",
        ["Push-up", "Squat", "Bicep Curl", "Shoulder Press"]
    )
    
    workout_name = st.sidebar.text_input(
        "Nome allenamento", 
        value="Sessione " + time.strftime("%Y-%m-%d")
    )
    
    # Controlli principali
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        start_workout = st.button("🚀 Inizia", type="primary")
    with col2:
        pause_workout = st.button("⏸️ Pausa")
    with col3:
        reset_workout = st.button("🔄 Reset")
    with col4:
        save_workout = st.button(
            "💾 Salva", 
            disabled=len(st.session_state.reps_data) == 0
        )
    
    # Gestione controlli
    if start_workout:
        if not st.session_state.workout_paused:
            # Nuovo allenamento
            st.session_state.countdown_started = True
            st.session_state.workout_started = False
            st.session_state.rep_count = 0
            st.session_state.series_count = 0
            st.session_state.reps_data = []
            st.session_state.total_pause_time = 0
            st.session_state.start_time = time.time()
            st.session_state.processor.rep_count = 0
            st.session_state.processor.stage = None
        else:
            # Riprendi allenamento
            st.session_state.workout_started = True
            st.session_state.workout_paused = False
            st.session_state.total_pause_time += time.time() - st.session_state.pause_time
    
    if pause_workout and st.session_state.workout_started:
        st.session_state.workout_started = False
        st.session_state.workout_paused = True
        st.session_state.pause_time = time.time()
    
    if reset_workout:
        st.session_state.rep_count = 0
        st.session_state.series_count = 0
        st.session_state.reps_data = []
        st.session_state.workout_started = False
        st.session_state.countdown_started = False
        st.session_state.workout_paused = False
        st.session_state.total_pause_time = 0
        if st.session_state.processor:
            st.session_state.processor.rep_count = 0
            st.session_state.processor.stage = None
    
    if save_workout and len(st.session_state.reps_data) > 0:
        workout_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'name': workout_name,
            'exercise': exercise_type,
            'target_reps': target_reps,
            'target_series': target_series,
            'completed_series': st.session_state.series_count,
            'total_reps': st.session_state.reps_data[-1]['reps'] if st.session_state.reps_data else 0,
            'data': st.session_state.reps_data.copy(),
            'duration': st.session_state.reps_data[-1]['timestamp'] if st.session_state.reps_data else 0
        }
        st.session_state.saved_workouts.append(workout_data)
        st.success(f"✅ Allenamento '{workout_name}' salvato!")
    
    # Layout principale
    video_col, stats_col = st.columns([2, 1])
    
    with video_col:
        st.subheader("📹 Video Feed")
        
        # WebRTC Streamer
        ctx = webrtc_streamer(
            key="workout-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_frame_callback=lambda frame: video_frame_callback(
                frame, 
                exercise_type, 
                st.session_state.workout_started,
                st.session_state.processor
            ),
            media_stream_constraints={
                "video": {
                    "width": {"min": 640, "ideal": 1280, "max": 1920},
                    "height": {"min": 480, "ideal": 720, "max": 1080},
                    "frameRate": {"min": 15, "ideal": 30, "max": 60}
                },
                "audio": False
            },
            async_processing=True,
        )
        
        # Informazioni per dispositivi mobili
        st.info("📱 Su mobile: tocca 'START' per attivare la fotocamera")
    
    with stats_col:
        st.subheader("📊 Statistiche Live")
        
        # Countdown
        if st.session_state.countdown_started and not st.session_state.workout_started:
            elapsed = time.time() - st.session_state.start_time
            remaining = countdown_time - elapsed
            
            if remaining > 0:
                st.markdown(f"### ⏰ Inizia tra: {int(remaining) + 1} secondi")
            else:
                st.session_state.countdown_started = False
                st.session_state.workout_started = True
                st.session_state.start_time = time.time()
        
        # Statistiche in tempo reale
        if st.session_state.processor:
            current_reps = st.session_state.processor.rep_count
            current_stage = st.session_state.processor.stage
            current_angle = st.session_state.processor.current_angle
            
            # Aggiorna dati se necessario
            if st.session_state.workout_started and current_reps != st.session_state.rep_count:
                st.session_state.rep_count = current_reps
                
                # Aggiungi ai dati
                adjusted_timestamp = time.time() - st.session_state.start_time - st.session_state.total_pause_time
                st.session_state.reps_data.append({
                    "timestamp": adjusted_timestamp,
                    "angle": current_angle,
                    "reps": current_reps
                })
                
                # Controlla serie completate
                if current_reps >= target_reps:
                    st.session_state.series_count += 1
                    st.session_state.processor.rep_count = 0
                    st.session_state.rep_count = 0
                    
                    if st.session_state.series_count >= target_series:
                        st.session_state.workout_started = False
                        st.success("🎉 Allenamento completato!")
            
            # Mostra metriche
            st.metric("Ripetizioni correnti", current_reps, f"Target: {target_reps}")
            st.metric("Serie completate", st.session_state.series_count, f"Target: {target_series}")
            st.metric("Esercizio", exercise_type)
            
            if current_stage:
                st.write(f"**Fase:** {current_stage}")
            if current_angle > 0:
                st.write(f"**Angolo:** {int(current_angle)}°")
                
        if st.session_state.workout_paused:
            st.warning("⏸️ Allenamento in pausa")
    
    # Grafici
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.subheader("📈 Grafico Movimento")
        if len(st.session_state.reps_data) > 1:
            fig = create_workout_charts(st.session_state.reps_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
    
    with chart_col2:
        st.subheader("📋 Progresso Serie")
        if target_series > 0:
            progress = (st.session_state.series_count / target_series) * 100
            st.progress(min(progress / 100, 1.0))
            st.write(f"Progresso: {min(progress, 100):.1f}%")
    
    # Sezione allenamenti salvati
    if len(st.session_state.saved_workouts) > 0:
        st.header("💾 Allenamenti Salvati")
        
        workout_names = [
            f"{w['name']} - {w['timestamp']}" 
            for w in st.session_state.saved_workouts
        ]
        
        selected_idx = st.selectbox(
            "Seleziona allenamento:", 
            range(len(workout_names)),
            format_func=lambda x: workout_names[x]
        )
        
        if selected_idx is not None:
            workout = st.session_state.saved_workouts[selected_idx]
            
            # Metriche base
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Esercizio", workout['exercise'])
            with col2:
                st.metric("Serie", f"{workout['completed_series']}/{workout['target_series']}")
            with col3:
                st.metric("Ripetizioni", workout['total_reps'])
            with col4:
                st.metric("Durata", f"{workout['duration']:.1f}s")
            
            # Grafico dettagliato
            if len(workout['data']) > 0:
                fig_detailed = create_workout_charts(workout['data'])
                if fig_detailed:
                    st.plotly_chart(fig_detailed, use_container_width=True)

if __name__ == "__main__":
    mostra_pagina()
