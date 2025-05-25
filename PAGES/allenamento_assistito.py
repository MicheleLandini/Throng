import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
from PIL import Image
import threading
import logging
import io
import base64

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WorkoutProcessor:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5,
            model_complexity=1
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Variables per tracking
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
    
    def process_image(self, image, exercise_type):
        """Processa una singola immagine"""
        try:
            # Converti BGR a RGB se necessario
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Processa con MediaPipe
            results = self.pose.process(rgb_image)
            
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
                    annotated_image = rgb_image.copy()
                    self.mp_drawing.draw_landmarks(
                        annotated_image, 
                        results.pose_landmarks, 
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                    )
                    
                    # Aggiungi testo overlay
                    h, w, _ = annotated_image.shape
                    cv2.putText(annotated_image, f'Reps: {reps}', (15, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_image, f'Stage: {stage if stage else "N/A"}', (15, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(annotated_image, f'Angle: {int(angle)}°', (15, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    return annotated_image, angle, reps, stage
            
            return rgb_image, 0, self.rep_count, self.stage
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return image, 0, self.rep_count, self.stage

    def process_video(self, video_path, exercise_type, progress_callback=None):
        """Processa un video completo e restituisce i frame processati e le statistiche"""
        cap = cv2.VideoCapture(video_path)
        processed_frames = []
        video_stats = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Reset contatori per il video
        self.rep_count = 0
        self.stage = None
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps
            
            # Processa il frame
            processed_frame, angle, reps, stage = self.process_image(frame, exercise_type)
            
            # Salva statistiche per questo frame
            video_stats.append({
                'frame': frame_count,
                'timestamp': timestamp,
                'angle': angle,
                'reps': reps,
                'stage': stage
            })
            
            # Salva ogni N frame per non occupare troppa memoria
            if frame_count % max(1, total_frames // 100) == 0:  # Massimo 100 frame salvati
                processed_frames.append(processed_frame)
            
            # Aggiorna progress bar se fornita
            if progress_callback:
                progress = frame_count / total_frames
                progress_callback(progress)
        
        cap.release()
        return processed_frames, video_stats

def initialize_session_state():
    """Inizializza lo stato della sessione"""
    defaults = {
        'rep_count': 0,
        'series_count': 0,
        'stage': None,
        'reps_data': [],
        'workout_started': False,
        'workout_paused': False,
        'saved_workouts': [],
        'pause_time': 0,
        'total_pause_time': 0,
        'start_time': 0,
        'processor': None,
        'camera_enabled': False,
        'current_angle': 0,
        'video_processed': False,
        'video_stats': [],
        'processed_frames': []
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

def create_video_stats_charts(video_stats):
    """Crea grafici dettagliati per le statistiche del video"""
    if len(video_stats) < 2:
        return None
    
    try:
        df = pd.DataFrame(video_stats)
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Angolo nel tempo', 'Ripetizioni cumulative', 'Velocità movimento'),
            vertical_spacing=0.1
        )
        
        # Grafico angolo
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
        
        # Grafico ripetizioni
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['reps'], 
                mode='lines+markers', 
                name='Reps',
                line=dict(color='green', width=2),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
        
        # Calcola velocità di movimento (differenza angolo)
        df['angle_velocity'] = df['angle'].diff().abs()
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'], 
                y=df['angle_velocity'], 
                mode='lines', 
                name='Velocità',
                line=dict(color='red', width=1)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            height=600, 
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        fig.update_xaxes(title_text="Tempo (s)", row=3, col=1)
        fig.update_yaxes(title_text="Gradi", row=1, col=1)
        fig.update_yaxes(title_text="Ripetizioni", row=2, col=1)
        fig.update_yaxes(title_text="Velocità", row=3, col=1)
        
        return fig
        
    except Exception as e:
        logger.error(f"Error creating video charts: {e}")
        return None

def show_webcam_component():
    """Mostra il componente webcam personalizzato"""
    webcam_html = """
    <div style="text-align: center; padding: 20px;">
        <video id="webcam" width="640" height="480" autoplay playsinline style="border: 2px solid #4CAF50; border-radius: 10px;"></video>
        <br><br>
        <button id="startBtn" onclick="startWebcam()" style="background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px;">
            📷 Avvia Webcam
        </button>
        <button id="stopBtn" onclick="stopWebcam()" style="background: #f44336; color: white; padding: 10px 20px; border: none; border-radius: 5px; margin: 5px;">
            ⏹️ Ferma Webcam
        </button>
        <canvas id="canvas" style="display: none;"></canvas>
        <div id="status" style="margin-top: 10px; font-weight: bold;"></div>
    </div>
    
    <script>
    let stream = null;
    let video = document.getElementById('webcam');
    let canvas = document.getElementById('canvas');
    let ctx = canvas.getContext('2d');
    
    async function startWebcam() {
        try {
            const constraints = {
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            document.getElementById('status').innerHTML = '✅ Webcam attiva';
            document.getElementById('status').style.color = 'green';
        } catch (err) {
            console.error('Errore accesso webcam:', err);
            document.getElementById('status').innerHTML = '❌ Errore accesso webcam: ' + err.message;
            document.getElementById('status').style.color = 'red';
        }
    }
    
    function stopWebcam() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            document.getElementById('status').innerHTML = '⏹️ Webcam fermata';
            document.getElementById('status').style.color = 'orange';
        }
    }
    
    // Auto-start per dispositivi mobili
    if (/Mobi|Android/i.test(navigator.userAgent)) {
        document.getElementById('status').innerHTML = '📱 Dispositivo mobile rilevato - Tocca "Avvia Webcam"';
    }
    </script>
    """
    
    st.components.v1.html(webcam_html, height=600)

def mostra_pagina():
    st.title("🏋️ Allenamento Assistito con AI")
    st.markdown("*Analizza i tuoi esercizi tramite video o webcam*")
    
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
    
    exercise_type = st.sidebar.selectbox(
        "Tipo di esercizio",
        ["Push-up", "Squat", "Bicep Curl", "Shoulder Press"]
    )
    
    workout_name = st.sidebar.text_input(
        "Nome allenamento", 
        value="Sessione " + time.strftime("%Y-%m-%d")
    )
    
    # Metodo di input
    input_method = st.radio(
        "Scegli il metodo di input:",
        ["🎥 Carica Video", "📱 Usa Webcam"]
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
            disabled=len(st.session_state.reps_data) == 0 and len(st.session_state.video_stats) == 0
        )
    
    # Gestione controlli
    if start_workout:
        if not st.session_state.workout_paused:
            st.session_state.workout_started = True
            st.session_state.rep_count = 0
            st.session_state.series_count = 0
            st.session_state.reps_data = []
            st.session_state.video_stats = []
            st.session_state.total_pause_time = 0
            st.session_state.start_time = time.time()
            st.session_state.processor.rep_count = 0
            st.session_state.processor.stage = None
            st.session_state.video_processed = False
        else:
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
        st.session_state.video_stats = []
        st.session_state.workout_started = False
        st.session_state.workout_paused = False
        st.session_state.total_pause_time = 0
        st.session_state.current_angle = 0
        st.session_state.video_processed = False
        if st.session_state.processor:
            st.session_state.processor.rep_count = 0
            st.session_state.processor.stage = None
            st.session_state.processor.current_angle = 0
    
    if save_workout and (len(st.session_state.reps_data) > 0 or len(st.session_state.video_stats) > 0):
        data_to_save = st.session_state.video_stats if len(st.session_state.video_stats) > 0 else st.session_state.reps_data
        workout_data = {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'name': workout_name,
            'exercise': exercise_type,
            'target_reps': target_reps,
            'target_series': target_series,
            'completed_series': st.session_state.series_count,
            'total_reps': data_to_save[-1]['reps'] if data_to_save else 0,
            'data': data_to_save.copy(),
            'duration': data_to_save[-1]['timestamp'] if data_to_save else 0
        }
        st.session_state.saved_workouts.append(workout_data)
        st.success(f"✅ Allenamento '{workout_name}' salvato!")
    
    # Layout principale
    if input_method == "🎥 Carica Video":
        st.subheader("🎥 Analisi Video")
        uploaded_video = st.file_uploader(
            "Carica un video del tuo allenamento", 
            type=['mp4', 'avi', 'mov', 'wmv']
        )
        
        if uploaded_video is not None:
            # Salva temporaneamente il video
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Mostra video originale
                st.video(uploaded_video)
                
                if st.session_state.workout_started and not st.session_state.video_processed:
                    st.info("🎬 Elaborazione video in corso...")
                    
                    # Barra di progresso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(progress):
                        progress_bar.progress(progress)
                        status_text.text(f"Elaborazione: {progress*100:.1f}%")
                    
                    # Processa il video
                    processed_frames, video_stats = st.session_state.processor.process_video(
                        tfile.name, exercise_type, update_progress
                    )
                    
                    st.session_state.video_stats = video_stats
                    st.session_state.processed_frames = processed_frames
                    st.session_state.video_processed = True
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Elaborazione completata!")
                    
                    # Calcola statistiche finali
                    if video_stats:
                        final_reps = video_stats[-1]['reps']
                        st.session_state.rep_count = final_reps
                        st.session_state.series_count = final_reps // target_reps
                        
                        st.success(f"🎉 Video analizzato! Ripetizioni rilevate: {final_reps}")
            
            with col2:
                st.subheader("📊 Statistiche Video")
                
                if st.session_state.video_processed and st.session_state.video_stats:
                    stats = st.session_state.video_stats
                    final_stats = stats[-1]
                    
                    # Metriche principali
                    st.metric("Ripetizioni totali", final_stats['reps'])
                    st.metric("Serie stimate", final_stats['reps'] // target_reps)
                    st.metric("Durata video", f"{final_stats['timestamp']:.1f}s")
                    
                    # Statistiche dettagliate
                    angles = [s['angle'] for s in stats if s['angle'] > 0]
                    if angles:
                        st.metric("Angolo medio", f"{np.mean(angles):.1f}°")
                        st.metric("Angolo min/max", f"{min(angles):.0f}°/{max(angles):.0f}°")
                    
                    # Tempo medio per ripetizione
                    if final_stats['reps'] > 0:
                        avg_time_per_rep = final_stats['timestamp'] / final_stats['reps']
                        st.metric("Tempo/ripetizione", f"{avg_time_per_rep:.1f}s")
                else:
                    st.info("Premi 'Inizia' per analizzare il video")
            
            # Grafici dettagliati del video
            if st.session_state.video_processed and st.session_state.video_stats:
                st.subheader("📈 Analisi Dettagliata Movimento")
                
                fig = create_video_stats_charts(st.session_state.video_stats)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Tabella con dati grezzi (sample)
                with st.expander("📋 Dati Dettagliati (Prime 20 righe)"):
                    df_sample = pd.DataFrame(st.session_state.video_stats[:20])
                    st.dataframe(df_sample)
    
    else:  # Webcam
        st.subheader("📱 Webcam Live")
        st.info("💡 **Funziona su PC e Mobile:** La webcam si adatta automaticamente al tuo dispositivo")
        
        # Mostra componente webcam
        show_webcam_component()
        
        # Statistiche live per webcam
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Statistiche Live")
            current_reps = st.session_state.processor.rep_count
            current_stage = st.session_state.processor.stage
            current_angle = st.session_state.processor.current_angle
            
            st.metric("Ripetizioni correnti", current_reps, f"Target: {target_reps}")
            st.metric("Serie completate", st.session_state.series_count, f"Target: {target_series}")
            
            if current_stage:
                st.write(f"**Fase:** {current_stage}")
            if current_angle > 0:
                st.write(f"**Angolo:** {int(current_angle)}°")
        
        with col2:
            st.subheader("📈 Grafico Live")
            if len(st.session_state.reps_data) > 1:
                fig = create_workout_charts(st.session_state.reps_data)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
        
        # Stato allenamento
        if st.session_state.workout_paused:
            st.warning("⏸️ Allenamento in pausa")
        elif st.session_state.workout_started:
            st.success("🏃 Allenamento attivo")
    
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
                if 'frame' in workout['data'][0]:  # Video data
                    fig_detailed = create_video_stats_charts(workout['data'])
                else:  # Live data
                    fig_detailed = create_workout_charts(workout['data'])
                
                if fig_detailed:
                    st.plotly_chart(fig_detailed, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Allenamento Assistito",
        page_icon="🏋️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    mostra_pagina()

if __name__ == "__main__":
    main()
