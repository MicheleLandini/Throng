import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math
import tempfile
import os
from pathlib import Path


def mostra_pagina():
    # CSS personalizzato per UI moderna
    st.markdown("""
    <style>
        /* Tema generale */
        .main {
            padding-top: 1rem;
        }
        
        /* Header personalizzato */
        .custom-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Card containers */
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        
        /* Pulsanti personalizzati */
        .stButton > button {
            border-radius: 25px;
            border: none;
            padding: 0.5rem 2rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
        }
        
        /* Sidebar personalizzata */
        .sidebar .sidebar-content {
            background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
        }
        
        /* Progress bar personalizzata */
        .stProgress > div > div > div > div {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        }
        
        /* Alert personalizzati */
        .alert-success {
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-info {
            background-color: #cce7ff;
            border: 1px solid #99d6ff;
            color: #0056b3;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .alert-warning {
            background-color: #fff3cd;
            border: 1px solid #ffecb3;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        /* Mobile responsiveness */
        @media (max-width: 768px) {
            .custom-header {
                padding: 1rem 0.5rem;
            }
            
            .metric-card {
                margin-bottom: 0.5rem;
                padding: 1rem;
            }
            
            .stButton > button {
                width: 100%;
                margin-bottom: 0.5rem;
            }
        }
        
        /* Video container */
        .video-container {
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        
        /* Stats container */
        .stats-container {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # Header moderno
    st.markdown("""
    <div class="custom-header">
        <h1>💪 FitTracker Pro</h1>
        <p>Il tuo assistente intelligente per l'allenamento con AI</p>
    </div>
    """, unsafe_allow_html=True)

    # Mediapipe setup
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Funzione per calcolare l'angolo tra tre punti
    def calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360-angle
            
        return angle

    # Funzione per rilevare le ripetizioni
    def detect_reps(angle, rep_count=0, stage=None, threshold_down=70, threshold_up=160):
        if angle > threshold_up:
            stage = "up"
        if angle < threshold_down and stage == "up":
            stage = "down"
            rep_count += 1
        return rep_count, stage


    def process_video_file(uploaded_video, exercise_type, target_reps, target_series):
        """
        Processa un file video caricato per rilevare ripetizioni di esercizi
        """
        import tempfile
        import time
        import gc
        
        # Inizializza le variabili locali
        rep_count = 0  # ← FIX: Inizializza rep_count
        reps_data = []
        stage = None
        
        # Crea file temporaneo
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_path = tmp_file.name
            tmp_file.write(uploaded_video.read())
        
        cap = None  # Inizializza cap per gestire l'eccezione
        
        try:
            # Carica il video
            cap = cv2.VideoCapture(tmp_path)
            
            if not cap.isOpened():
                st.error("❌ Impossibile aprire il video")
                return None
            
            # Ottieni informazioni video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if fps == 0:
                fps = 30  # Default fallback
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_preview = st.empty()
            
            frame_count = 0
            last_update_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                timestamp = frame_count / fps
                
                # Processa frame per rilevamento pose
                results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                
                if results.pose_landmarks:
                    # Estrai landmarks
                    landmarks = []
                    for lm in results.pose_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    # Calcola angoli per l'esercizio specifico
                    if exercise_type == "Push-up":
                        # Logica per push-up
                        left_elbow_angle = calculate_angle(
                            [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
                            [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
                            [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        )
                        
                        # Stato push-up
                        if left_elbow_angle > 160:
                            stage = "up"
                        if left_elbow_angle < 90 and stage == 'up':
                            stage = "down"
                            rep_count += 1
                            reps_data.append({
                                'reps': rep_count,
                                'timestamp': timestamp,
                                'angle': left_elbow_angle
                            })
                    
                    elif exercise_type == "Squat":
                        # Logica per squat
                        left_knee_angle = calculate_angle(
                            [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y],
                            [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
                            [results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        )
                        
                        # Stato squat
                        if left_knee_angle > 160:
                            stage = "up"
                        if left_knee_angle < 90 and stage == 'up':
                            stage = "down"
                            rep_count += 1
                            reps_data.append({
                                'reps': rep_count,
                                'timestamp': timestamp,
                                'angle': left_knee_angle
                            })
                
                # Aggiorna progress ogni secondo
                current_time = time.time()
                if current_time - last_update_time >= 1.0:
                    progress = frame_count / total_frames
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processando frame {frame_count}/{total_frames} - Ripetizioni: {rep_count}")
                    
                    # Mostra frame preview ridimensionato
                    preview_frame = cv2.resize(frame, (320, 240))
                    frame_preview.image(preview_frame, channels="BGR", width=320)
                    
                    last_update_time = current_time
            
            # Completa il processing
            progress_bar.progress(1.0)
            status_text.text(f"✅ Completato! Ripetizioni totali: {rep_count}")
            
        except Exception as e:
            st.error(f"❌ Errore durante il processing del video: {str(e)}")
            return None
            
        finally:
            # IMPORTANTE: Chiudi tutti i riferimenti al video PRIMA di eliminare il file
            if cap is not None:
                cap.release()
            
            # Forza la garbage collection per liberare risorse
            gc.collect()
            
            # Pulisci UI
            if 'progress_bar' in locals():
                progress_bar.empty()
            if 'status_text' in locals():
                status_text.empty()
            if 'frame_preview' in locals():
                frame_preview.empty()
            
            # Prova a rimuovere il file temporaneo con retry
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    break  # Se riesce, esci dal loop
                except PermissionError:
                    if attempt < max_retries - 1:
                        time.sleep(0.5)  # Aspetta mezzo secondo e riprova
                        gc.collect()  # Forza garbage collection
                    else:
                        # Se fallisce anche l'ultimo tentativo, logga l'errore ma non bloccare
                        st.warning(f"⚠️ Impossibile rimuovere il file temporaneo {tmp_path}. "
                                f"Verrà rimosso automaticamente al riavvio del sistema.")
                except Exception as e:
                    st.warning(f"⚠️ Errore nella rimozione del file temporaneo: {str(e)}")
                    break
        
        # Restituisci i risultati
        return {
            'reps_data': reps_data,
            'total_reps': rep_count,
            'duration': timestamp if 'timestamp' in locals() else 0,
            'total_frames': total_frames if 'total_frames' in locals() else 0,
            'fps': fps if 'fps' in locals() else 30
        }

    # Inizializza variabili di sessione
    session_vars = {
        'rep_count': 0,
        'stage': None,
        'reps_data': [],
        'workout_started': False,
        'countdown_started': False,
        'series_count': 0,
        'chart_counter': 0,
        'workout_paused': False,
        'saved_workouts': [],
        'pause_time': 0,
        'total_pause_time': 0,
        'workout_mode': 'webcam'  # 'webcam' o 'video'
    }

    for key, default_value in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Sidebar migliorata con design moderno
    with st.sidebar:
        st.markdown("### ⚙️ Configurazione Allenamento")
        
        # Selezione modalità
        st.markdown("#### 📹 Modalità di Allenamento")
        workout_mode = st.radio(
            "Scegli modalità:",
            ["📷 Webcam Live", "🎥 Carica Video"],
            help="La webcam potrebbe non funzionare su dispositivi mobili. Usa 'Carica Video' per analizzare video registrati."
        )
        
        st.session_state.workout_mode = 'webcam' if workout_mode == "📷 Webcam Live" else 'video'
        
        st.markdown("---")
        
        # Parametri allenamento
        st.markdown("#### 🎯 Parametri")
        target_reps = st.number_input("🔄 Ripetizioni per serie", min_value=1, max_value=100, value=10)
        target_series = st.number_input("📊 Numero di serie", min_value=1, max_value=20, value=3)
        
        if st.session_state.workout_mode == 'webcam':
            countdown_time = st.number_input("⏱️ Countdown iniziale (sec)", min_value=3, max_value=10, value=5)
        
        # Selezione esercizio con emoji
        exercise_options = {
            "💪 Push-up": "Push-up",
            "🏋️ Squat": "Squat", 
            "💪 Bicep Curl": "Bicep Curl",
            "🤸 Shoulder Press": "Shoulder Press"
        }
        
        exercise_display = st.selectbox("🏃 Tipo di esercizio", list(exercise_options.keys()))
        exercise_type = exercise_options[exercise_display]
        
        st.markdown("---")
        
        # Informazioni sessione
        st.markdown("#### 📝 Informazioni")
        workout_name = st.text_input("📅 Nome allenamento", value="Sessione " + time.strftime("%Y-%m-%d"))
        
        # Upload video se modalità video selezionata
        uploaded_video = None
        if st.session_state.workout_mode == 'video':
            st.markdown("#### 📤 Carica Video")
            uploaded_video = st.file_uploader(
                "Scegli un file video",
                type=['mp4', 'avi', 'mov', 'mkv'],
                help="Formati supportati: MP4, AVI, MOV, MKV"
            )
            
            if uploaded_video is not None:
                st.success(f"✅ Video caricato: {uploaded_video.name}")
                st.info(f"📊 Dimensione: {uploaded_video.size / (1024*1024):.1f} MB")

    # Layout principale con design moderno
    st.markdown("### 🎮 Controlli Allenamento")

    # Controlli con layout responsive
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.session_state.workout_mode == 'video' and uploaded_video is not None:
            start_button_text = "🎬 Analizza Video"
        else:
            start_button_text = "🚀 Inizia"
        
        if st.button(start_button_text, type="primary", use_container_width=True):
            if st.session_state.workout_mode == 'video' and uploaded_video is not None:
                # Processa video caricato
                with st.spinner("🔄 Analizzando video..."):
                    result = process_video_file(uploaded_video, exercise_type, target_reps, target_series)
                
                if result:
                    st.session_state.reps_data = result['reps_data']
                    st.session_state.rep_count = result['total_reps']
                    
                    # Salva automaticamente i risultati
                    workout_data = {
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'name': workout_name,
                        'exercise': exercise_type,
                        'target_reps': target_reps,
                        'target_series': target_series,
                        'completed_series': result['total_reps'] // target_reps,
                        'total_reps': result['total_reps'],
                        'data': result['reps_data'],
                        'duration': result['duration'],
                        'source': 'video_upload'
                    }
                    st.session_state.saved_workouts.append(workout_data)
                    
                    st.success(f"🎉 Video analizzato con successo! Rilevate {result['total_reps']} ripetizioni.")
            else:
                # Modalità webcam
                if not st.session_state.workout_paused:
                    st.session_state.countdown_started = True
                    st.session_state.workout_started = False
                    st.session_state.rep_count = 0
                    st.session_state.series_count = 0
                    st.session_state.reps_data = []
                    st.session_state.total_pause_time = 0
                else:
                    st.session_state.workout_started = True
                    st.session_state.workout_paused = False
                    st.session_state.total_pause_time += time.time() - st.session_state.pause_time

    with col2:
        if st.button("⏸️ Pausa", use_container_width=True):
            if st.session_state.workout_started:
                st.session_state.workout_started = False
                st.session_state.workout_paused = True
                st.session_state.pause_time = time.time()

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            for key in ['rep_count', 'series_count', 'reps_data', 'workout_started', 
                    'countdown_started', 'workout_paused', 'total_pause_time']:
                if key == 'reps_data':
                    st.session_state[key] = []
                else:
                    st.session_state[key] = 0 if 'count' in key or 'time' in key else False

    with col4:
        if st.button("💾 Salva", disabled=len(st.session_state.reps_data) == 0, use_container_width=True):
            if len(st.session_state.reps_data) > 0:
                workout_data = {
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'name': workout_name,
                    'exercise': exercise_type,
                    'target_reps': target_reps,
                    'target_series': target_series,
                    'completed_series': st.session_state.series_count,
                    'total_reps': st.session_state.reps_data[-1]['reps'] if st.session_state.reps_data else 0,
                    'data': st.session_state.reps_data.copy(),
                    'duration': st.session_state.reps_data[-1]['timestamp'] if st.session_state.reps_data else 0,
                    'source': 'webcam_live'
                }
                st.session_state.saved_workouts.append(workout_data)
                
                st.markdown("""
                <div class="alert-success">
                    ✅ Allenamento salvato con successo!
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")

    # Layout principale migliorato
    if st.session_state.workout_mode == 'webcam':
        # Layout per webcam
        video_col, stats_col = st.columns([2, 1])
        
        with video_col:
            st.markdown("### 📹 Video Live")
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            frame_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
        with stats_col:
            st.markdown("### 📊 Statistiche Live")
            st.markdown('<div class="stats-container">', unsafe_allow_html=True)
            stats_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            countdown_placeholder = st.empty()
            
        # Webcam processing (codice originale adattato)
        rep_count = st.session_state.rep_count
        stage = st.session_state.stage
        
        run = st.session_state.workout_started or st.session_state.countdown_started
        
        if run:
            cap = cv2.VideoCapture(0)
            start_time = time.time()
            countdown_start = time.time()
            
            # Verifica se la webcam è disponibile
            if not cap.isOpened():
                st.error("❌ Impossibile accedere alla webcam. Prova a usare la modalità 'Carica Video'.")
                st.session_state.workout_started = False
                st.session_state.countdown_started = False
            else:
                while cap.isOpened() and (st.session_state.workout_started or st.session_state.countdown_started):
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Countdown logic
                    if st.session_state.countdown_started and not st.session_state.workout_started:
                        elapsed = time.time() - countdown_start
                        remaining = countdown_time - elapsed
                        
                        if remaining > 0:
                            cv2.putText(frame_rgb, f"Inizia tra: {int(remaining) + 1}", 
                                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                            countdown_placeholder.markdown(f"""
                            <div class="alert-info">
                                ⏰ Inizia tra: <strong>{int(remaining) + 1}</strong> secondi
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.session_state.countdown_started = False
                            st.session_state.workout_started = True
                            start_time = time.time()
                            countdown_placeholder.empty()

                    # Processing del pose solo durante l'allenamento
                    if st.session_state.workout_started:
                        results = pose.process(frame_rgb)
                        
                        if results.pose_landmarks:
                            landmarks = results.pose_landmarks.landmark
                            
                            # Calcola angolo in base all'esercizio (codice originale)
                            if exercise_type == "Push-up":
                                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                                angle = calculate_angle(shoulder, elbow, wrist)
                                
                            elif exercise_type == "Squat":
                                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                                angle = calculate_angle(hip, knee, ankle)
                                
                            elif exercise_type == "Bicep Curl":
                                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                                wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                                angle = calculate_angle(shoulder, elbow, wrist)
                                
                            else:  # Shoulder Press
                                elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                                shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                                hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                                angle = calculate_angle(elbow, shoulder, hip)

                            # Rilevamento ripetizioni
                            rep_count = st.session_state.get("rep_count", 0)
                            stage = st.session_state.get("stage", None)
                            rep_count, stage = detect_reps(angle, rep_count, stage)

                            st.session_state.rep_count = rep_count
                            st.session_state.stage = stage

                            
                            # Salva dati
                            adjusted_timestamp = time.time() - start_time - st.session_state.total_pause_time
                            st.session_state.reps_data.append({
                                "timestamp": adjusted_timestamp,
                                "angle": angle,
                                "reps": rep_count
                            })
                            
                            # Disegna pose landmarks
                            mp_drawing.draw_landmarks(
                                frame_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                            )
                            
                            # Informazioni sul frame
                            cv2.putText(frame_rgb, f'Reps: {rep_count}', (15, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame_rgb, f'Stage: {stage}', (15, 70), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame_rgb, f'Angle: {int(angle)}°', (15, 110), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                            # Controllo serie completate
                            if rep_count >= target_reps:
                                st.session_state.series_count += 1
                                st.session_state.rep_count = 0
                                rep_count = 0
                                
                                if st.session_state.series_count >= target_series:
                                    st.session_state.workout_started = False
                                    st.markdown("""
                                    <div class="alert-success">
                                        🎉 Allenamento completato con successo!
                                    </div>
                                    """, unsafe_allow_html=True)

                    # Mostra frame
                    frame_resized = cv2.resize(frame_rgb, (640, 480))
                    frame_placeholder.image(frame_resized, channels="RGB")
                    
                    # Aggiorna statistiche
                    with stats_placeholder.container():
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("🔄 Ripetizioni", st.session_state.rep_count, f"Target: {target_reps}")
                            st.metric("📊 Serie", st.session_state.series_count, f"Target: {target_series}")
                        with col2:
                            st.metric("🏃 Esercizio", exercise_type)
                            if st.session_state.stage:
                                st.write(f"**Fase:** {st.session_state.stage.upper()}")
                        
                        if st.session_state.workout_paused:
                            st.markdown("""
                            <div class="alert-warning">
                                ⏸️ Allenamento in pausa - Premi 'Inizia Allenamento' per riprendere
                            </div>
                            """, unsafe_allow_html=True)
                    
                    time.sleep(0.1)

            cap.release()
        
        else:
            if st.session_state.workout_paused:
                st.markdown("""
                <div class="alert-warning">
                    ⏸️ Allenamento in pausa - Premi 'Inizia Allenamento' per riprendere
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="alert-info">
                    👆 Clicca 'Inizia Allenamento' per cominciare o passa alla modalità 'Carica Video' per analizzare video registrati
                </div>
                """, unsafe_allow_html=True)

    else:
        # Layout per modalità video
        if uploaded_video is None:
            st.markdown("""
            <div class="alert-info">
                📤 Carica un video dalla sidebar per iniziare l'analisi dell'allenamento
            </div>
            """, unsafe_allow_html=True)
            
            # Istruzioni per registrare video ottimali
            st.markdown("### 📝 Come registrare un video ottimale:")
            
            tips_col1, tips_col2 = st.columns(2)
            
            with tips_col1:
                st.markdown("""
                **📹 Configurazione Video:**
                - Posiziona la camera di lato per esercizi come push-up e squat
                - Posiziona la camera frontalmente per bicep curl e shoulder press
                - Assicurati che tutto il corpo sia visibile
                - Usa buona illuminazione
                """)
                
            with tips_col2:
                st.markdown("""
                **⚙️ Impostazioni Tecniche:**
                - Risoluzione minima: 720p
                - Formato supportato: MP4, AVI, MOV, MKV
                - Durata massima consigliata: 10 minuti
                - Evita movimenti della camera
                """)

    # Grafici e Analytics (solo se ci sono dati)
    if len(st.session_state.reps_data) > 1:
        st.markdown("---")
        st.markdown("### 📈 Analytics in Tempo Reale")
        
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            st.markdown("#### 📊 Movimento e Ripetizioni")
            df = pd.DataFrame(st.session_state.reps_data)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Angolo nel Tempo', 'Ripetizioni Cumulative'),
                vertical_spacing=0.1
            )
            
            # Grafico angolo con zone colorate
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['angle'], 
                        mode='lines', name='Angolo',
                        line=dict(color='#667eea', width=3)),
                row=1, col=1
            )
            
            # Zone di performance
            fig.add_hline(y=160, line_dash="dash", line_color="green", 
                        annotation_text="Zona Alta", row=1, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                        annotation_text="Zona Bassa", row=1, col=1)
            
            # Riempimento zone
            fig.add_hrect(y0=160, y1=180, fillcolor="green", opacity=0.1, row=1, col=1)
            fig.add_hrect(y0=50, y1=70, fillcolor="red", opacity=0.1, row=1, col=1)
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['reps'], 
                        mode='lines+markers', name='Ripetizioni',
                        line=dict(color='#764ba2', width=3),
                        marker=dict(size=6, color='white', line=dict(color='#764ba2', width=2))),
                row=2, col=1
            )
            
            fig.update_layout(
                height=500, 
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            fig.update_xaxes(title_text="Tempo (s)", row=2, col=1)
            fig.update_yaxes(title_text="Angolo (°)", row=1, col=1)
            fig.update_yaxes(title_text="Ripetizioni", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        with chart_col2:
            st.markdown("#### 📋 Progresso Allenamento")
            
            # Progress bars animate
            progress_reps = (st.session_state.rep_count / target_reps) * 100
            progress_series = (st.session_state.series_count / target_series) * 100
            
            st.markdown("**Progresso Ripetizioni Correnti:**")
            st.progress(min(progress_reps / 100, 1.0))
            st.write(f"{st.session_state.rep_count}/{target_reps} ripetizioni ({progress_reps:.1f}%)")
            
            st.markdown("**Progresso Serie Totali:**")
            st.progress(progress_series / 100)
            st.write(f"{st.session_state.series_count}/{target_series} serie ({progress_series:.1f}%)")
            
            # Statistiche istantanee
            if len(df) > 0:
                current_angle = df['angle'].iloc[-1]
                avg_angle = df['angle'].mean()
                angle_std = df['angle'].std()
                
                st.markdown("**📊 Statistiche Correnti:**")
                
                metrics_col1, metrics_col2 = st.columns(2)
                with metrics_col1:
                    st.metric("Angolo Attuale", f"{current_angle:.1f}°")
                    st.metric("Angolo Medio", f"{avg_angle:.1f}°")
                with metrics_col2:
                    st.metric("Deviazione", f"{angle_std:.1f}°")
                    duration = df['timestamp'].iloc[-1] if len(df) > 0 else 0
                    st.metric("Durata", f"{duration:.1f}s")

    # Sezione Allenamenti Salvati con design migliorato
    if len(st.session_state.saved_workouts) > 0:
        st.markdown("---")
        st.markdown("### 💾 Storico Allenamenti")
        
        # Tab per diversi tipi di visualizzazione
        tab1, tab2, tab3 = st.tabs(["📋 Lista Allenamenti", "📊 Analisi Dettagliata", "⚔️ Confronto"])
        
        with tab1:
            # Lista allenamenti con cards moderne
            for i, workout in enumerate(reversed(st.session_state.saved_workouts)):
                with st.expander(f"🏋️ {workout['name']} - {workout['exercise']} ({workout['timestamp']})"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("💪 Esercizio", workout['exercise'])
                    with col2:
                        st.metric("🔄 Ripetizioni", workout['total_reps'])
                    with col3:
                        st.metric("📊 Serie", f"{workout['completed_series']}/{workout['target_series']}")
                    with col4:
                        st.metric("⏱️ Durata", f"{workout['duration']:.1f}s")
                    
                    # Indicatore sorgente
                    source_icon = "📷" if workout.get('source') == 'webcam_live' else "🎥"
                    source_text = "Webcam Live" if workout.get('source') == 'webcam_live' else "Video Caricato"
                    st.write(f"{source_icon} **Sorgente:** {source_text}")
        
        with tab2:
            # Analisi dettagliata
            st.markdown("#### Seleziona un allenamento per l'analisi dettagliata:")
            
            selected_idx = st.selectbox(
                "Scegli allenamento:",
                options=range(len(st.session_state.saved_workouts)),
                format_func=lambda x: f"{st.session_state.saved_workouts[x]['name']} - {st.session_state.saved_workouts[x]['timestamp']}"
            )
            
            if selected_idx is not None:
                workout = st.session_state.saved_workouts[selected_idx]
                
                # Header informazioni
                info_col1, info_col2, info_col3, info_col4 = st.columns(4)
                with info_col1:
                    st.metric("💪 Esercizio", workout['exercise'])
                with info_col2:
                    st.metric("📊 Serie", f"{workout['completed_series']}/{workout['target_series']}")
                with info_col3:
                    st.metric("🔄 Ripetizioni", workout['total_reps'])
                with info_col4:
                    st.metric("⏱️ Durata", f"{workout['duration']:.1f}s")
                
                # Grafico super dettagliato
                if len(workout['data']) > 0:
                    df_workout = pd.DataFrame(workout['data'])
                    
                    # Crea dashboard completa
                    fig_dashboard = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=(
                            '📈 Andamento Angolo', '📊 Distribuzione Angoli',
                            '🔄 Ripetizioni nel Tempo', '⚡ Velocità Movimento',
                            '🎯 Analisi Performance', '📋 Riepilogo Statistiche'
                        ),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                            [{"secondary_y": False}, {"secondary_y": False}],
                            [{"secondary_y": False}, {"type": "table"}]],
                        vertical_spacing=0.1
                    )
                    
                    # 1. Andamento angolo con zone performance
                    fig_dashboard.add_trace(
                        go.Scatter(x=df_workout['timestamp'], y=df_workout['angle'], 
                                mode='lines', name='Angolo', 
                                line=dict(color='#667eea', width=2)),
                        row=1, col=1
                    )
                    
                    # Zone di performance colorate
                    fig_dashboard.add_hrect(y0=160, y1=180, fillcolor="green", opacity=0.2, row=1, col=1)
                    fig_dashboard.add_hrect(y0=70, y1=160, fillcolor="yellow", opacity=0.1, row=1, col=1)
                    fig_dashboard.add_hrect(y0=50, y1=70, fillcolor="red", opacity=0.2, row=1, col=1)
                    
                    # 2. Distribuzione angoli
                    fig_dashboard.add_trace(
                        go.Histogram(x=df_workout['angle'], nbinsx=25, name='Distribuzione',
                                marker_color='#764ba2', opacity=0.7),
                        row=1, col=2
                    )
                    
                    # 3. Ripetizioni nel tempo
                    fig_dashboard.add_trace(
                        go.Scatter(x=df_workout['timestamp'], y=df_workout['reps'], 
                                mode='lines+markers', name='Ripetizioni',
                                line=dict(color='#2ecc71', width=3),
                                marker=dict(size=8, color='white', line=dict(color='#2ecc71', width=2))),
                        row=2, col=1
                    )
                    
                    # 4. Velocità movimento
                    if len(df_workout) > 1:
                        velocity = np.gradient(df_workout['angle'], df_workout['timestamp'])
                        fig_dashboard.add_trace(
                            go.Scatter(x=df_workout['timestamp'], y=velocity, 
                                    mode='lines', name='Velocità',
                                    line=dict(color='#e74c3c', width=2)),
                            row=2, col=2
                        )
                    
                    # 5. Analisi performance (radar chart simulato con bar)
                    performance_metrics = {
                        'Consistenza': max(0, 100 - df_workout['angle'].std()),
                        'Range Movimento': min(100, (df_workout['angle'].max() - df_workout['angle'].min()) / 1.8),
                        'Ritmo': min(100, (workout['total_reps'] / workout['duration']) * 20),
                        'Completamento': (workout['completed_series'] / workout['target_series']) * 100
                    }
                    
                    fig_dashboard.add_trace(
                        go.Bar(x=list(performance_metrics.keys()), 
                            y=list(performance_metrics.values()),
                            name='Performance',
                            marker_color=['#3498db', '#9b59b6', '#f39c12', '#2ecc71']),
                        row=3, col=1
                    )
                    
                    # 6. Tabella statistiche
                    stats_data = [
                        ['Angolo Medio', f"{df_workout['angle'].mean():.1f}°"],
                        ['Angolo Min/Max', f"{df_workout['angle'].min():.1f}° / {df_workout['angle'].max():.1f}°"],
                        ['Range Movimento', f"{df_workout['angle'].max() - df_workout['angle'].min():.1f}°"],
                        ['Deviazione Standard', f"{df_workout['angle'].std():.1f}°"],
                        ['Reps/Minuto', f"{(workout['total_reps'] / workout['duration']) * 60:.1f}"],
                        ['Efficienza', f"{(workout['total_reps'] / (workout['target_reps'] * workout['target_series'])) * 100:.1f}%"]
                    ]
                    
                    fig_dashboard.add_trace(
                        go.Table(
                            header=dict(values=['Metrica', 'Valore'],
                                    fill_color='#667eea',
                                    font=dict(color='white', size=12)),
                            cells=dict(values=[[row[0] for row in stats_data], 
                                            [row[1] for row in stats_data]],
                                    fill_color='#f8f9fa',
                                    font=dict(size=11))
                        ),
                        row=3, col=2
                    )
                    
                    fig_dashboard.update_layout(
                        height=1000,
                        showlegend=False,
                        title_text=f"📊 Dashboard Completa: {workout['name']}",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_dashboard, use_container_width=True)
                    
                    # Insights AI-powered
                    st.markdown("#### 🤖 Insights Intelligenti")
                    
                    insight_col1, insight_col2 = st.columns(2)
                    
                    with insight_col1:
                        # Analisi consistenza
                        angle_consistency = df_workout['angle'].std()
                        if angle_consistency < 10:
                            consistency_level = "🌟 Eccellente"
                            consistency_color = "success"
                        elif angle_consistency < 20:
                            consistency_level = "👍 Buona"
                            consistency_color = "info"
                        else:
                            consistency_level = "⚠️ Da migliorare"
                            consistency_color = "warning"
                        
                        st.markdown(f"**Consistenza Movimento:** {consistency_level}")
                        st.write(f"Deviazione standard: {angle_consistency:.1f}°")
                    
                    with insight_col2:
                        # Analisi ritmo
                        reps_per_min = (workout['total_reps'] / workout['duration']) * 60
                        if reps_per_min > 15:
                            pace_level = "🚀 Ritmo veloce"
                        elif reps_per_min > 8:
                            pace_level = "👌 Ritmo ottimale"
                        else:
                            pace_level = "🐌 Ritmo lento"
                        
                        st.markdown(f"**Ritmo Allenamento:** {pace_level}")
                        st.write(f"{reps_per_min:.1f} ripetizioni/minuto")
                    
                    # Suggerimenti personalizzati
                    st.markdown("#### 💡 Suggerimenti Personalizzati")
                    
                    suggestions = []
                    if angle_consistency > 20:
                        suggestions.append("🎯 Lavora sulla consistenza del movimento - cerca di mantenere un range di movimento più uniforme")
                    
                    if reps_per_min < 8:
                        suggestions.append("⚡ Prova ad aumentare leggermente il ritmo per migliorare l'intensità dell'allenamento")
                    elif reps_per_min > 20:
                        suggestions.append("🐌 Considera di rallentare per concentrarti sulla forma corretta")
                    
                    range_movement = df_workout['angle'].max() - df_workout['angle'].min()
                    if range_movement < 80:
                        suggestions.append("📏 Cerca di aumentare il range di movimento per massimizzare i benefici dell'esercizio")
                    
                    completion_rate = (workout['completed_series'] / workout['target_series']) * 100
                    if completion_rate < 100:
                        suggestions.append(f"💪 Hai completato {completion_rate:.1f}% dell'allenamento pianificato - la prossima volta punta a completare tutte le serie!")
                    
                    if not suggestions:
                        suggestions.append("🏆 Ottimo allenamento! Continua così per mantenere questa qualità di esecuzione")
                    
                    for suggestion in suggestions:
                        st.write(f"• {suggestion}")
        
        with tab3:
            # Confronto tra allenamenti
            if len(st.session_state.saved_workouts) > 1:
                st.markdown("#### ⚔️ Confronta Due Allenamenti")
                
                workout_options = list(range(len(st.session_state.saved_workouts)))
                workout_labels = [f"{w['name']} - {w['timestamp']}" for w in st.session_state.saved_workouts]
                
                comp_col1, comp_col2 = st.columns(2)
                
                with comp_col1:
                    selected_w1 = st.selectbox("🥇 Allenamento 1", options=workout_options, 
                                            format_func=lambda x: workout_labels[x])
                with comp_col2:
                    selected_w2 = st.selectbox("🥈 Allenamento 2", options=workout_options, index=1,
                                            format_func=lambda x: workout_labels[x])
                
                if selected_w1 != selected_w2:
                    w1 = st.session_state.saved_workouts[selected_w1]
                    w2 = st.session_state.saved_workouts[selected_w2]
                    
                    # Confronto visivo
                    st.markdown("##### 📊 Confronto Metriche")
                    
                    metrics_comparison = {
                        'Serie Completate': [w1['completed_series'], w2['completed_series']],
                        'Ripetizioni Totali': [w1['total_reps'], w2['total_reps']],
                        'Durata (s)': [w1['duration'], w2['duration']],
                    }
                    
                    # Aggiungi metriche avanzate se disponibili
                    if w1['data'] and w2['data']:
                        df1 = pd.DataFrame(w1['data'])
                        df2 = pd.DataFrame(w2['data'])
                        
                        metrics_comparison.update({
                            'Angolo Medio': [df1['angle'].mean(), df2['angle'].mean()],
                            'Consistenza': [100 - df1['angle'].std(), 100 - df2['angle'].std()],
                            'Reps/Min': [(w1['total_reps']/w1['duration'])*60, (w2['total_reps']/w2['duration'])*60]
                        })
                    
                    # Crea grafico comparativo
                    fig_comparison = go.Figure()
                    
                    categories = list(metrics_comparison.keys())
                    values_w1 = [metrics_comparison[cat][0] for cat in categories]
                    values_w2 = [metrics_comparison[cat][1] for cat in categories]
                    
                    fig_comparison.add_trace(go.Bar(
                        name=w1['name'],
                        x=categories,
                        y=values_w1,
                        marker_color='#667eea',
                        opacity=0.8
                    ))
                    
                    fig_comparison.add_trace(go.Bar(
                        name=w2['name'],
                        x=categories,
                        y=values_w2,
                        marker_color='#764ba2',
                        opacity=0.8
                    ))
                    
                    fig_comparison.update_layout(
                        title="📊 Confronto Diretto delle Metriche",
                        xaxis_title="Metriche",
                        yaxis_title="Valori",
                        barmode='group',
                        height=500,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_comparison, use_container_width=True)
                    
                    # Tabella di confronto dettagliata
                    st.markdown("##### 📋 Confronto Dettagliato")
                    
                    comparison_data = []
                    for metric, values in metrics_comparison.items():
                        winner = "🥇 Allenamento 1" if values[0] > values[1] else "🥈 Allenamento 2" if values[1] > values[0] else "🤝 Pari"
                        # Per la durata, meno è meglio
                        if metric == 'Durata (s)':
                            winner = "🥇 Allenamento 1" if values[0] < values[1] else "🥈 Allenamento 2" if values[1] < values[0] else "🤝 Pari"
                        
                        comparison_data.append([
                            metric,
                            f"{values[0]:.1f}",
                            f"{values[1]:.1f}",
                            winner
                        ])
                    
                    comparison_df = pd.DataFrame(comparison_data, 
                                                columns=['Metrica', f'{w1["name"]} (1)', f'{w2["name"]} (2)', 'Migliore'])
                    
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Insights del confronto
                    st.markdown("##### 🎯 Analisi del Progresso")
                    
                    improvements = []
                    declines = []
                    
                    for metric, values in metrics_comparison.items():
                        if metric == 'Durata (s)':  # Per durata, meno è meglio
                            if values[1] < values[0]:
                                improvements.append(f"⚡ {metric}: miglioramento di {values[0] - values[1]:.1f}")
                            elif values[1] > values[0]:
                                declines.append(f"⏳ {metric}: aumento di {values[1] - values[0]:.1f}")
                        else:
                            if values[1] > values[0]:
                                improvements.append(f"📈 {metric}: miglioramento di {values[1] - values[0]:.1f}")
                            elif values[1] < values[0]:
                                declines.append(f"📉 {metric}: diminuzione di {values[0] - values[1]:.1f}")
                    
                    if improvements:
                        st.markdown("**🎉 Miglioramenti:**")
                        for imp in improvements:
                            st.write(f"• {imp}")
                    
                    if declines:
                        st.markdown("**⚠️ Aree da migliorare:**")
                        for dec in declines:
                            st.write(f"• {dec}")
                    
                    if not improvements and not declines:
                        st.markdown("**🤝 Performance simili tra i due allenamenti**")
            else:
                st.info("💡 Serve almeno 2 allenamenti salvati per abilitare il confronto")

    # Footer con informazioni aggiuntive
    st.markdown("---")
    st.markdown("### ℹ️ Informazioni")

    footer_col1, footer_col2, footer_col3 = st.columns(3)

    with footer_col1:
        st.markdown("""
        **📱 Compatibilità:**
        - Desktop: Webcam + Video Upload
        - Mobile: Solo Video Upload
        - Formati supportati: MP4, AVI, MOV, MKV
        """)

    with footer_col2:
        st.markdown("""
        **🎯 Esercizi Supportati:**
        - Push-up (vista laterale)
        - Squat (vista laterale)  
        - Bicep Curl (vista frontale)
        - Shoulder Press (vista frontale)
        """)

    with footer_col3:
        st.markdown("""
        **🔧 Tecnologie:**
        - MediaPipe per pose detection
        - OpenCV per video processing
        - Plotly per visualizzazioni avanzate
        - Streamlit per l'interfaccia
        """)

    # Aggiunge un po' di spazio alla fine
    st.markdown("<br><br>", unsafe_allow_html=True)
