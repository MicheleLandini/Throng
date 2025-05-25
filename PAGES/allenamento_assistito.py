import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

st.set_page_config(page_title="Allenamento Assistito", layout="wide")

st.title("🏋️ Allenamento Assistito Avanzato")

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
def detect_reps(angle, threshold_down=70, threshold_up=160):
    global rep_count, stage
    
    if angle > threshold_up:
        stage = "up"
    if angle < threshold_down and stage == "up":
        stage = "down"
        rep_count += 1
        
    return rep_count, stage

# Inizializza variabili di sessione
if 'rep_count' not in st.session_state:
    st.session_state.rep_count = 0
if 'stage' not in st.session_state:
    st.session_state.stage = None
if 'reps_data' not in st.session_state:
    st.session_state.reps_data = []
if 'workout_started' not in st.session_state:
    st.session_state.workout_started = False
if 'countdown_started' not in st.session_state:
    st.session_state.countdown_started = False
if 'series_count' not in st.session_state:
    st.session_state.series_count = 0
if 'chart_counter' not in st.session_state:
    st.session_state.chart_counter = 0
if 'workout_paused' not in st.session_state:
    st.session_state.workout_paused = False
if 'saved_workouts' not in st.session_state:
    st.session_state.saved_workouts = []
if 'pause_time' not in st.session_state:
    st.session_state.pause_time = 0
if 'total_pause_time' not in st.session_state:
    st.session_state.total_pause_time = 0

# Sidebar per impostazioni
st.sidebar.header("⚙️ Impostazioni Allenamento")

# Input numerici
target_reps = st.sidebar.number_input("Ripetizioni per serie", min_value=1, max_value=100, value=10)
target_series = st.sidebar.number_input("Numero di serie", min_value=1, max_value=20, value=3)
countdown_time = st.sidebar.number_input("Countdown iniziale (secondi)", min_value=3, max_value=10, value=5)

# Selezione esercizio
exercise_type = st.sidebar.selectbox(
    "Tipo di esercizio",
    ["Push-up", "Squat", "Bicep Curl", "Shoulder Press"]
)

# Campi di testo per informazioni
workout_name = st.sidebar.text_input("Nome allenamento", value="Sessione " + time.strftime("%Y-%m-%d"))
notes = st.sidebar.text_area("Note aggiuntive", placeholder="Inserisci note sull'allenamento...")


# Controlli
control_col1, control_col2, control_col3, control_col4 = st.columns(4)

with control_col1:
    if st.button("🚀 Inizia Allenamento", type="primary"):
        if not st.session_state.workout_paused:
            # Nuovo allenamento
            st.session_state.countdown_started = True
            st.session_state.workout_started = False
            st.session_state.rep_count = 0
            st.session_state.series_count = 0
            st.session_state.reps_data = []
            st.session_state.total_pause_time = 0
        else:
            # Riprendi allenamento
            st.session_state.workout_started = True
            st.session_state.workout_paused = False
            st.session_state.total_pause_time += time.time() - st.session_state.pause_time

with control_col2:
    if st.button("⏸️ Pausa"):
        if st.session_state.workout_started:
            st.session_state.workout_started = False
            st.session_state.workout_paused = True
            st.session_state.pause_time = time.time()
        
with control_col3:
    if st.button("🔄 Reset"):
        st.session_state.rep_count = 0
        st.session_state.series_count = 0
        st.session_state.reps_data = []
        st.session_state.workout_started = False
        st.session_state.countdown_started = False
        st.session_state.workout_paused = False
        st.session_state.total_pause_time = 0

with control_col4:
    if st.button("💾 Salva", disabled=len(st.session_state.reps_data) == 0):
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
                'notes': notes,
                'duration': st.session_state.reps_data[-1]['timestamp'] if st.session_state.reps_data else 0
            }
            st.session_state.saved_workouts.append(workout_data)
            st.success(f"✅ Allenamento '{workout_name}' salvato con successo!")


# Layout principale
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Video Feed")
    frame_placeholder = st.empty()
    
with col2:
    st.subheader("📊 Statistiche Live")
    stats_placeholder = st.empty()
    countdown_placeholder = st.empty()

# Grafici
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("📈 Grafico Movimento")
    chart_placeholder = st.empty()
    
with chart_col2:
    st.subheader("📋 Progresso Serie")
    progress_placeholder = st.empty()

# Variabili globali per il conteggio
rep_count = st.session_state.rep_count
stage = st.session_state.stage

# Webcam e processing
run = st.session_state.workout_started or st.session_state.countdown_started

if run:
    cap = cv2.VideoCapture(0)
    start_time = time.time()
    countdown_start = time.time()
    
    while cap.isOpened() and (st.session_state.workout_started or st.session_state.countdown_started):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # Mirror image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Countdown logic
        if st.session_state.countdown_started and not st.session_state.workout_started:
            elapsed = time.time() - countdown_start
            remaining = countdown_time - elapsed
            
            if remaining > 0:
                # Mostra countdown sul frame
                cv2.putText(frame_rgb, f"Inizia tra: {int(remaining) + 1}", 
                           (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                countdown_placeholder.markdown(f"### ⏰ Inizia tra: {int(remaining) + 1} secondi")
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
                
                # Estrai coordinate in base al tipo di esercizio
                if exercise_type == "Push-up":
                    # Angolo del gomito per push-up
                    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                elif exercise_type == "Squat":
                    # Angolo del ginocchio per squat
                    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                           landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                    angle = calculate_angle(hip, knee, ankle)
                    
                elif exercise_type == "Bicep Curl":
                    # Angolo del gomito per bicep curl
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                    angle = calculate_angle(shoulder, elbow, wrist)
                    
                else:  # Shoulder Press
                    # Angolo del braccio per shoulder press
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    angle = calculate_angle(elbow, shoulder, hip)

                # Rilevamento ripetizioni
                rep_count, stage = detect_reps(angle)
                st.session_state.rep_count = rep_count
                st.session_state.stage = stage
                
                # Salva dati per il grafico (compensando per il tempo di pausa)
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
                
                # Mostra informazioni sul frame
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
                        st.success("🎉 Allenamento completato!")

        # Mostra frame
        frame_resized = cv2.resize(frame_rgb, (640, 480))
        frame_placeholder.image(frame_resized, channels="RGB")
        
        # Aggiorna statistiche
        with stats_placeholder.container():
            st.metric("Ripetizioni correnti", st.session_state.rep_count, f"Target: {target_reps}")
            st.metric("Serie completate", st.session_state.series_count, f"Target: {target_series}")
            st.metric("Esercizio", exercise_type)
            if st.session_state.stage:
                st.write(f"**Fase:** {st.session_state.stage}")
            if st.session_state.workout_paused:
                st.warning("⏸️ Allenamento in pausa - Premi 'Inizia Allenamento' per riprendere")
        
        # Aggiorna grafico in tempo reale
        if len(st.session_state.reps_data) > 1:
            df = pd.DataFrame(st.session_state.reps_data)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Angolo nel tempo', 'Ripetizioni cumulative'),
                vertical_spacing=0.1
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['angle'], mode='lines', name='Angolo'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=df['timestamp'], y=df['reps'], mode='lines+markers', name='Reps'),
                row=2, col=1
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.session_state.chart_counter += 1
            chart_placeholder.plotly_chart(fig, use_container_width=True, key=f"workout_chart_{st.session_state.chart_counter}")
        
        # Progresso serie
        with progress_placeholder.container():
            progress = (st.session_state.series_count / target_series) * 100
            st.progress(progress / 100)
            st.write(f"Progresso: {progress:.1f}%")

        time.sleep(0.1)  # Piccola pausa per evitare sovraccarico

    cap.release()

else:
    if st.session_state.workout_paused:
        st.warning("⏸️ Allenamento in pausa - Premi 'Inizia Allenamento' per riprendere")
    else:
        st.info("👆 Clicca 'Inizia Allenamento' per cominciare")

# Sezione Allenamenti Salvati
if len(st.session_state.saved_workouts) > 0:
    st.header("💾 Allenamenti Salvati")
    
    # Selezione allenamento per visualizzazione dettagliata
    selected_workout = st.selectbox(
        "Seleziona un allenamento per visualizzazione dettagliata:",
        options=range(len(st.session_state.saved_workouts)),
        format_func=lambda x: f"{st.session_state.saved_workouts[x]['name']} - {st.session_state.saved_workouts[x]['timestamp']}"
    )
    
    if selected_workout is not None:
        workout = st.session_state.saved_workouts[selected_workout]
        
        # Informazioni generali
        info_col1, info_col2, info_col3, info_col4 = st.columns(4)
        with info_col1:
            st.metric("Esercizio", workout['exercise'])
        with info_col2:
            st.metric("Serie Completate", f"{workout['completed_series']}/{workout['target_series']}")
        with info_col3:
            st.metric("Ripetizioni Totali", workout['total_reps'])
        with info_col4:
            st.metric("Durata", f"{workout['duration']:.1f}s")
        
        # Grafico dettagliato
        if len(workout['data']) > 0:
            df_saved = pd.DataFrame(workout['data'])
            
            # Crea grafico super dettagliato
            fig_detailed = make_subplots(
                rows=3, cols=2,
                subplot_titles=(
                    'Angolo nel Tempo', 'Distribuzione Angoli',
                    'Ripetizioni Cumulative', 'Velocità Movimento',
                    'Analisi Fase Movimento', 'Performance per Serie'
                ),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]],
                vertical_spacing=0.08
            )
            
            # 1. Grafico angolo nel tempo con zone di performance
            fig_detailed.add_trace(
                go.Scatter(x=df_saved['timestamp'], y=df_saved['angle'], 
                          mode='lines', name='Angolo', line=dict(color='blue', width=2)),
                row=1, col=1
            )
            
            # Zone di performance
            fig_detailed.add_hline(y=160, line_dash="dash", line_color="green", 
                                 annotation_text="Zona Alta", row=1, col=1)
            fig_detailed.add_hline(y=70, line_dash="dash", line_color="red", 
                                 annotation_text="Zona Bassa", row=1, col=1)
            
            # 2. Istogramma distribuzione angoli
            fig_detailed.add_trace(
                go.Histogram(x=df_saved['angle'], nbinsx=20, name='Distribuzione', 
                           marker_color='lightblue'),
                row=1, col=2
            )
            
            # 3. Ripetizioni cumulative
            fig_detailed.add_trace(
                go.Scatter(x=df_saved['timestamp'], y=df_saved['reps'], 
                          mode='lines+markers', name='Ripetizioni', 
                          line=dict(color='green', width=3)),
                row=2, col=1
            )
            
            # 4. Velocità di movimento (derivata dell'angolo)
            if len(df_saved) > 1:
                angle_velocity = np.gradient(df_saved['angle'], df_saved['timestamp'])
                fig_detailed.add_trace(
                    go.Scatter(x=df_saved['timestamp'], y=angle_velocity, 
                              mode='lines', name='Velocità', line=dict(color='orange')),
                    row=2, col=2
                )
            
            # 5. Analisi fase movimento
            phases = []
            for i, angle in enumerate(df_saved['angle']):
                if angle > 160:
                    phases.append('Alto')
                elif angle < 70:
                    phases.append('Basso')
                else:
                    phases.append('Transizione')
            
            phase_counts = pd.Series(phases).value_counts()
            fig_detailed.add_trace(
                go.Bar(x=phase_counts.index, y=phase_counts.values, 
                       name='Fasi', marker_color=['red', 'orange', 'green']),
                row=3, col=1
            )
            
            # 6. Performance per serie (se applicabile)
            if workout['completed_series'] > 0:
                reps_per_series = workout['total_reps'] / max(workout['completed_series'], 1)
                series_data = [reps_per_series] * workout['completed_series']
                series_labels = [f"Serie {i+1}" for i in range(workout['completed_series'])]
                
                fig_detailed.add_trace(
                    go.Bar(x=series_labels, y=series_data, name='Reps/Serie',
                           marker_color='purple'),
                    row=3, col=2
                )
            
            fig_detailed.update_layout(
                height=900, 
                showlegend=False,
                title_text=f"Analisi Dettagliata: {workout['name']} - {workout['exercise']}"
            )
            
            st.plotly_chart(fig_detailed, use_container_width=True)
            
            # Statistiche avanzate
            st.subheader("📊 Statistiche Avanzate")
            adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
            
            with adv_col1:
                avg_angle = np.mean(df_saved['angle'])
                st.metric("Angolo Medio", f"{avg_angle:.1f}°")
                
            with adv_col2:
                angle_std = np.std(df_saved['angle'])
                st.metric("Deviazione Standard", f"{angle_std:.1f}°")
                
            with adv_col3:
                max_angle = np.max(df_saved['angle'])
                min_angle = np.min(df_saved['angle'])
                range_motion = max_angle - min_angle
                st.metric("Range di Movimento", f"{range_motion:.1f}°")
                
            with adv_col4:
                reps_per_minute = (workout['total_reps'] / workout['duration']) * 60
                st.metric("Ripetizioni/Minuto", f"{reps_per_minute:.1f}")
            
            # Note e commenti
            if workout['notes']:
                st.subheader("📝 Note")
                st.write(workout['notes'])

# Sidebar con riassunto sessione
if len(st.session_state.reps_data) > 0:
    st.sidebar.header("📈 Riassunto Sessione")
    st.sidebar.write(f"**Nome:** {workout_name}")
    st.sidebar.write(f"**Esercizio:** {exercise_type}")
    st.sidebar.write(f"**Serie completate:** {st.session_state.series_count}/{target_series}")
    st.sidebar.write(f"**Ripetizioni totali:** {sum([d['reps'] for d in st.session_state.reps_data[-1:]]) if st.session_state.reps_data else 0}")
    if notes:
        st.sidebar.write(f"**Note:** {notes}")


# Confronto tra due allenamenti salvati
if len(st.session_state.saved_workouts) > 1:
    st.header("⚔️ Confronta Allenamenti Salvati")
    
    workout_options = list(range(len(st.session_state.saved_workouts)))
    workout_labels = [f"{w['name']} - {w['timestamp']}" for w in st.session_state.saved_workouts]
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        selected_w1 = st.selectbox("Seleziona Allenamento 1", options=workout_options, format_func=lambda x: workout_labels[x])
    with col_comp2:
        selected_w2 = st.selectbox("Seleziona Allenamento 2", options=workout_options, index=1, format_func=lambda x: workout_labels[x])
    
    if selected_w1 is not None and selected_w2 is not None and selected_w1 != selected_w2:
        w1 = st.session_state.saved_workouts[selected_w1]
        w2 = st.session_state.saved_workouts[selected_w2]
        
        # Calcola metriche principali per confronto
        def avg_angle(workout):
            data = workout['data']
            if len(data) == 0:
                return 0
            return np.mean([d['angle'] for d in data])
        
        w1_avg_angle = avg_angle(w1)
        w2_avg_angle = avg_angle(w2)
        
        # Confronta metriche
        def compare_metric(name, val1, val2, higher_is_better=True):
            better = None
            if val1 > val2:
                better = "Allenamento 1" if higher_is_better else "Allenamento 2"
            elif val2 > val1:
                better = "Allenamento 2" if higher_is_better else "Allenamento 1"
            else:
                better = "Pari"
            return better
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Serie Completate", f"{w1['completed_series']}", f"vs {w2['completed_series']}")
        with col2:
            st.metric("Ripetizioni Totali", f"{w1['total_reps']}", f"vs {w2['total_reps']}")
        with col3:
            st.metric("Durata (s)", f"{w1['duration']:.1f}", f"vs {w2['duration']:.1f}")
        
        st.write("---")
        
        st.subheader("📊 Confronto Avanzato")
        
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            st.write(f"**Allenamento 1: {w1['name']}**")
            st.write(f"- Angolo Medio: {w1_avg_angle:.1f}°")
            st.write(f"- Serie Completate: {w1['completed_series']}")
            st.write(f"- Ripetizioni Totali: {w1['total_reps']}")
            st.write(f"- Durata: {w1['duration']:.1f} s")
            
        with comp_col2:
            st.write(f"**Allenamento 2: {w2['name']}**")
            st.write(f"- Angolo Medio: {w2_avg_angle:.1f}°")
            st.write(f"- Serie Completate: {w2['completed_series']}")
            st.write(f"- Ripetizioni Totali: {w2['total_reps']}")
            st.write(f"- Durata: {w2['duration']:.1f} s")
        
        # Riassunto miglioramenti
        st.write("---")
        st.subheader("📈 Risultati del Confronto")
        
        improved_series = compare_metric("Serie Completate", w2['completed_series'], w1['completed_series'])
        improved_reps = compare_metric("Ripetizioni Totali", w2['total_reps'], w1['total_reps'])
        improved_duration = compare_metric("Durata", w1['duration'], w2['duration'], higher_is_better=False)
        improved_angle = compare_metric("Angolo Medio", w2_avg_angle, w1_avg_angle)
        
        st.write(f"- Serie Completate: {improved_series}")
        st.write(f"- Ripetizioni Totali: {improved_reps}")
        st.write(f"- Durata: {improved_duration} (più breve è meglio)")
        st.write(f"- Angolo Medio: {improved_angle}")
