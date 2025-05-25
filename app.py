# ===============================================================
# IMPORTAZIONI
# ===============================================================
import json
import os
import streamlit as st #Per web app
import sqlite3  # AGGIUNTO: db per sqlite3
from reportlab.lib.pagesizes import letter # Per la creazione dei file PDF
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
import time  # x titoli nomi file
from PIL import Image, ImageDraw #per creare delimitatore modificato
import io #Usato per buffer di immagini/PDF, ad es. per st.download_button()
import base64 #usato per html nella home page per centrare logo in maniera responsive
import google.generativeai as genai
import pandas as pd # lo usato per mostrare le tabelle del db a schermo
# Importa il nostro DatabaseManager
from database_helper import DatabaseManager
import re

# ===============================================================
# INIZIALIZZAZIONE
# ===============================================================

# Imposta la pagina iniziale alla home se non è ancora definita
if "page" not in st.session_state:
    st.session_state.page = "home"

st.set_page_config(
    page_title="Throng",
    page_icon="static/logo-min-nobg.png",
)

# ===============================================================
# INIZIALIZZA DATABASE
# ===============================================================
@st.cache_resource
def init_database():
    """Inizializza il database (cache per evitare reinizializzazioni)"""
    db = DatabaseManager()
    
    # Migrazione automatica dai file JSON se esistono
    if os.path.exists("modelli.json"):
        db.migra_da_json("modelli.json")
        # Opzionale: rinomina il file dopo la migrazione
        # os.rename("modelli.json", "modelli.json.backup")
    
    if os.path.exists("SETTINGS/settings.json"):
        db.migra_settings_da_json("SETTINGS/settings.json")
        # os.rename("SETTINGS/settings.json", "SETTINGS/settings.json.backup")
    
    return db

# Inizializza il database
db = init_database()

# Carica modelli nella sessione all'avvio (ora dal database)
if "modelli_esercizi" not in st.session_state:
    st.session_state.modelli_esercizi = db.get_modelli_esercizi()

# ===============================================================
# FUNZIONI DI UTILITÀ PER REINDIRIZZAMENTO ALLE PAGINE
# ===============================================================

# Cambia la pagina attiva
def vai_a(nome_pagina):
    st.session_state.page = nome_pagina



# ===============================================================
# DEFINISCO LA FUNZIONE PER CREARE IL PDF (MODIFICATA)
# ===============================================================

def crea_scheda_pdf(nome, livello, obiettivo, esercizi_per_giorno):
    # === Caricamento impostazioni DAL DATABASE ===
    settings = db.get_settings()

    font_options = ["Helvetica", "Times-Roman", "Courier"]
    font_bold_map = {
        "Helvetica": "Helvetica-Bold",
        "Times-Roman": "Times-Bold",
        "Courier": "Courier-Bold"
    }

    font = settings.get("font", "Helvetica")
    if font not in font_options:
        font = "Helvetica"  # fallback se font non valido

    colore_sfondo = settings.get("colore_sfondo", "#FFFFFF")
    colore_testo = settings.get("colore_testo", "#000000")
    colore_titoli = settings.get("colore_titoli", "#000000")
    colore_g_settimana = settings.get("colore_g_settimana", "#000000")
    stile_delimitatore = settings.get("stile_delimitatore", "Linea semplice")
    dim_logo = settings.get("dimimm", 500)  # in pixel, lo convertiremo
    dim_es = settings.get("dimimmes", 4)
    dimtitoli = settings.get("dimtitoli", 18)     # default 18 pt
    dimtesti = settings.get("dimtesti", 12)       # default 12 pt
    dimgset = settings.get("dimgset", 14)         # default 14 pt

    # NUOVA FUNZIONE: Processa immagini per gestire trasparenza
    def processa_immagine_per_pdf(image_path, background_color):
        """
        Processa un'immagine PNG per gestire la trasparenza.
        Se ha trasparenza, sostituisce lo sfondo trasparente con il colore specificato.
        """
        try:
            # Apri l'immagine con PIL
            img = Image.open(image_path)
            
            # Se l'immagine ha un canale alpha (trasparenza)
            if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
                # Converti il colore hex in RGB
                if background_color.startswith('#'):
                    hex_color = background_color[1:]
                    bg_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
                else:
                    bg_rgb = (255, 255, 255)  # Bianco come fallback
                
                # Crea un'immagine di sfondo con il colore desiderato
                background = Image.new('RGB', img.size, bg_rgb)
                
                # Se l'immagine è in modalità P (palette) con trasparenza
                if img.mode == 'P' and 'transparency' in img.info:
                    img = img.convert('RGBA')
                
                # Incolla l'immagine originale sullo sfondo usando l'alpha come maschera
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[-1])  # Usa il canale alpha come maschera
                elif img.mode == 'LA':
                    # Per immagini in scala di grigi con alpha
                    img_rgb = img.convert('RGBA')
                    background.paste(img_rgb, mask=img_rgb.split()[-1])
                
                # Salva l'immagine processata in un buffer
                buffer = io.BytesIO()
                background.save(buffer, format='PNG')
                buffer.seek(0)
                return ImageReader(buffer)
            else:
                # Se non ha trasparenza, usa l'immagine originale
                return ImageReader(image_path)
                
        except Exception as e:
            print(f"Errore nel processare l'immagine {image_path}: {e}")
            return None

    filename = f"{nome.replace(' ', '_')}_scheda.pdf"
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter

    margin_x = 2 * cm
    margin_y = 2 * cm
    line_height = 14
    y = height - margin_y

    # Sfondo pagina
    c.setFillColor(colore_sfondo)
    c.rect(0, 0, width, height, fill=True, stroke=False)
    
    # Logo (MODIFICATO per gestire trasparenza e posizione dinamica)
    logo_path = "static/logo_scheda.png"
    if os.path.isfile(logo_path):
        try:
            # Usa la nuova funzione per processare il logo
            logo_processed = processa_immagine_per_pdf(logo_path, colore_sfondo)

            if logo_processed:
                logo_width, logo_height = logo_processed.getSize()
                max_width = (dim_logo / 100) * cm
                scale = max_width / logo_width
                logo_width_scaled = logo_width * scale
                logo_height_scaled = logo_height * scale

                # Posiziona il logo vicino alla parte alta della pagina con piccolo margine
                top_margin = 20  # margine superiore in punti (puoi ridurre/incrementare)
                logo_y = height - logo_height_scaled - top_margin

                c.drawImage(
                    logo_processed,
                    x=(width - logo_width_scaled) / 2,
                    y=logo_y,
                    width=logo_width_scaled,
                    height=logo_height_scaled
                )

                # Aggiorna y per il contenuto successivo
                y = logo_y - 20  # spazio tra il logo e il resto del contenuto

        except Exception as e:
            print(f"Errore nel caricare il logo: {e}")


    def get_font_bold(base_font):
        return font_bold_map.get(base_font, base_font)

    def disegna_titolo_principale():
        nonlocal y
        font_bold = get_font_bold(font)
        c.setFont(font_bold, dimtitoli)
        c.setFillColor(colore_titoli)
        c.drawCentredString(width / 2, y, "Scheda di Allenamento")
        y -= 2 * line_height

    def disegna_info_cliente():
        nonlocal y
        font_bold = get_font_bold(font)
        info = [
            ("Nome:", nome),
            ("Livello:", livello),
            ("Obiettivo:", obiettivo),
            ("Giorni di allenamento:", ', '.join(esercizi_per_giorno.keys()))
        ]
            # Calcola altezza riga dinamica (es. 1.6 volte il font più grande)
        max_font_size = max(dimtitoli, dimtesti)
        dynamic_line_height = max_font_size * 1.6

        for etichetta, valore in info:
            c.setFont(font_bold, dimtitoli)
            c.setFillColor(colore_titoli)
            c.drawString(margin_x, y, etichetta)

            etichetta_width = c.stringWidth(etichetta, font_bold, dimtitoli)

            c.setFont(font, dimtesti)
            c.setFillColor(colore_testo)
            c.drawString(margin_x + etichetta_width + 10, y, str(valore))

            y -= dynamic_line_height  # usa altezza riga dinamica

        y -= 4
        disegna_delimitatore()
        y -= line_height
        #y -= 6
        y -= dynamic_line_height  # usa altezza riga dinamica

    def disegna_titolo_giorno(giorno):
        nonlocal y
        font_bold = get_font_bold(font)
        c.setFont(font_bold, dimgset)
        c.setFillColor(colore_g_settimana)
        c.drawString(margin_x, y, giorno)
        y -= line_height * 1.5
        c.setFont(font, dimtesti)
        c.setFillColor(colore_testo)

    def disegna_delimitatore():
        c.setFillColor(colore_testo)
        c.setStrokeColor(colore_testo)
        if stile_delimitatore == "Linea semplice":
            c.line(margin_x, y, width - margin_x, y)
        elif stile_delimitatore == "Linea tratteggiata":
            x = margin_x
            while x < width - margin_x:
                c.line(x, y, x + 5, y)
                x += 10
        elif stile_delimitatore == "Doppia linea":
            c.line(margin_x, y + 2, width - margin_x, y + 2)
            c.line(margin_x, y - 2, width - margin_x, y - 2)
        elif stile_delimitatore == "Rettangolo sottile":
            c.rect(margin_x, y - 1, width - 2 * margin_x, 2, fill=True)

    # INIZIO PDF
    disegna_titolo_principale()
    disegna_info_cliente()

    giorno_corrente = None

    for giorno, esercizi in esercizi_per_giorno.items():
        nuovo_giorno = True
        for i, es in enumerate(esercizi, 1):
            nome_es = es.get("nome", "")
            serie = es.get("serie", "")
            immagine_filename = es.get("immagine", "").strip()
            immagine_path = os.path.join("IMG", immagine_filename)
            testo = f"{i}. {nome_es} ({serie})" if serie else f"{i}. {nome_es}"

            img_height = 0
            img_processed = None

            if immagine_path and os.path.isfile(immagine_path):
                try:
                    img_processed = processa_immagine_per_pdf(immagine_path, colore_sfondo)
                    if img_processed:
                        iw, ih = img_processed.getSize()
                        max_width = dim_es * cm
                        scale = max_width / iw
                        img_width = iw * scale
                        img_height = ih * scale
                except Exception as e:
                    print(f"Errore immagine '{immagine_path}': {e}")
                    img_height = 0
            # Calcola altezza riga dinamica (es. 1.6 volte il font più grande)
            max_font_size = max(dimtitoli, dimtesti)
            dynamic_line_height = max_font_size * 1.6
            spazio_necessario = line_height + (img_height + 10 if img_height else 10)
            if y - spazio_necessario < margin_y:
                c.showPage()
                y = height - margin_y

                # Sfondo pagina
                c.setFillColor(colore_sfondo)
                c.rect(0, 0, width, height, fill=True, stroke=False)

                # Ridisegna solo se è un nuovo giorno
                if nuovo_giorno:
                    c.setFont(font, dimtesti)
                    c.setFillColor(colore_testo)
                    disegna_titolo_giorno(giorno)
                    nuovo_giorno = False

            # Disegna giorno solo una volta per gruppo
            if nuovo_giorno:
                disegna_titolo_giorno(giorno)
                nuovo_giorno = False

            c.setFillColor(colore_testo)
            c.setFont(font, dimtesti)
            c.drawString(margin_x, y, testo)
            y -= dynamic_line_height  # Spazio extra sotto il testo dell’esercizio

            if img_processed and img_height:
                # Posizionare immagine a destra, rispettando i margini
                img_x = width - margin_x - img_width
                img_y = y - img_height
                c.drawImage(img_processed, img_x, img_y, width=img_width, height=img_height)
                y -= img_height + 10
            else:
                y -= 10


        y -= dynamic_line_height  # Spazio extra sotto il testo dell’esercizio

    c.save()
    return filename

# ===============================================================
# RESTO DEL CODICE RIMANE UGUALE FINO ALLA SEZIONE MODIFICA ESERCIZI
# ===============================================================


if st.session_state.page == "home":


    def get_base64_image(path):
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()

    img_base64 = get_base64_image("STATIC/logo-min-nobg.png")


    st.markdown(
        f"""
        <div style="display: flex; justify-content: center;">
            <img src="data:image/png;base64,{img_base64}" width="100" height="100" style="object-fit: contain;" />
        </div>
        """,
        unsafe_allow_html=True
    )

    st.title("Home - Gestionale schede palestra")


    with st.container():
        st.button("➕ Crea scheda cliente", key="btn_crea_scheda", use_container_width=True, on_click=vai_a, args=("info_cliente",))
        st.button("✏️ Modifica esercizi", key="btn_mod_esercizi", use_container_width=True, on_click=vai_a, args=("modifica_modelli",))
        st.button("🎨 Mod. stile scheda", key="btn_mod_scheda", use_container_width=True, on_click=vai_a, args=("stile_scheda",))
        st.button("🏋️‍♂️ Allenamento Assistito", key="btn_allenamento", use_container_width=True, on_click=vai_a, args=("allenamento_assistito",))

# ===============================================================
# PAGINA: Gestione PDF esistenti
# ===============================================================
if st.session_state.page == "home":
    st.markdown("### Mini gestionali   ")

    with st.expander("📂 Gestione dei PDF installa/elimina", expanded=False):
        pdf_files = [f for f in os.listdir() if f.endswith(".pdf")]

        if not pdf_files:
            st.info("Nessun file PDF trovato nella cartella.")
        else:
            for pdf in pdf_files:
                col1, col2, col3 = st.columns([6, 1, 1])

                with col1:
                    st.markdown(f"📄 **{pdf}**")

                with col2:
                    with open(pdf, "rb") as f:
                        st.download_button(
                            label="📥",
                            data=f,
                            file_name=pdf,
                            mime="application/pdf",
                            key=f"download_{pdf}"
                        )

                with col3:
                    if st.button("🗑️", key=f"delete_{pdf}"):
                        os.remove(pdf)
                        #st.success(f"{pdf} eliminato.")
                        st.rerun()


# ===============================================================
# PAGINA 1.2: Home - GALLERIA IMMAGINI
# ===============================================================
    IMG_DIR = "IMG"
    os.makedirs(IMG_DIR, exist_ok=True)
    #st.markdown("### Gestione Immagini")

    with st.expander("🖼️ Gestione Immagini nella Cartella IMG", expanded=False):
        immagini = [f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]

        if immagini:
            n_colonne = 3  # Fisso a 3 colonne, Streamlit le adatta su mobile

            for i in range(0, len(immagini), n_colonne):
                cols = st.columns(n_colonne)
                for col, img in zip(cols, immagini[i:i + n_colonne]):
                    with col:
                        st.image(os.path.join(IMG_DIR, img), use_container_width=True)
                        if st.button("🗑️ Elimina", key=f"del_{img}"):
                            os.remove(os.path.join(IMG_DIR, img))
                            st.success(f"Immagine '{img}' eliminata.")
                            st.rerun()
        else:
            st.info("Nessuna immagine trovata nella cartella IMG.")

# ===============================================================
# PAGINA: IA
# ===============================================================
    #--- UI per l'inserimento della Chiave API ---
    st.sidebar.title("Configurazione API")
    api_key_input = st.sidebar.text_input(
        "Inserisci qui la tua Chiave API di Google Gemini:",
        type="password", # Nasconde il testo per sicurezza
        help="Puoi ottenere la tua chiave API su Google AI Studio (https://makersuite.google.com/app)."
    )

    model = None
    api_configured = False

    if api_key_input:
        try:
            genai.configure(api_key=api_key_input)
            model = genai.GenerativeModel('gemini-1.5-flash') # Modello aggiornato
            st.sidebar.success("Chiave API configurata con successo!")
            api_configured = True
        except Exception as e:
            st.sidebar.error(f"Errore durante la configurazione dell'API: {e}")
            api_configured = False
    else:
        st.sidebar.warning("Inserisci la tua Chiave API di Google Gemini per iniziare la chat.")
        api_configured = False


    # === Funzione per interagire con Google Gemini ===
    def generate_reply_gemini(messages, current_model):
        try:
            system_instruction = ""
            if messages and messages[0]["role"] == "system":
                system_instruction = messages[0]["content"]
                chat_messages = messages[1:]
            else:
                chat_messages = messages
            
            formatted_messages = []
            for msg in chat_messages:
                if msg["role"] == "user":
                    formatted_messages.append({'role':'user', 'parts': [msg["content"]]})
                elif msg["role"] == "assistant":
                    formatted_messages.append({'role':'model', 'parts': [msg["content"]]})

            chat = current_model.start_chat(history=formatted_messages)
            
            if formatted_messages:
                # Assicurati che l'ultimo messaggio sia dell'utente per inviarlo
                if formatted_messages[-1]['role'] == 'user':
                    response = chat.send_message(formatted_messages[-1]['parts'][0])
                    return response.text.strip()
                else:
                    return "Errore: L'ultimo messaggio nella cronologia non è dell'utente."
            else:
                return "Nessun messaggio utente da inviare."
                
        except Exception as e:
            return f"Errore API Gemini: {str(e)}"


    # --- Inizializza la memoria della chat ---
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "system",
                "content": (
                    "Sei un esperto personal trainer e nutrizionista virtuale. "
                    "Rispondi sempre in italiano in modo professionale, chiaro e amichevole. "
                    "Dai consigli utili su dieta, esercizio fisico e benessere."
                )
            }
        ]

    # === UI Chat ===

    with st.expander("💬 Chat con il Personal Trainer IA", expanded=False):
        # Mostra la conversazione storica
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"**Tu:** {msg['content']}")
            elif msg["role"] == "assistant":
                st.markdown(f"**Personal Trainer IA:** {msg['content']}")
            # Il messaggio di sistema non viene mostrato direttamente nella chat

        # --- Nuovo approccio con st.form ---
        # Creiamo un form per gestire l'invio dell'input dell'utente
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input(
                "Scrivi qualcosa al tuo personal trainer:",
                key="user_input_box",
                disabled=not api_configured
            )
            submit_button = st.form_submit_button(label='Invia', disabled=not api_configured)

            if submit_button and user_input and api_configured:
                # Aggiungi il messaggio dell'utente alla session_state
                st.session_state.messages.append({"role": "user", "content": user_input})
                
                # Genera la risposta usando Gemini
                reply = generate_reply_gemini(st.session_state.messages, model)
                
                # Aggiungi la risposta dell'assistente alla session_state
                st.session_state.messages.append({"role": "assistant", "content": reply})
                
                # Ricarica l'applicazione per mostrare i nuovi messaggi
                st.rerun() # Ancora necessario per aggiornare la visualizzazione della chat




# ===============================================================
# PAGINA INFO CLIENTE - Versione Database Compatibile
# ===============================================================

if st.session_state.page == "info_cliente":
    from database_helper import DatabaseManager
    
    # Inizializza il database manager
    db = DatabaseManager()
    
    st.title("Generatore di Scheda Palestra")

    # Campi di input utente
    nome = st.text_input("Nome")
    sesso = st.radio("Sesso", ["Uomo", "Donna"])
    livello = st.selectbox("Livello", ["Principiante", "Intermedio", "Avanzato"])
    obiettivo = st.selectbox("Obiettivo", ["Dimagrimento", "Massa", "Definizione"])
    giorni = st.slider("Quanti giorni a settimana ti alleni?", 1, 7, 3)

    # Salva le informazioni in session_state per uso successivo
    st.session_state.nome = nome
    st.session_state.sesso = sesso
    st.session_state.livello = livello
    st.session_state.obiettivo = obiettivo
    st.session_state.giorni = giorni

    # Carica modelli esercizi dal database invece che dal JSON
    try:
        modelli_esercizi = db.get_modelli_esercizi()
        st.session_state.modelli_esercizi = modelli_esercizi
        
        # Debug info (opzionale)
        if st.checkbox("Mostra info debug", key="debug_modelli"):
            st.write(f"Modelli caricati dal database: {len(modelli_esercizi)} categorie")
            for sesso_key in modelli_esercizi.keys():
                st.write(f"• {sesso_key}")
        
    except Exception as e:
        st.error(f"Errore nel caricamento modelli dal database: {e}")
        # Fallback: prova a caricare dal JSON se il database fallisce
        st.warning("Tentativo di caricamento da file JSON...")
        try:
            import json
            with open("modelli.json", "r") as f:
                modelli_esercizi = json.load(f)
            st.session_state.modelli_esercizi = modelli_esercizi
            st.info("Modelli caricati da file JSON di backup")
        except:
            st.error("Impossibile caricare i modelli esercizi")

    # Colonne per i pulsanti di navigazione
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Indietro", key="btn_indietro_2", on_click=vai_a, args=("home",)):
            pass

    with col2:
        # Controlla che i modelli siano stati caricati prima di procedere
        if nome and sesso and livello and obiettivo:
            if st.button("Avanti", key="btn_avanti_2", on_click=vai_a, args=("selezione_giorni",)):
                # Salva le informazioni del cliente nel database prima di procedere
                try:
                    conn = sqlite3.connect("throng.db")
                    cursor = conn.cursor()
                    
                    # Inserisci o aggiorna il cliente
                    cursor.execute('''
                        INSERT OR REPLACE INTO clienti 
                        (nome, sesso, livello, obiettivo, giorni_settimana)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (nome, sesso, livello, obiettivo, giorni))
                    
                    conn.commit()
                    conn.close()
                    
                except Exception as e:
                    st.warning(f"Errore nel salvataggio cliente: {e}")
        else:
            st.button("Avanti", key="btn_avanti_2", disabled=True)
            if not nome:
                st.warning("Inserisci il nome per continuare")

# ===============================================================
# PAGINA 2: Selezione dei giorni della settimana / creazione pdf - Versione Database
# ===============================================================

elif st.session_state.page == "selezione_giorni":
    from database_helper import DatabaseManager
    
    db = DatabaseManager()
    
    st.title("Seleziona i giorni della settimana in cui ti alleni")

    giorni_settimana = ["Lunedì", "Martedì", "Mercoledì", "Giovedì", "Venerdì", "Sabato", "Domenica"]

    giorni_selezionati = st.multiselect(
        f"Scegli i giorni (esattamente {st.session_state.giorni}):",
        giorni_settimana,
        key="giorni_settimana_key"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Indietro", key="btn_indietro", on_click=vai_a, args=("info_cliente",)):
            pass

    with col2:
        if st.button("Home", key="btn_home_giorni", on_click=vai_a, args=("home",)):
            pass

    if len(giorni_selezionati) != st.session_state.giorni:
        st.warning(f"Seleziona esattamente {st.session_state.giorni} giorni.")
    else:
        # Carica esercizi dal database invece che da session_state
        try:
            esercizi_base = db.get_esercizi(
                st.session_state.sesso, 
                st.session_state.livello, 
                st.session_state.obiettivo
            )
            
            if not esercizi_base:
                st.error(f"Nessun esercizio trovato per: {st.session_state.sesso}, {st.session_state.livello}, {st.session_state.obiettivo}")
                
                # Opzione per migrare dati dal JSON se il database è vuoto
                if st.button("🔄 Importa esercizi da JSON"):
                    if db.migra_da_json():
                        st.success("Esercizi importati dal JSON!")
                        st.rerun()
                    else:
                        st.error("File modelli.json non trovato")
            else:
                esercizi_per_giorno = {}

                # Prepara la lista stringhe per la multiselect
                lista_esercizi_str = [f"{es['nome']} ({es['serie']})" for es in esercizi_base]

                st.markdown("### Assegna esercizi ai giorni selezionati")

                for giorno in giorni_selezionati:
                    st.markdown(f"**{giorno}**")
                    esercizi_scelti_str = st.multiselect(
                        f"Seleziona esercizi per {giorno}",
                        lista_esercizi_str,
                        key=f"esercizi_{giorno}"
                    )
                    # Mappa le stringhe selezionate ai dizionari originali
                    esercizi_scelti = [es for es in esercizi_base if f"{es['nome']} ({es['serie']})" in esercizi_scelti_str]
                    esercizi_per_giorno[giorno] = esercizi_scelti

                if st.button("Genera scheda", key="genera_scheda"):
                    with st.spinner("Generazione scheda in corso..."):
                        esercizi_finali = []
                        for giorno in giorni_selezionati:
                            esercizi_finali.extend(esercizi_per_giorno.get(giorno, []))

                        if not esercizi_finali:
                            st.error("Devi selezionare almeno un esercizio.")
                        else:
                        # Salva la scheda nel database prima di generare il PDF
                            try:
                                conn = sqlite3.connect("throng.db")
                                cursor = conn.cursor()
                            
                            # Trova l'ID del cliente (assume che sia stato salvato in precedenza)
                                cursor.execute('''
                                    SELECT id FROM clienti 
                                    WHERE nome = ? AND sesso = ? AND livello = ? AND obiettivo = ?
                                    ORDER BY created_at DESC LIMIT 1
                                ''', (st.session_state.nome, st.session_state.sesso, 
                                    st.session_state.livello, st.session_state.obiettivo))
                            
                                cliente_row = cursor.fetchone()
                                if cliente_row:
                                    cliente_id = cliente_row[0]
                                
                                    # Elimina esercizi precedenti per questo cliente
                                    cursor.execute('DELETE FROM schede_esercizi WHERE cliente_id = ?', (cliente_id,))
                                
                                    # Inserisci i nuovi esercizi
                                    for giorno, esercizi_giorno in esercizi_per_giorno.items():
                                        for ordine, esercizio in enumerate(esercizi_giorno):
                                            # Trova l'ID dell'esercizio
                                            cursor.execute('''
                                                SELECT id FROM esercizi 
                                                WHERE nome = ? AND sesso = ? AND livello = ? AND obiettivo = ?
                                            ''', (esercizio['nome'], st.session_state.sesso, 
                                                st.session_state.livello, st.session_state.obiettivo))
                                        
                                            esercizio_row = cursor.fetchone()
                                            if esercizio_row:
                                                esercizio_id = esercizio_row[0]
                                                cursor.execute('''
                                                    INSERT INTO schede_esercizi 
                                                    (cliente_id, giorno, esercizio_id, ordine)
                                                    VALUES (?, ?, ?, ?)
                                                ''', (cliente_id, giorno, esercizio_id, ordine))
                                
                                    conn.commit()
                                    st.success("Scheda salvata nel database!")
                            
                                conn.close()
                            
                            except Exception as e:
                                st.warning(f"Errore nel salvataggio scheda: {e}")
                        
                        # Genera il PDF (assumendo che la funzione crea_scheda_pdf esista)
                        try:
                            filename = crea_scheda_pdf(
                                st.session_state.nome,
                                st.session_state.livello,
                                st.session_state.obiettivo,
                                esercizi_per_giorno
                            )

                            with open(filename, "rb") as f:
                                st.download_button("📄 Scarica PDF", f, file_name=filename, key="btn_carica_pdf")
                                
                        except Exception as e:
                            st.error(f"Errore nella generazione PDF: {e}")
                        
        except Exception as e:
            st.error(f"Errore nel caricamento esercizi: {e}")

# ===============================================================
# FUNZIONE AGGIUNTIVA: Visualizza schede salvate
# ===============================================================

def mostra_schede_salvate():
    """Funzione opzionale per visualizzare le schede salvate nel database"""
    st.title("📋 Schede Salvate")
    
    db = DatabaseManager()
    
    try:
        conn = sqlite3.connect("throng.db")
        cursor = conn.cursor()
        
        # Recupera tutti i clienti
        cursor.execute('''
            SELECT c.id, c.nome, c.sesso, c.livello, c.obiettivo, c.giorni_settimana, c.created_at
            FROM clienti c
            ORDER BY c.created_at DESC
        ''')
        
        clienti = cursor.fetchall()
        
        if not clienti:
            st.info("Nessuna scheda salvata")
        else:
            for cliente in clienti:
                cliente_id, nome, sesso, livello, obiettivo, giorni, created_at = cliente
                
                with st.expander(f"🏋️ {nome} - {sesso} {livello} ({obiettivo})"):
                    st.write(f"**Creata il:** {created_at}")
                    st.write(f"**Giorni settimana:** {giorni}")
                    
                    # Recupera gli esercizi per questo cliente
                    cursor.execute('''
                        SELECT se.giorno, e.nome, e.serie
                        FROM schede_esercizi se
                        JOIN esercizi e ON se.esercizio_id = e.id
                        WHERE se.cliente_id = ?
                        ORDER BY se.giorno, se.ordine
                    ''', (cliente_id,))
                    
                    esercizi_scheda = cursor.fetchall()
                    
                    if esercizi_scheda:
                        giorni_dict = {}
                        for giorno, nome_es, serie in esercizi_scheda:
                            if giorno not in giorni_dict:
                                giorni_dict[giorno] = []
                            giorni_dict[giorno].append(f"{nome_es} ({serie})")
                        
                        for giorno, esercizi in giorni_dict.items():
                            st.write(f"**{giorno}:**")
                            for esercizio in esercizi:
                                st.write(f"  • {esercizio}")
                    else:
                        st.write("Nessun esercizio associato")
        
        conn.close()
        
    except Exception as e:
        st.error(f"Errore nel caricamento schede: {e}")

# Puoi aggiungere questa funzione come nuova pagina:
# if st.session_state.page == "schede_salvate":
#     mostra_schede_salvate()

# ===============================================================
# PAGINA 3: Modifica esercizi (MODIFICATA PER USARE IL DATABASE)
# ===============================================================

IMG_DIR = "IMG"
os.makedirs(IMG_DIR, exist_ok=True)

if st.session_state.page == "modifica_modelli":
    st.title("Modifica Modelli Esercizi")

    sesso = st.selectbox("Seleziona sesso", ["Uomo", "Donna"])
    livello = st.selectbox("Seleziona livello", ["Principiante", "Intermedio", "Avanzato"])
    obiettivo = st.selectbox("Seleziona obiettivo", ["Dimagrimento", "Massa", "Definizione"])

    # Carica esercizi dal database
    esercizi_correnti = db.get_esercizi(sesso, livello, obiettivo)

    st.write(f"Esercizi attuali per {sesso} - {livello} - {obiettivo}:")
    st.markdown("---")  # 🔹 Aggiunge linea divisoria
    for es in esercizi_correnti:
        immagine = es.get("immagine", "").strip()
        if immagine:
            img_path = os.path.join(IMG_DIR, immagine)
            if os.path.isfile(img_path):
                st.image(img_path, width=100)
            else:
                st.warning(f"Immagine non trovata: {img_path}")
        st.write(f"**{es['nome']}** - Serie: {es['serie']}")
        st.markdown("---")  # 🔹 Aggiunge linea divisoria

    # Reset automatico dei campi al rerun se necessario
    if st.session_state.get("reset_aggiunta", False):
        st.session_state["nuovo_nome"] = ""
        st.session_state["nuove_serie"] = ""
        #st.session_state["nuova_img_file"] = ""
        st.session_state["reset_aggiunta"] = False

    # Ottieni le immagini già caricate nella cartella IMG_DIR
    immagini_esistenti = [img for img in os.listdir(IMG_DIR) if img.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))]
    immagini_esistenti.insert(0, "")  # opzione vuota (nessuna immagine selezionata)

    # ======= AGGIUNGI =======
    with st.expander("➕ Aggiungi nuovo esercizio", expanded=st.session_state.get("espandi_aggiunta", False)):
        nuovo_nome = st.text_input("Nome esercizio", key="nuovo_nome")
        nuove_serie = st.text_input("Serie es: 3x12", key="nuove_serie")
        
        col1, col2 = st.columns(2)
        with col1:
            nuova_img_file = st.file_uploader("📁 Carica immagine (opzionale)", type=["png", "jpg", "jpeg", "gif"], key="nuova_img_file")
        with col2:
            immagine_selezionata = st.selectbox("🖼️ Seleziona immagine esistente", immagini_esistenti, key="img_selezionata")

        if st.button("Aggiungi esercizio", key="btn_aggiungi_esercizio"):
            if nuovo_nome and nuove_serie:
                immagine_filename = ""
                if nuova_img_file is not None:
                    timestamp = int(time.time())
                    ext = os.path.splitext(nuova_img_file.name)[1]
                    immagine_filename = f"{timestamp}_{nuovo_nome.replace(' ', '_')}{ext}"
                    salva_path = os.path.join(IMG_DIR, immagine_filename)
                    with open(salva_path, "wb") as f:
                        f.write(nuova_img_file.getbuffer())
                elif immagine_selezionata:
                    immagine_filename = immagine_selezionata

                successo = db.aggiungi_esercizio(sesso, livello, obiettivo, nuovo_nome, nuove_serie, immagine_filename)
                
                if successo:
                    st.session_state.modelli_esercizi = db.get_modelli_esercizi()
                    st.success("Esercizio aggiunto con successo!")
                    st.session_state["reset_aggiunta"] = True
                    st.rerun()
                else:
                    st.error("Errore: Esercizio già esistente!")
            else:
                st.warning("Compila nome e serie.")

    # ======= MODIFICA =======
    with st.expander("✏️ Modifica esercizio", expanded=st.session_state.get("espandi_modifica", False)):
        if esercizi_correnti:
            nomi_esercizi = [es["nome"] for es in esercizi_correnti]
            esercizio_da_modificare = st.selectbox("Seleziona esercizio da modificare", nomi_esercizi)

            esercizio_corrente = next((es for es in esercizi_correnti if es["nome"] == esercizio_da_modificare), None)

            if esercizio_corrente:
                nuovo_nome = st.text_input("Nuovo nome", value=esercizio_corrente["nome"])
                nuove_serie = st.text_input("Nuove serie es: 3x12", value=esercizio_corrente["serie"])

                st.write("Immagine attuale:")
                if esercizio_corrente.get("immagine"):
                    img_path = os.path.join(IMG_DIR, esercizio_corrente["immagine"])
                    if os.path.isfile(img_path):
                        st.image(img_path, width=100)
                    else:
                        st.warning(f"Immagine non trovata: {img_path}")

                col1, col2 = st.columns(2)
                with col1:
                    nuova_img_file = st.file_uploader("📁 Carica nuova immagine", type=["png", "jpg", "jpeg", "gif"])
                with col2:
                    immagine_selezionata = st.selectbox("🖼️ Oppure seleziona immagine esistente", immagini_esistenti)

                if st.button("Salva modifiche esercizio"):
                    nuova_immagine = esercizio_corrente.get("immagine", "")

                    def safe_filename(name):
                        return re.sub(r'[^a-zA-Z0-9_\-]', '_', name)

                    if nuova_img_file is not None:
                        timestamp = int(time.time())
                        ext = os.path.splitext(nuova_img_file.name)[1]
                        immagine_filename = f"{timestamp}_{safe_filename(nuovo_nome)}{ext}"
                        salva_path = os.path.join(IMG_DIR, immagine_filename)
                        with open(salva_path, "wb") as f:
                            f.write(nuova_img_file.getbuffer())
                        nuova_immagine = immagine_filename
                    elif immagine_selezionata:
                        nuova_immagine = immagine_selezionata

                    db.modifica_esercizio(sesso, livello, obiettivo, esercizio_da_modificare, nuovo_nome, nuove_serie, nuova_immagine)
                    st.session_state.modelli_esercizi = db.get_modelli_esercizi()
                    st.success("Esercizio modificato con successo.")
                    st.rerun()

    # ======= RIMUOVI =======
    with st.expander("➖ Rimuovi esercizio", expanded=st.session_state.get("espandi_rimozione", False)):
        if esercizi_correnti:
            nomi_esercizi = [es["nome"] for es in esercizi_correnti]
            esercizio_da_rimuovere = st.selectbox("Seleziona esercizio", nomi_esercizi)

            if st.button("Rimuovi esercizio selezionato"):
                # Rimuovi dal database
                db.rimuovi_esercizio(sesso, livello, obiettivo, esercizio_da_rimuovere)
                
                # Aggiorna session_state per compatibilità
                st.session_state.modelli_esercizi = db.get_modelli_esercizi()
                st.success("Esercizio rimosso con successo.")
                st.rerun()
        else:
            st.info("Nessun esercizio disponibile per la rimozione.")

    if st.button("Torna alla home", key="torna_home_3", on_click=vai_a, args=("home",)):
        pass


# ===============================================================
# PAGINA 4: Modifica stile scheda (modifica_schede/logo) - Versione Database
# ===============================================================

if st.session_state.page == "stile_scheda":
    from database_helper import DatabaseManager
    
    # Inizializza il database manager
    db = DatabaseManager()

    # Funzione per creare anteprima delimitatore
    def crea_anteprima_delimitatore(stile):
        img = Image.new("RGB", (300, 40), "white")
        draw = ImageDraw.Draw(img)

        if stile == "Linea semplice":
            draw.line((10, 20, 290, 20), fill="black", width=2)
        elif stile == "Linea tratteggiata":
            for x in range(10, 290, 10):
                draw.line((x, 20, x+5, 20), fill="black", width=2)
        elif stile == "Doppia linea":
            draw.line((10, 16, 290, 16), fill="black", width=1)
            draw.line((10, 24, 290, 24), fill="black", width=1)
        elif stile == "Rettangolo sottile":
            draw.rectangle((10, 18, 290, 22), fill="gray")

        return img

    st.title("Impostazioni Stili")

    # Carica le impostazioni dal database
    settings = db.get_settings()

    logo_path = "static/logo_scheda.png"

    # Mostra logo corrente (se esiste)
    if os.path.isfile(logo_path):
        st.image(logo_path, width=150)

    # Carica nuovo logo
    uploaded_logo = st.file_uploader("Carica nuovo logo", type=["png", "jpg", "jpeg"])
    if uploaded_logo:
        # Assicurati che la directory static esista
        os.makedirs("static", exist_ok=True)
        with open(logo_path, "wb") as f:
            f.write(uploaded_logo.read())
        st.success("Logo aggiornato con successo!")

    st.markdown("### Dimensione logo")

    dimimm = st.slider(
        "Scegli dimensione immagine (px: Default = 500 x 500)",
        min_value=100,
        max_value=1000,
        value=settings.get("dimimm", 500),
        step=50
    )
    st.write(f"Dimensione immagine selezionata: {dimimm} x {dimimm} px")

    st.markdown("### Dimensione immagini esercizi")

    dimimmes = st.slider(
        "Scegli dimensione immagine (valori da: 1 a 8)",
        min_value=1,
        max_value=8,
        value=settings.get("dimimmes", 4),
        step=1
    )
    st.write(f"Dimensione immagini esercizi: {dimimmes} px")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Colore sfondo 🎨")
        colorbg = st.color_picker(
            "Scegli colore sfondo",
            value=settings.get("colore_sfondo", "#ffffff")
        )
        st.write("Colore sfondo selezionato:", colorbg)

    with col2:
        st.markdown("### Colore testo 🎨")
        colortxt = st.color_picker(
            "Scegli colore testo",
            value=settings.get("colore_testo", "#000000")
        )
        st.write("Colore testo selezionato:", colortxt)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Colore titolo 🎨")
        colortit = st.color_picker(
            "Scegli colore titoli",
            value=settings.get("colore_titoli", "#ffffff")
        )
        st.write("Colore titolo selezionato:", colortit)

    with col2:
        st.markdown("### Colore giorni della sett. 🎨")
        colorgset = st.color_picker(
            "Scegli colore giorni",
            value=settings.get("colore_g_settimana", "#000000")
        )
        st.write("Colore giorni sett. selezionato:", colorgset)

    st.markdown("### Fonts")

    font_options = ["Helvetica", "Helvetica-Bold", "Times-Roman", "Times-Bold", "Courier", "Courier-Bold"]

    selected_font = st.selectbox(
        "Seleziona font per il PDF",
        font_options,
        index=font_options.index(settings.get("font", "Helvetica"))
        if settings.get("font") in font_options else 0
    )
    st.write(f"Hai selezionato il font: {selected_font}")

    ############################################################################

    st.markdown("### Dimensione titoli")

    dimtitoli = st.number_input(
        "Scegli dimensione titoli (pt)",
        min_value=6,
        max_value=48,
        value=settings.get("dimtitoli", 18),
        step=1,
        key="numinput_dimtitoli"
    )
    st.write(f"Dimensione titoli: {dimtitoli} pt")

    st.markdown("### Dimensione giorni settimana")

    dimgset = st.number_input(
        "Scegli dimensione giorni della settimana (pt)",
        min_value=6,
        max_value=48,
        value=settings.get("dimgset", 14),
        step=1,
        key="numinput_dimgset"
    )
    st.write(f"Dimensione giorni della settimana: {dimgset} pt")

    st.markdown("### Dimensione testi")

    dimtesti = st.number_input(
        "Scegli dimensione testi (pt)",
        min_value=6,
        max_value=48,
        value=settings.get("dimtesti", 12),
        step=1,
        key="numinput_dimtesti"
    )
    st.write(f"Dimensione testi: {dimtesti} pt")

##################################################

    stili_delimitatore = [
        "Linea semplice",
        "Linea tratteggiata",
        "Doppia linea",
        "Rettangolo sottile"
    ]

    st.markdown("### ✨ Selettore stile delimitatore PDF")

    cols = st.columns(2)
    for idx, stile in enumerate(stili_delimitatore):
        anteprima_img = crea_anteprima_delimitatore(stile)
        buffer = io.BytesIO()
        anteprima_img.save(buffer, format="PNG")
        cols[idx % 2].image(buffer.getvalue(), caption=stile, use_container_width=True)

    selezionato = st.selectbox(
        "Stile delimitatore",
        stili_delimitatore,
        label_visibility="collapsed",
        index=stili_delimitatore.index(settings.get("stile_delimitatore", "Linea semplice"))
        if settings.get("stile_delimitatore") in stili_delimitatore else 0
    )

    st.session_state["stile_delimitatore"] = selezionato

    st.success(f"Hai selezionato: {selezionato}")

    # Bottone per salvare le impostazioni
    if st.button("Salva impostazioni"):
        settings_to_save = {
            "dimimm": dimimm,
            "dimimmes": dimimmes,
            "colore_sfondo": colorbg,
            "colore_testo": colortxt,
            "colore_titoli": colortit,
            "colore_g_settimana": colorgset,
            "font": selected_font,
            "stile_delimitatore": selezionato,
            "dimtitoli": dimtitoli,
            "dimgset": dimgset,
            "dimtesti": dimtesti
        }
        
        # Salva nel database invece che nel file JSON
        db.salva_settings(settings_to_save)
        st.success("Impostazioni salvate correttamente nel database!")

    # Sezione per migrazione dati (opzionale, da mostrare solo se necessario)
    with st.expander("🔄 Migrazione dati da file JSON"):
        st.info("Usa questa sezione solo se hai impostazioni salvate nel vecchio formato JSON")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Migra settings da JSON"):
                if db.migra_settings_da_json():
                    st.success("Settings migrati con successo!")
                    st.rerun()  # Ricarica la pagina per mostrare i nuovi settings
                else:
                    st.warning("File settings.json non trovato")
        
        with col2:
            if st.button("Esporta settings in JSON"):
                # Funzione per esportare le impostazioni attuali in formato JSON
                current_settings = db.get_settings()
                os.makedirs("SETTINGS", exist_ok=True)
                with open("SETTINGS/settings_backup.json", "w") as f:
                    json.dump(current_settings, f, indent=4)
                st.success("Settings esportati in SETTINGS/settings_backup.json")

    if st.button("Torna alla home", key="torna_home_4", on_click=vai_a, args=("home",)):
        pass
