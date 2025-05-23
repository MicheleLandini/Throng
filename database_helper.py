import sqlite3
import json
import os
from typing import Dict, List, Any

class DatabaseManager:
    def __init__(self, db_path: str = "throng.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Inizializza il database con le tabelle necessarie"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Tabella per i modelli degli esercizi
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS esercizi (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sesso TEXT NOT NULL,
                livello TEXT NOT NULL,
                obiettivo TEXT NOT NULL,
                nome TEXT NOT NULL,
                serie TEXT NOT NULL,
                immagine TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(sesso, livello, obiettivo, nome)
            )
        ''')
        
        # Tabella per le impostazioni
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chiave TEXT UNIQUE NOT NULL,
                valore TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabella per i clienti e le loro schede
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS clienti (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nome TEXT NOT NULL,
                sesso TEXT NOT NULL,
                livello TEXT NOT NULL,
                obiettivo TEXT NOT NULL,
                giorni_settimana INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Tabella per associare esercizi ai giorni delle schede clienti
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS schede_esercizi (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cliente_id INTEGER NOT NULL,
                giorno TEXT NOT NULL,
                esercizio_id INTEGER NOT NULL,
                ordine INTEGER DEFAULT 0,
                FOREIGN KEY (cliente_id) REFERENCES clienti (id),
                FOREIGN KEY (esercizio_id) REFERENCES esercizi (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def migra_da_json(self, json_path: str = "modelli.json"):
        """Migra i dati esistenti dal file JSON al database"""
        if not os.path.exists(json_path):
            return False
            
        with open(json_path, 'r') as f:
            modelli = json.load(f)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for sesso, livelli in modelli.items():
            for livello, obiettivi in livelli.items():
                for obiettivo, esercizi in obiettivi.items():
                    for esercizio in esercizi:
                        cursor.execute('''
                            INSERT OR REPLACE INTO esercizi 
                            (sesso, livello, obiettivo, nome, serie, immagine)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            sesso, livello, obiettivo,
                            esercizio['nome'],
                            esercizio['serie'],
                            esercizio.get('immagine', '')
                        ))
        
        conn.commit()
        conn.close()
        return True
    
    def migra_settings_da_json(self, json_path: str = "SETTINGS/settings.json"):
        """Migra le impostazioni dal file JSON al database"""
        if not os.path.exists(json_path):
            return False
            
        with open(json_path, 'r') as f:
            settings = json.load(f)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chiave, valore in settings.items():
            cursor.execute('''
                INSERT OR REPLACE INTO settings (chiave, valore, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (chiave, json.dumps(valore)))
        
        conn.commit()
        conn.close()
        return True
    
    # === GESTIONE ESERCIZI ===
    def get_esercizi(self, sesso: str, livello: str, obiettivo: str) -> List[Dict]:
        """Recupera gli esercizi per una specifica combinazione"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT nome, serie, immagine FROM esercizi
            WHERE sesso = ? AND livello = ? AND obiettivo = ?
            ORDER BY nome
        ''', (sesso, livello, obiettivo))
        
        esercizi = []
        for row in cursor.fetchall():
            esercizi.append({
                'nome': row[0],
                'serie': row[1],
                'immagine': row[2] or ''
            })
        
        conn.close()
        return esercizi
    
    def get_modelli_esercizi(self) -> Dict:
        """Recupera tutti i modelli in formato compatibile con il codice esistente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT sesso, livello, obiettivo, nome, serie, immagine FROM esercizi')
        
        modelli = {}
        for row in cursor.fetchall():
            sesso, livello, obiettivo, nome, serie, immagine = row
            
            if sesso not in modelli:
                modelli[sesso] = {}
            if livello not in modelli[sesso]:
                modelli[sesso][livello] = {}
            if obiettivo not in modelli[sesso][livello]:
                modelli[sesso][livello][obiettivo] = []
            
            modelli[sesso][livello][obiettivo].append({
                'nome': nome,
                'serie': serie,
                'immagine': immagine or ''
            })
        
        conn.close()
        return modelli
    
    def aggiungi_esercizio(self, sesso: str, livello: str, obiettivo: str, 
                          nome: str, serie: str, immagine: str = ""):
        """Aggiunge un nuovo esercizio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO esercizi (sesso, livello, obiettivo, nome, serie, immagine)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (sesso, livello, obiettivo, nome, serie, immagine))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Esercizio già esistente
        finally:
            conn.close()
    
    def modifica_esercizio(self, sesso: str, livello: str, obiettivo: str,
                          nome_originale: str, nuovo_nome: str, nuove_serie: str, 
                          nuova_immagine: str = None):
        """Modifica un esercizio esistente"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if nuova_immagine is not None:
            cursor.execute('''
                UPDATE esercizi 
                SET nome = ?, serie = ?, immagine = ?
                WHERE sesso = ? AND livello = ? AND obiettivo = ? AND nome = ?
            ''', (nuovo_nome, nuove_serie, nuova_immagine, sesso, livello, obiettivo, nome_originale))
        else:
            cursor.execute('''
                UPDATE esercizi 
                SET nome = ?, serie = ?
                WHERE sesso = ? AND livello = ? AND obiettivo = ? AND nome = ?
            ''', (nuovo_nome, nuove_serie, sesso, livello, obiettivo, nome_originale))
        
        conn.commit()
        conn.close()
    
    def rimuovi_esercizio(self, sesso: str, livello: str, obiettivo: str, nome: str):
        """Rimuove un esercizio"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM esercizi
            WHERE sesso = ? AND livello = ? AND obiettivo = ? AND nome = ?
        ''', (sesso, livello, obiettivo, nome))
        
        conn.commit()
        conn.close()
    
    # === GESTIONE SETTINGS ===
    def get_settings(self) -> Dict:
        """Recupera tutte le impostazioni"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT chiave, valore FROM settings')
        
        settings = {}
        for row in cursor.fetchall():
            chiave, valore = row
            try:
                settings[chiave] = json.loads(valore)
            except json.JSONDecodeError:
                settings[chiave] = valore
        
        conn.close()
        return settings
    
    def salva_settings(self, settings: Dict):
        """Salva le impostazioni"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for chiave, valore in settings.items():
            cursor.execute('''
                INSERT OR REPLACE INTO settings (chiave, valore, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
            ''', (chiave, json.dumps(valore)))
        
        conn.commit()
        conn.close()
    
    def get_setting(self, chiave: str, default=None):
        """Recupera una singola impostazione"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT valore FROM settings WHERE chiave = ?', (chiave,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                return row[0]
        return default