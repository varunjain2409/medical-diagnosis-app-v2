import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "medical_app.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            role TEXT DEFAULT 'patient',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            age INTEGER,
            gender TEXT,
            blood_group TEXT,
            allergies TEXT,
            medical_history TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS diagnoses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER NOT NULL,
            symptoms TEXT NOT NULL,
            predicted_disease TEXT,
            confidence REAL,
            severity TEXT,
            recommendations TEXT,
            ai_report TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        );

        CREATE TABLE IF NOT EXISTS symptoms_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            diagnosis_id INTEGER,
            symptom_name TEXT NOT NULL,
            severity_level INTEGER DEFAULT 1,
            FOREIGN KEY (diagnosis_id) REFERENCES diagnoses(id)
        );

        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            specialization TEXT,
            hospital TEXT,
            contact TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)

    # Seed default admin user
    admin_hash = hashlib.sha256("admin123".encode()).hexdigest()
    cursor.execute("""
        INSERT OR IGNORE INTO users (username, password_hash, full_name, email, role)
        VALUES ('admin', ?, 'Administrator', 'admin@mediai.com', 'admin')
    """, (admin_hash,))

    conn.commit()
    conn.close()
    print("[DB] Database initialized successfully.")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ─── USER FUNCTIONS ───────────────────────────────────────────────────────────

def register_user(username, password, full_name, email, role="patient"):
    conn = get_connection()
    try:
        conn.execute("""
            INSERT INTO users (username, password_hash, full_name, email, role)
            VALUES (?, ?, ?, ?, ?)
        """, (username, hash_password(password), full_name, email, role))
        user_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        conn.execute("INSERT INTO patients (user_id) VALUES (?)", (user_id,))
        conn.commit()
        return True, "Registration successful!"
    except sqlite3.IntegrityError as e:
        return False, "Username or email already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = get_connection()
    user = conn.execute("""
        SELECT * FROM users WHERE username=? AND password_hash=?
    """, (username, hash_password(password))).fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    conn = get_connection()
    user = conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
    conn.close()
    return dict(user) if user else None

# ─── PATIENT FUNCTIONS ────────────────────────────────────────────────────────

def get_patient_profile(user_id):
    conn = get_connection()
    patient = conn.execute("""
        SELECT p.*, u.full_name, u.email, u.username
        FROM patients p JOIN users u ON p.user_id = u.id
        WHERE p.user_id = ?
    """, (user_id,)).fetchone()
    conn.close()
    return dict(patient) if patient else None

def update_patient_profile(user_id, age, gender, blood_group, allergies, medical_history):
    conn = get_connection()
    conn.execute("""
        UPDATE patients SET age=?, gender=?, blood_group=?, allergies=?, medical_history=?
        WHERE user_id=?
    """, (age, gender, blood_group, allergies, medical_history, user_id))
    conn.commit()
    conn.close()

# ─── DIAGNOSIS FUNCTIONS ──────────────────────────────────────────────────────

def save_diagnosis(user_id, symptoms, predicted_disease, confidence, severity, recommendations, ai_report):
    conn = get_connection()
    patient = conn.execute("SELECT id FROM patients WHERE user_id=?", (user_id,)).fetchone()
    if not patient:
        conn.close()
        return None
    patient_id = patient["id"]
    cursor = conn.execute("""
        INSERT INTO diagnoses (patient_id, symptoms, predicted_disease, confidence, severity, recommendations, ai_report)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (patient_id, symptoms, predicted_disease, confidence, severity, recommendations, ai_report))
    diag_id = cursor.lastrowid
    for s in symptoms.split(","):
        s = s.strip()
        if s:
            conn.execute("INSERT INTO symptoms_log (diagnosis_id, symptom_name) VALUES (?, ?)", (diag_id, s))
    conn.commit()
    conn.close()
    return diag_id

def get_patient_history(user_id, limit=20):
    conn = get_connection()
    rows = conn.execute("""
        SELECT d.* FROM diagnoses d
        JOIN patients p ON d.patient_id = p.id
        WHERE p.user_id = ?
        ORDER BY d.created_at DESC LIMIT ?
    """, (user_id, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_diagnosis_by_id(diag_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM diagnoses WHERE id=?", (diag_id,)).fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_diagnoses_admin():
    conn = get_connection()
    rows = conn.execute("""
        SELECT d.*, u.full_name, u.email
        FROM diagnoses d
        JOIN patients p ON d.patient_id = p.id
        JOIN users u ON p.user_id = u.id
        ORDER BY d.created_at DESC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]
