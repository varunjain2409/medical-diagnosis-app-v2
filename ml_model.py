import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ─── SYMPTOM & DISEASE DATASET ────────────────────────────────────────────────

ALL_SYMPTOMS = [
    "fever", "cough", "headache", "fatigue", "nausea", "vomiting",
    "diarrhea", "chest_pain", "shortness_of_breath", "sore_throat",
    "runny_nose", "body_ache", "chills", "rash", "dizziness",
    "abdominal_pain", "loss_of_appetite", "joint_pain", "swollen_lymph_nodes",
    "night_sweats", "weight_loss", "jaundice", "dark_urine", "pale_stool",
    "frequent_urination", "excessive_thirst", "blurred_vision", "numbness",
    "back_pain", "constipation", "bloating", "heartburn", "sneezing",
    "itchy_eyes", "skin_rash", "palpitations", "anxiety", "insomnia"
]

DISEASE_DATA = {
    "Common Cold": {
        "symptoms": ["runny_nose", "sore_throat", "sneezing", "cough", "headache", "fatigue"],
        "severity": "mild",
        "recommendations": "Rest, stay hydrated, take OTC cold medicine. See a doctor if symptoms worsen after 10 days.",
        "doctors": ["General Physician", "ENT Specialist"]
    },
    "Influenza (Flu)": {
        "symptoms": ["fever", "cough", "body_ache", "headache", "chills", "fatigue", "sore_throat"],
        "severity": "moderate",
        "recommendations": "Rest, fluids, antiviral medications if caught early. Seek medical care if fever is very high.",
        "doctors": ["General Physician", "Pulmonologist"]
    },
    "COVID-19": {
        "symptoms": ["fever", "cough", "shortness_of_breath", "fatigue", "body_ache", "headache", "loss_of_appetite"],
        "severity": "moderate",
        "recommendations": "Isolate immediately, get tested. Seek emergency care if breathing is difficult.",
        "doctors": ["General Physician", "Pulmonologist", "Infectious Disease Specialist"]
    },
    "Typhoid Fever": {
        "symptoms": ["fever", "headache", "abdominal_pain", "nausea", "fatigue", "loss_of_appetite", "constipation"],
        "severity": "severe",
        "recommendations": "Antibiotics required. Visit a doctor immediately. Avoid self-medication.",
        "doctors": ["General Physician", "Gastroenterologist", "Infectious Disease Specialist"]
    },
    "Malaria": {
        "symptoms": ["fever", "chills", "headache", "nausea", "vomiting", "fatigue", "body_ache", "night_sweats"],
        "severity": "severe",
        "recommendations": "Urgent medical care required. Blood test needed to confirm. Antimalarial drugs prescribed by doctor.",
        "doctors": ["General Physician", "Infectious Disease Specialist"]
    },
    "Dengue Fever": {
        "symptoms": ["fever", "headache", "body_ache", "rash", "joint_pain", "fatigue", "nausea", "vomiting"],
        "severity": "severe",
        "recommendations": "No specific antiviral. Stay hydrated, monitor platelet count. Visit doctor urgently.",
        "doctors": ["General Physician", "Infectious Disease Specialist"]
    },
    "Gastroenteritis": {
        "symptoms": ["nausea", "vomiting", "diarrhea", "abdominal_pain", "fever", "fatigue"],
        "severity": "moderate",
        "recommendations": "ORS solution, rest, bland diet. Seek care if diarrhea lasts more than 3 days.",
        "doctors": ["Gastroenterologist", "General Physician"]
    },
    "Pneumonia": {
        "symptoms": ["fever", "cough", "shortness_of_breath", "chest_pain", "fatigue", "chills", "body_ache"],
        "severity": "severe",
        "recommendations": "Antibiotics required. Hospital admission may be needed. See a doctor immediately.",
        "doctors": ["Pulmonologist", "General Physician"]
    },
    "Diabetes (Type 2)": {
        "symptoms": ["frequent_urination", "excessive_thirst", "fatigue", "blurred_vision", "weight_loss", "numbness"],
        "severity": "chronic",
        "recommendations": "Blood sugar testing required. Lifestyle changes, possible medication. See an endocrinologist.",
        "doctors": ["Endocrinologist", "Diabetologist", "General Physician"]
    },
    "Jaundice / Hepatitis": {
        "symptoms": ["jaundice", "dark_urine", "pale_stool", "fatigue", "abdominal_pain", "nausea", "loss_of_appetite"],
        "severity": "severe",
        "recommendations": "Liver function tests needed. Avoid alcohol, fatty foods. See a doctor immediately.",
        "doctors": ["Hepatologist", "Gastroenterologist", "General Physician"]
    },
    "Hypertension": {
        "symptoms": ["headache", "dizziness", "chest_pain", "palpitations", "blurred_vision", "fatigue"],
        "severity": "chronic",
        "recommendations": "Blood pressure monitoring needed. Lifestyle changes and medication. See a cardiologist.",
        "doctors": ["Cardiologist", "General Physician"]
    },
    "Allergic Rhinitis": {
        "symptoms": ["sneezing", "runny_nose", "itchy_eyes", "skin_rash", "headache", "fatigue"],
        "severity": "mild",
        "recommendations": "Antihistamines, avoid allergens. See an allergist for skin/blood allergy testing.",
        "doctors": ["Allergist", "ENT Specialist"]
    },
    "Anxiety / Panic Disorder": {
        "symptoms": ["palpitations", "anxiety", "insomnia", "dizziness", "chest_pain", "shortness_of_breath", "fatigue"],
        "severity": "moderate",
        "recommendations": "Therapy (CBT), breathing exercises. Medication if needed. See a mental health professional.",
        "doctors": ["Psychiatrist", "Psychologist", "General Physician"]
    },
    "GERD / Acid Reflux": {
        "symptoms": ["heartburn", "chest_pain", "bloating", "nausea", "abdominal_pain", "constipation"],
        "severity": "mild",
        "recommendations": "Avoid spicy/fatty foods, don't lie down after eating. Antacids may help. See a gastroenterologist.",
        "doctors": ["Gastroenterologist", "General Physician"]
    }
}

MODEL_PATH = "ml/disease_model.pkl"
ENCODER_PATH = "ml/label_encoder.pkl"

# ─── BUILD TRAINING DATA ──────────────────────────────────────────────────────

def build_training_data():
    X, y = [], []
    symptom_index = {s: i for i, s in enumerate(ALL_SYMPTOMS)}

    for disease, info in DISEASE_DATA.items():
        base = [0] * len(ALL_SYMPTOMS)
        for s in info["symptoms"]:
            if s in symptom_index:
                base[symptom_index[s]] = 1

        # Augment with partial symptom combos for robustness
        import random
        random.seed(42)
        for _ in range(30):
            vec = base.copy()
            # Randomly drop 1-2 symptoms
            drop = random.sample(info["symptoms"], min(2, len(info["symptoms"]) - 2))
            for s in drop:
                if s in symptom_index:
                    vec[symptom_index[s]] = 0
            # Randomly add 1 noise symptom
            noise = random.choice(ALL_SYMPTOMS)
            vec[symptom_index[noise]] = 1
            X.append(vec)
            y.append(disease)

    return np.array(X), np.array(y)

# ─── TRAIN MODEL ─────────────────────────────────────────────────────────────

def train_model():
    os.makedirs("ml", exist_ok=True)
    X, y = build_training_data()
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=12)
    clf.fit(X_train, y_train)

    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"[ML] Model trained. Accuracy: {acc:.2%}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    return clf, le

def load_model():
    if not os.path.exists(MODEL_PATH):
        return train_model()
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    return clf, le

# ─── PREDICT FUNCTION ─────────────────────────────────────────────────────────

def predict_disease(symptoms_list):
    """
    symptoms_list: list of symptom strings e.g. ["fever", "cough", "headache"]
    Returns: dict with disease, confidence, severity, recommendations
    """
    clf, le = load_model()
    symptom_index = {s: i for i, s in enumerate(ALL_SYMPTOMS)}

    vec = [0] * len(ALL_SYMPTOMS)
    matched = []
    for s in symptoms_list:
        s_clean = s.strip().lower().replace(" ", "_")
        if s_clean in symptom_index:
            vec[symptom_index[s_clean]] = 1
            matched.append(s_clean)

    if sum(vec) == 0:
        return {
            "disease": "Unknown",
            "confidence": 0.0,
            "severity": "unknown",
            "recommendations": "No recognizable symptoms found. Please consult a doctor.",
            "matched_symptoms": [],
            "doctors": ["General Physician"]
        }

    proba = clf.predict_proba([vec])[0]
    top_idx = np.argsort(proba)[::-1][:3]
    top_diseases = [(le.inverse_transform([i])[0], round(proba[i] * 100, 1)) for i in top_idx]

    best_disease, best_conf = top_diseases[0]
    info = DISEASE_DATA.get(best_disease, {})

    return {
        "disease": best_disease,
        "confidence": best_conf,
        "severity": info.get("severity", "unknown"),
        "recommendations": info.get("recommendations", "Please consult a general physician."),
        "matched_symptoms": matched,
        "top_predictions": top_diseases,
        "doctors": info.get("doctors", ["General Physician"])
    }

def get_all_symptoms():
    return ALL_SYMPTOMS

def normalize_symptoms(raw_text):
    """Parse comma-separated symptoms from user input."""
    return [s.strip().lower().replace(" ", "_") for s in raw_text.split(",") if s.strip()]
