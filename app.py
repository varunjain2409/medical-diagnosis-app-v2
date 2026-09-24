from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
from functools import wraps
import json
from database import (
    init_db, register_user, login_user, get_patient_profile,
    update_patient_profile, save_diagnosis, get_patient_history,
    get_diagnosis_by_id, get_all_diagnoses_admin
)
from ml_model import predict_disease, normalize_symptoms, get_all_symptoms, train_model
from ai_report import generate_ai_report

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-key")

# ─── DECORATORS ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if session.get("role") != "admin":
            flash("Admin access required.", "danger")
            return redirect(url_for("dashboard"))
        return f(*args, **kwargs)
    return decorated

# ─── AUTH ROUTES ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_id" in session:
        return redirect(url_for("dashboard"))
    return render_template("index.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip()

        if not all([username, password, full_name, email]):
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if len(password) < 6:
            flash("Password must be at least 6 characters.", "danger")
            return render_template("register.html")

        success, msg = register_user(username, password, full_name, email)
        if success:
            flash(msg + " Please log in.", "success")
            return redirect(url_for("login"))
        else:
            flash(msg, "danger")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user = login_user(username, password)
        if user:
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            session["full_name"] = user["full_name"]
            session["role"] = user["role"]
            flash(f"Welcome back, {user['full_name']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("index"))

# ─── MAIN ROUTES ─────────────────────────────────────────────────────────────

@app.route("/dashboard")
@login_required
def dashboard():
    history = get_patient_history(session["user_id"], limit=5)
    profile = get_patient_profile(session["user_id"])
    return render_template("dashboard.html", history=history, profile=profile)

@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    if request.method == "POST":
        update_patient_profile(
            user_id=session["user_id"],
            age=request.form.get("age"),
            gender=request.form.get("gender"),
            blood_group=request.form.get("blood_group"),
            allergies=request.form.get("allergies", ""),
            medical_history=request.form.get("medical_history", "")
        )
        flash("Profile updated successfully!", "success")
        return redirect(url_for("profile"))
    patient = get_patient_profile(session["user_id"])
    return render_template("profile.html", patient=patient)

@app.route("/diagnose", methods=["GET", "POST"])
@login_required
def diagnose():
    all_symptoms = get_all_symptoms()
    result = None

    if request.method == "POST":
        raw_symptoms = request.form.get("symptoms", "")
        selected = request.form.getlist("symptom_checkboxes")

        # Combine manually entered + checkbox selected symptoms
        typed = normalize_symptoms(raw_symptoms)
        all_input = list(set(typed + selected))

        if not all_input:
            flash("Please enter or select at least one symptom.", "warning")
            return render_template("diagnose.html", all_symptoms=all_symptoms)

        prediction = predict_disease(all_input)
        patient = get_patient_profile(session["user_id"])
        report = generate_ai_report(patient or {}, all_input, prediction)

        diag_id = save_diagnosis(
            user_id=session["user_id"],
            symptoms=", ".join(all_input),
            predicted_disease=prediction["disease"],
            confidence=prediction["confidence"],
            severity=prediction["severity"],
            recommendations=prediction["recommendations"],
            ai_report=report
        )

        result = {
            "prediction": prediction,
            "report": report,
            "diag_id": diag_id,
            "symptoms": all_input
        }

    return render_template("diagnose.html", all_symptoms=all_symptoms, result=result)

@app.route("/history")
@login_required
def history():
    records = get_patient_history(session["user_id"])
    return render_template("history.html", records=records)

@app.route("/report/<int:diag_id>")
@login_required
def view_report(diag_id):
    diag = get_diagnosis_by_id(diag_id)
    if not diag:
        flash("Report not found.", "danger")
        return redirect(url_for("history"))
    return render_template("report.html", diag=diag)

# ─── ADMIN ROUTES ─────────────────────────────────────────────────────────────

@app.route("/admin")
@login_required
@admin_required
def admin():
    all_diagnoses = get_all_diagnoses_admin()
    return render_template("admin.html", all_diagnoses=all_diagnoses)

# ─── API ENDPOINTS (for AJAX) ─────────────────────────────────────────────────

@app.route("/api/quick-diagnose", methods=["POST"])
@login_required
def api_quick_diagnose():
    data = request.get_json()
    symptoms = data.get("symptoms", [])
    if not symptoms:
        return jsonify({"error": "No symptoms provided"}), 400
    result = predict_disease(symptoms)
    return jsonify(result)

@app.route("/api/symptoms")
def api_symptoms():
    return jsonify(get_all_symptoms())

# ─── INIT & RUN ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    print("[ML] Training model...")
    train_model()
    print("[App] Starting Medical Diagnosis Assistant...")
    app.run(debug=True, host="0.0.0.0", port=5000)
