<div align="center">

<a href="https://github.com/varunjain2409/medical-diagnosis-app-v2"><img src="https://img.shields.io/badge/GitHub-varunjain2409%2Fmedical--diagnosis--app--v2-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
<a href="https://github.com/varunjain2409/medical-diagnosis-app-v2/stargazers"><img src="https://img.shields.io/github/stars/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Stars"/></a>
<a href="https://github.com/varunjain2409/medical-diagnosis-app-v2/network/members"><img src="https://img.shields.io/github/forks/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Forks"/></a>
<a href="https://github.com/varunjain2409/medical-diagnosis-app-v2/commits"><img src="https://img.shields.io/github/last-commit/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Last commit"/></a>
<a href="https://github.com/varunjain2409/medical-diagnosis-app-v2/issues"><img src="https://img.shields.io/github/issues/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Issues"/></a>
<a href="https://github.com/varunjain2409/medical-diagnosis-app-v2/graphs/contributors"><img src="https://img.shields.io/github/contributors/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Contributors"/></a>
<img src="https://img.shields.io/github/repo-size/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Size"/>
<img src="https://img.shields.io/github/languages/top/varunjain2409/medical-diagnosis-app-v2?style=for-the-badge&logo=github&logoColor=white" alt="Top language"/>
<img src="https://img.shields.io/badge/License-MIT-00C2A8?style=for-the-badge&logo=open%20source%20initiative&logoColor=white" alt="License"/>
<img src="https://img.shields.io/github/actions/workflow/status/varunjain2409/medical-diagnosis-app-v2/main?style=flat&logo=githubactions&logoColor=white&label=CI" alt="CI"/>

<img src="https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python-3.x-blue?logo=python"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask-000000?logo=flask"/>
<img src="https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white" alt="SQLite-003B57?logo=sqlite"/>
<img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="scikit--learn-F7931E?logo=scikitlearn"/>
<img src="https://img.shields.io/badge/Bootstrap-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white" alt="Bootstrap-7952B3?logo=bootstrap"/>
<img src="https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white" alt="HTML5-E34F26?logo=html5"/>
<img src="https://img.shields.io/badge/Jinja-E43D2B?style=for-the-badge&logo=jinja&logoColor=white" alt="Jinja-E43D2B?logo=jinja"/>

</div>

---

# 🏥 MediAI – AI-Powered Medical Diagnosis Assistant

A full-stack college project built with **Python + SQL + AI/ML**.

## 🗂️ Project Structure

```
medical_diagnosis_app/
├── app.py              # Flask backend (main entry point)
├── database.py         # SQLite DB schema & query functions
├── ml_model.py         # Random Forest ML disease classifier
├── ai_report.py        # AI report generator (OpenAI or rule-based)
├── requirements.txt    # Python dependencies
├── templates/
│   ├── base.html       # Shared layout with sidebar & navbar
│   ├── index.html      # Landing page
│   ├── login.html      # Login form
│   ├── register.html   # Registration form
│   ├── dashboard.html  # Main dashboard
│   ├── diagnose.html   # Symptom input & diagnosis page ⭐
│   ├── history.html    # Past diagnosis records
│   ├── profile.html    # Patient profile editor
│   ├── report.html     # Full AI report viewer
│   └── admin.html      # Admin panel
└── ml/                 # Auto-created: stores trained model
    ├── disease_model.pkl
    └── label_encoder.pkl
```

## 🚀 Setup & Run

### 1. Install Python 3.8+

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Add OpenAI API Key for AI reports
```bash
# Windows
set OPENAI_API_KEY=your_key_here

# Mac/Linux
export OPENAI_API_KEY=your_key_here
```
> Without the key, the app still works with rule-based reports.

### 4. Run the app
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

### 6. Login with demo admin
- Username: `admin`
- Password: `admin123`

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| Frontend | HTML5, Bootstrap 5, Jinja2 |
| Backend | Python, Flask |
| Database | SQLite (via sqlite3) |
| ML Model | scikit-learn Random Forest |
| AI Reports | OpenAI GPT / Rule-based fallback |
| Auth | SHA-256 hashing, Flask sessions |

## 🗄️ Database Schema

```sql
users         -- Authentication (username, password_hash, role)
patients      -- Medical profile (age, gender, blood_group, allergies)
diagnoses     -- Diagnosis records (symptoms, predicted_disease, confidence)
symptoms_log  -- Individual symptom records per diagnosis
doctors       -- Doctor profiles (for major project extension)
```

## 🤖 ML Model

- Algorithm: Random Forest (200 trees)
- Training: 14 diseases × 30 augmented samples each = 420 records
- Features: 38 binary symptom features
- Output: Disease + confidence probability

## 💊 Supported Diseases (14)

Common Cold, Influenza, COVID-19, Typhoid, Malaria, Dengue,
Gastroenteritis, Pneumonia, Diabetes Type 2, Jaundice/Hepatitis,
Hypertension, Allergic Rhinitis, Anxiety/Panic Disorder, GERD

---

## 📈 Upgrade to Major Project

| Feature | How to Add |
|---|---|
| Voice input | Web Speech API + JS |
| PDF reports | `reportlab` library |
| More diseases | Add to `DISEASE_DATA` in ml_model.py |
| Doctor login | Add role-based views |
| Appointment booking | New DB table + UI |
| Charts/Analytics | Chart.js in dashboard |
| Email notifications | Flask-Mail |
| Deploy online | Render.com / Railway (free) |

---

## ⚠️ Disclaimer

This is a student project for educational purposes.
Not a substitute for professional medical diagnosis.


---

<div align="center">

**Built by [Varun Jain](https://github.com/varunjain2409) - [Portfolio](https://varunjain2409.github.io/portfolio/) - [LinkedIn](https://www.linkedin.com/in/varunjain2409)**

Found this useful? Please star the repo on [GitHub](https://github.com/varunjain2409/medical-diagnosis-app-v2). Thanks!

</div>
