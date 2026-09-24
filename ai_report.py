import os
import json
from datetime import datetime

# Try importing openai; fall back gracefully if not installed/key missing
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


def generate_ai_report(patient_info, symptoms, prediction_result):
    """
    Generate a detailed medical report using AI.
    Falls back to a rule-based report if OpenAI is unavailable.
    """
    if OPENAI_AVAILABLE and OPENAI_API_KEY:
        return _openai_report(patient_info, symptoms, prediction_result)
    else:
        return _rule_based_report(patient_info, symptoms, prediction_result)


def _openai_report(patient_info, symptoms, prediction):
    """Generate report via OpenAI GPT."""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        prompt = f"""
You are a medical AI assistant. Generate a professional but easy-to-understand preliminary medical report.

Patient Info:
- Name: {patient_info.get('full_name', 'N/A')}
- Age: {patient_info.get('age', 'N/A')}
- Gender: {patient_info.get('gender', 'N/A')}
- Blood Group: {patient_info.get('blood_group', 'N/A')}
- Known Allergies: {patient_info.get('allergies', 'None')}
- Medical History: {patient_info.get('medical_history', 'None')}

Reported Symptoms: {', '.join(symptoms)}
AI Predicted Condition: {prediction['disease']} (Confidence: {prediction['confidence']}%)
Severity: {prediction['severity']}

Write a structured report with:
1. Summary
2. Possible Diagnosis
3. What this condition means
4. Recommended next steps
5. Lifestyle advice
6. When to seek emergency care

Keep it clear, empathetic, and non-alarmist. End with a disclaimer that this is an AI-generated preliminary report and not a substitute for professional medical advice.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800
        )
        return response.choices[0].message.content
    except Exception as e:
        return _rule_based_report(patient_info, symptoms, prediction) + f"\n\n[Note: AI report unavailable: {str(e)}]"


def _rule_based_report(patient_info, symptoms, prediction):
    """Fallback rule-based report generation."""
    severity_emoji = {"mild": "🟡", "moderate": "🟠", "severe": "🔴", "chronic": "🔵", "unknown": "⚪"}
    severity_advice = {
        "mild": "You can manage this at home with rest and OTC medication. Monitor symptoms closely.",
        "moderate": "We recommend scheduling a doctor visit within 24–48 hours.",
        "severe": "Please seek medical attention as soon as possible. This condition requires professional evaluation.",
        "chronic": "This appears to be a chronic condition. Regular medical supervision and lifestyle changes are essential.",
        "unknown": "Please consult a general physician for a proper evaluation."
    }

    now = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    sev = prediction.get("severity", "unknown")
    emoji = severity_emoji.get(sev, "⚪")

    top_preds = prediction.get("top_predictions", [(prediction["disease"], prediction["confidence"])])
    alt_str = ""
    if len(top_preds) > 1:
        alts = [f"{d} ({c}%)" for d, c in top_preds[1:]]
        alt_str = f"\n**Alternative possibilities:** {', '.join(alts)}"

    report = f"""
╔══════════════════════════════════════════════╗
   AI MEDICAL DIAGNOSIS REPORT
   Generated: {now}
╚══════════════════════════════════════════════╝

PATIENT INFORMATION
───────────────────
Name        : {patient_info.get('full_name', 'N/A')}
Age         : {patient_info.get('age', 'N/A')} years
Gender      : {patient_info.get('gender', 'N/A')}
Blood Group : {patient_info.get('blood_group', 'N/A')}
Allergies   : {patient_info.get('allergies', 'None reported')}

REPORTED SYMPTOMS
─────────────────
{chr(10).join(f"  • {s.replace('_', ' ').title()}" for s in symptoms)}

AI DIAGNOSIS RESULT
───────────────────
Primary Condition : {prediction['disease']}
Confidence        : {prediction['confidence']}%
Severity Level    : {emoji} {sev.upper()}
{alt_str}

RECOMMENDATIONS
───────────────
{prediction['recommendations']}

URGENCY LEVEL
─────────────
{severity_advice.get(sev, severity_advice['unknown'])}

SPECIALIST REFERRAL
───────────────────
Suggested Doctors: {', '.join(prediction.get('doctors', ['General Physician']))}

─────────────────────────────────────────────
⚠️  DISCLAIMER: This is an AI-generated preliminary report based on symptom analysis.
It is NOT a substitute for professional medical diagnosis or treatment.
Please consult a qualified healthcare professional before making any medical decisions.
─────────────────────────────────────────────
"""
    return report.strip()
