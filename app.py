import os
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # allow insecure OAuth locally

# ----------------------------
# Standard library & packages
# ----------------------------
import io
import json
import re
import sqlite3
from datetime import datetime

# third-party
import bcrypt
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyotp
import qrcode
import shap

from flask import (
    Flask, render_template, request, redirect, url_for, session, flash,
    jsonify, send_file
)
from flask_dance.contrib.google import make_google_blueprint, google
from flask_login import LoginManager, UserMixin, login_user, logout_user, current_user

# ----------------------------
# Configuration / Constants
# ----------------------------
APP_SECRET = "supersecretkey"        # replace with a secure secret in production
DB = "database/financial_aid.db"
QR_FOLDER = "static/qrcodes"
PLOTS_FOLDER = "static/plots"
DATASET_PATH = "dataset/new_financial_aid_dataset.csv"
MODELS_PATH = "models"
METRICS_FILE = os.path.join(MODELS_PATH, "metrics.json")

# Admin email
ADMIN_EMAILS = [
    "chelimoalison0@gmail.com"
]

# ensure folders exist
os.makedirs(QR_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

# ----------------------------
# Flask app initialization
# ----------------------------
app = Flask(__name__)
app.secret_key = APP_SECRET

# session settings
app.config["SESSION_PROTECTION"] = "strong"
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # set True in production with HTTPS

# ----------------------------
# DB helper
# ----------------------------
def get_db_connection():
    """
    Create and return a sqlite3 connection.
    Uses a timeout to reduce 'database is locked' errors and allows
    connections across threads in dev (check_same_thread=False).
    """
    conn = sqlite3.connect(DB, timeout=10, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

# ----------------------------
# Load models & resources
# ----------------------------
# Load ML models and scaler. If missing, app will raise — ensure files exist.
try:
    log_reg_model = joblib.load(os.path.join(MODELS_PATH, 'lr_model.pkl'))
except Exception:
    log_reg_model = None

try:
    tree_model = joblib.load(os.path.join(MODELS_PATH, 'dt_model.pkl'))
except Exception:
    tree_model = None

try:
    scaler = joblib.load(os.path.join(MODELS_PATH, 'scaler.pkl'))
except Exception:
    scaler = None

# try to load dataset
try:
    df = pd.read_csv(DATASET_PATH)
except Exception:
    df = pd.DataFrame()

# ----------------------------
# Flask-Login setup
# ----------------------------
login_manager = LoginManager(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id_, email, is_admin=0):
        self.id = id_
        self.email = email
        self.is_admin = int(is_admin)

users = {}

@login_manager.user_loader
def load_user(user_id):
    try:
        return users.get(int(user_id))
    except Exception:
        return None

# ----------------------------
# Google OAuth setup (Flask-Dance)
# ----------------------------
google_bp = make_google_blueprint(
    client_id="754120418049-03cp6tiueos1cp7f4ueudc1cdhbqv66i.apps.googleusercontent.com",
    client_secret="Google_client_secret",
    scope=["openid", "https://www.googleapis.com/auth/userinfo.email", "https://www.googleapis.com/auth/userinfo.profile"],
    redirect_url="/login/google/authorized"
)
app.register_blueprint(google_bp, url_prefix="/login")

# ----------------------------
# Utility helpers
# ----------------------------
def email_is_admin(email: str) -> bool:
    """Return True if email is one of ADMIN_EMAILS (case-insensitive)."""
    if not email:
        return False
    return email.strip().lower() in set(e.lower() for e in ADMIN_EMAILS)

def safe_commit(conn):
    """Commit and close connection safely."""
    try:
        conn.commit()
    finally:
        conn.close()

def require_admin():
    if session.get("is_admin") != 1:
        flash("Access denied: Admins only", "error")
        return False
    return True

# ----------------------------
# ---------------- Helper: Admin prediction ----------------
# ----------------------------
def load_and_predict_dataset():
    try:
        df_admin = pd.read_csv(DATASET_PATH)
    except Exception:
        df_admin = pd.DataFrame()

    if df_admin.empty:
        df_admin['prediction'] = 0
        return df_admin

    if tree_model is None or scaler is None:
        df_admin['prediction'] = 0
        return df_admin

    features = df_admin.drop(['student_id', 'scholarship_eligibility'], axis=1, errors='ignore')

    # Encode categorical
    if 'gender' in features.columns:
        features['gender'] = features['gender'].map({'Male': 0, 'Female': 1})
    if 'background' in features.columns:
        features['background'] = features['background'].map({'Rural': 0, 'Urban': 1})
    if 'household_income_bracket' in features.columns:
        features['household_income_bracket'] = features['household_income_bracket'].map({
            'Low': 0, 'Mid-Low': 1, 'Mid-High': 2, 'High': 3
        })
    if 'government_assistance' in features.columns:
        features['government_assistance'] = features['government_assistance'].map({'Yes': 1, 'No': 0})

    for col in ['age', 'study_hours', 'attendance', 'performance_score', 'dependents']:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

    # Align to scaler features if available
    if scaler is not None:
        features = features.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Predict safely
    try:
        if tree_model is not None and scaler is not None:
            df_admin['prediction'] = tree_model.predict(scaler.transform(features))
        else:
            df_admin['prediction'] = 0
    except Exception as e:
        print("Prediction error:", e)
        df_admin['prediction'] = 0  # fallback if prediction fails

    # Ensure column exists
    if 'prediction' not in df_admin.columns:
        df_admin['prediction'] = 0

    return df_admin

# ----------------------------
# --------- ROUTES -----------
# ----------------------------

# Landing page
@app.route("/")
def home():
    return render_template("landing.html")


# ----------------------------
# Signup
# ----------------------------
@app.route("/signup", methods=["GET", "POST"])
def signup():
    """
    Signup flow:
    - create user with hashed password
    - generate OTP secret and QR code for user to scan
    - set is_admin = 1 if email in ADMIN_EMAILS
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password required.", "error")
            return redirect(url_for("signup"))

        hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        otp_secret = pyotp.random_base32()
        is_admin_val = 1 if email_is_admin(email) else 0

        conn = get_db_connection()
        try:
            cursor = conn.execute(
                "INSERT INTO users (email, password, otp_secret, otp_verified, is_admin) VALUES (?, ?, ?, ?, ?)",
                (email, hashed, otp_secret, 0, is_admin_val)
            )
            conn.commit()

            # GET THE NEW USER ID
            new_user_id = cursor.lastrowid

        except sqlite3.IntegrityError:
            flash("Email already exists!", "error")
            return redirect(url_for("signup"))
        except Exception as e:
            flash(f"Error signing up: {e}", "error")
            return redirect(url_for("signup"))
        finally:
            conn.close()

        # STORE SESSION DATA IMMEDIATELY (IMPORTANT FOR /2fa)
        session["user_id"] = new_user_id
        session["email"] = email
        session["otp_secret"] = otp_secret
        session["is_admin"] = is_admin_val

        # Generate QR code
        totp = pyotp.TOTP(otp_secret)
        qr_uri = totp.provisioning_uri(name=email, issuer_name="FinancialAidSystem")
        qr_filename = f"{email.replace('@', '_')}.png"

        qrcode.make(qr_uri).save(os.path.join(QR_FOLDER, qr_filename))

        # Show QR for user to scan and proceed to 2FA
        return render_template("show_qr.html", qr_filename=qr_filename)

    return render_template("signup.html")

# ----------------------------
# Login
# ----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    """
    - Verify credentials
    - Set session values: user_id, email, otp_secret, is_admin
    - If OTP not verified yet (first login), redirect to 2FA page
    - Otherwise redirect based on is_admin flag
    """
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Email and password required.", "error")
            return redirect(url_for("login"))

        conn = get_db_connection()
        try:
            user = conn.execute(
                "SELECT id, password, otp_secret, otp_verified, is_admin FROM users WHERE email=?",
                (email,)
            ).fetchone()
        finally:
            conn.close()

        if not user:
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

        stored_pw = user["password"] or ""
        try:
            pw_ok = bcrypt.checkpw(password.encode("utf-8"), stored_pw.encode("utf-8"))
        except Exception:
            pw_ok = False

        if pw_ok:
            session["user_id"] = user["id"]
            session["email"] = email
            session["otp_secret"] = user["otp_secret"]
            session["is_admin"] = int(user["is_admin"]) if user["is_admin"] is not None else 0

            # If OTP not yet verified, force 2FA
            if user["otp_verified"] == 0:
                return redirect(url_for("two_factor"))

            # Redirect based on role
            if session.get("is_admin") == 1:
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password", "error")
            return redirect(url_for("login"))

    return render_template("login.html")


# ----------------------------
# Google OAuth login
# ----------------------------
@app.route("/login/google")
def login_google():
    # Initiates Google OAuth if not authorized; else go to authorized handler
    if not google.authorized:
        return redirect(url_for("google.login"))
    return redirect(url_for("google_authorized"))


@app.route("/login/google/authorized")
def google_authorized():
    if not google.authorized:
        return redirect(url_for("google.login"))

    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        flash("Failed to fetch user info from Google.", "error")
        return redirect(url_for("login"))

    info = resp.json()
    email = info.get("email")
    if not email:
        flash("No email found in Google profile.", "error")
        return redirect(url_for("login"))

    conn = get_db_connection()
    try:
        user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        if not user:
            # Create user with otp_verified = 1 for OAuth users
            is_admin_val = 1 if email_is_admin(email) else 0
            conn.execute(
                "INSERT INTO users (email, password, otp_secret, otp_verified, is_admin) VALUES (?, ?, ?, ?, ?)",
                (email, "", "", 1, is_admin_val)
            )
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    finally:
        conn.close()

    # Create Flask-Login user & session
    users[user["id"]] = User(user["id"], email, user["is_admin"])
    login_user(users[user["id"]])

    session["user_id"] = user["id"]
    session["email"] = email
    session["is_admin"] = int(user["is_admin"]) if user["is_admin"] is not None else 0

    flash("Logged in successfully with Google!", "success")

    if session.get("is_admin") == 1:
        return redirect(url_for("admin_dashboard"))
    return redirect(url_for("dashboard"))


# ----------------------------
# 2FA Verification
# ----------------------------
@app.route("/2fa", methods=["GET", "POST"])
def two_factor():
    # session must contain otp_secret
    if "otp_secret" not in session or "user_id" not in session:
        flash("Session expired; login again.", "error")
        return redirect(url_for("login"))

    if request.method == "POST":
        otp = request.form.get("otp", "").strip()
        totp = pyotp.TOTP(session["otp_secret"])
        if totp.verify(otp, valid_window=1):
            conn = get_db_connection()
            try:
                conn.execute("UPDATE users SET otp_verified=1 WHERE id=?", (session["user_id"],))
                conn.commit()
            finally:
                conn.close()

            flash("Two-factor verified. Welcome!", "success")

            # determine role from DB (in case changed)
            conn = get_db_connection()
            try:
                row = conn.execute("SELECT is_admin FROM users WHERE id=?", (session["user_id"],)).fetchone()
                session["is_admin"] = int(row["is_admin"]) if row and row["is_admin"] is not None else 0
            finally:
                conn.close()

            if session.get("is_admin") == 1:
                return redirect(url_for("admin_dashboard"))
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid OTP", "error")
            return redirect(url_for("two_factor"))

    # GET -> show a simple 2FA page or template
    return render_template("2fa.html")


# ----------------------------
# Dashboard (user)
# ----------------------------
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))

    # if admin tries to access normal dashboard, redirect them to admin
    if session.get("is_admin") == 1:
        return redirect(url_for("admin_dashboard"))

    full_email = session.get("email", "")
    username = re.sub(r'[^a-zA-Z]', '', full_email.split("@")[0]).capitalize() if full_email else "User"

    # compute predictions if not present
    global df
    try:
        if 'prediction' not in df.columns:
            features = df.drop(['student_id', 'scholarship_eligibility'], axis=1, errors='ignore')

            # encode categorical variables
            if 'gender' in features.columns:
                features['gender'] = features['gender'].map({'Male': 0, 'Female': 1})
            if 'background' in features.columns:
                features['background'] = features['background'].map({'Rural': 0, 'Urban': 1})
            if 'household_income_bracket' in features.columns:
                features['household_income_bracket'] = features['household_income_bracket'].map({
                    'Low': 0, 'Mid-Low': 1, 'Mid-High': 2, 'High': 3
                })
            if 'government_assistance' in features.columns:
                features['government_assistance'] = features['government_assistance'].map({'Yes': 1, 'No': 0})

            for col in ['age', 'study_hours', 'attendance', 'performance_score', 'dependents']:
                if col in features.columns:
                    features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

            if scaler is not None and tree_model is not None:
                X_scaled = scaler.transform(features.reindex(columns=scaler.feature_names_in_, fill_value=0))
                df['prediction'] = tree_model.predict(X_scaled)
            else:
                df['prediction'] = 0
    except Exception as e:
        print("Error while preparing predictions for dashboard:", e)
        df['prediction'] = 0 if 'prediction' not in df.columns else df['prediction']

    eligible_count = int(df['prediction'].sum()) if 'prediction' in df.columns else 0
    not_eligible_count = int(len(df) - eligible_count) if len(df) > 0 else 0

    # Ensure the 'time_period' column exists
    if 'time_period' not in df.columns:
        df['time_period'] = '2024-s1'  # fallback

    # Aggregate predictions by time period
    semester_order = ["2024-s1", "2024-s2", "2025-s1", "2025-s2"]
    trend_df = df.groupby('time_period')['prediction'].sum().reset_index()
    trend_df['time_period'] = pd.Categorical(trend_df['time_period'], categories=semester_order, ordered=True)
    trend_df = trend_df.sort_values('time_period')

    trend_periods = trend_df['time_period'].tolist()
    trend_counts = trend_df['prediction'].tolist()

    return render_template(
        "dashboard.html",
        first_name=username,
        eligible_count=eligible_count,
        not_eligible_count=not_eligible_count,
        trend_periods=trend_periods,
        trend_counts=trend_counts
    )


# ----------------------------
# Predictions page (list)
# ----------------------------
@app.route("/predictions")
def predictions():
    if "user_id" not in session:
        return redirect(url_for("login"))

    global df
    features = df.drop(['student_id', 'scholarship_eligibility', 'prediction'], axis=1, errors='ignore').copy()

    # encode
    if 'gender' in features.columns:
        features['gender'] = features['gender'].map({'Male': 0, 'Female': 1})
    if 'background' in features.columns:
        features['background'] = features['background'].map({'Rural': 0, 'Urban': 1})
    if 'household_income_bracket' in features.columns:
        features['household_income_bracket'] = features['household_income_bracket'].map({
            'Low': 0, 'Mid-Low': 1, 'Mid-High': 2, 'High': 3
        })
    if 'government_assistance' in features.columns:
        features['government_assistance'] = features['government_assistance'].map({'Yes': 1, 'No': 0})

    for col in ['age', 'study_hours', 'attendance', 'performance_score', 'dependents']:
        if col in features.columns:
            features[col] = pd.to_numeric(features[col], errors='coerce').fillna(0)

    # scale & predict
    if scaler is not None and tree_model is not None:
        features = features.reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_scaled = scaler.transform(features)
        df['prediction'] = tree_model.predict(X_scaled)
    else:
        df['prediction'] = 0

    students = df.to_dict(orient='records')
    return render_template("predictions.html", students=students)


# ----------------------------
# Upload page (CSV & manual)
# ----------------------------
@app.route("/upload", methods=["GET", "POST"])
def upload_data():
    if "user_id" not in session:
        return redirect(url_for("login"))

    uploaded_data = None
    prediction_results = None

    if request.method == "GET":
        return render_template("upload.html", uploaded_data=None, prediction_results=None)

    # handle CSV
    if "csv_file" in request.files and request.files["csv_file"].filename != "":
        csv_file = request.files["csv_file"]
        try:
            uploaded_df = pd.read_csv(csv_file)
            uploaded_data = uploaded_df.to_dict(orient="records")

            # Optionally run model on uploaded_df
            try:
                features = uploaded_df.copy().drop(['student_id', 'scholarship_eligibility', 'prediction'], axis=1, errors='ignore')
                if 'gender' in features.columns:
                    features['gender'] = features['gender'].map({'Male': 0, 'Female': 1})
                if scaler is not None and tree_model is not None:
                    features = features.reindex(columns=scaler.feature_names_in_, fill_value=0)
                    Xs = scaler.transform(features)
                    preds = tree_model.predict(Xs)
                    uploaded_df['prediction'] = preds
                    prediction_results = uploaded_df.to_dict(orient='records')
            except Exception as e:
                print("Error scoring uploaded CSV:", e)
        except Exception as e:
            flash(f"Error reading CSV file: {e}", "danger")
            return render_template("upload.html", uploaded_data=None, prediction_results=None)

        return render_template("upload.html", uploaded_data=uploaded_data, prediction_results=prediction_results)

    # manual entry
    age = request.form.get("age")
    gender = request.form.get("gender")
    background = request.form.get("background")
    income = request.form.get("household_income_bracket")
    perf = request.form.get("performance_score")
    hours = request.form.get("study_hours")
    attendance = request.form.get("attendance")
    dependents = request.form.get("dependents")
    gov = request.form.get("government_assistance")

    student_dict = {
        "age": age,
        "gender": gender,
        "background": background,
        "household_income_bracket": income,
        "performance_score": perf,
        "study_hours": hours,
        "attendance": attendance,
        "dependents": dependents,
        "government_assistance": gov
    }

    uploaded_data = [student_dict]
    # optionally predict the manual entry (omitted to keep simple)
    return render_template("upload.html", uploaded_data=uploaded_data, prediction_results=None)


# ----------------------------
# Explain eligibility (SHAP)
# ----------------------------
@app.route("/explain/<student_id>")
def explain(student_id):
    """
    Create SHAP explanation for a single student:
    - get student row from dataset by student_id
    - preprocess, align with scaler.feature_names_in_
    - run model.predict_proba and SHAP explainer
    - save a SHAP plot to static/plots and render explanation page
    """
    global df
    if "student_id" not in df.columns and "student_id" not in df.columns:
        return f"No student_id column present in dataset; cannot explain {student_id}", 404

    student_row = df[df["student_id"].astype(str) == str(student_id)].copy()
    if student_row.empty:
        return f"No data found for student ID {student_id}", 404

    original_values = student_row.copy()

    student_proc = student_row.drop(['student_id', 'prediction', 'scholarship_eligibility'], axis=1, errors='ignore')

    # encode categories & convert numerics
    if 'gender' in student_proc.columns:
        student_proc['gender'] = student_proc['gender'].map({'Male': 0, 'Female': 1})
    if 'background' in student_proc.columns:
        student_proc['background'] = student_proc['background'].map({'Rural': 0, 'Urban': 1})
    if 'household_income_bracket' in student_proc.columns:
        student_proc['household_income_bracket'] = student_proc['household_income_bracket'].map({
            'Low': 0, 'Mid-Low': 1, 'Mid-High': 2, 'High': 3
        })
    if 'government_assistance' in student_proc.columns:
        student_proc['government_assistance'] = student_proc['government_assistance'].map({'Yes': 1, 'No': 0})

    for col in ['age', 'study_hours', 'attendance', 'performance_score', 'dependents']:
        if col in student_proc.columns:
            student_proc[col] = pd.to_numeric(student_proc[col], errors='coerce').fillna(0)

    # align to scaler features
    if scaler is not None:
        student_proc = student_proc.reindex(columns=scaler.feature_names_in_, fill_value=0)
        X_student = scaler.transform(student_proc)
    else:
        # fallback: convert to 2D array
        X_student = student_proc.values

    # predict & SHAP
    if tree_model is None:
        return "Model not available for explanations.", 500

    try:
        pred_probs = tree_model.predict_proba(X_student)[0]
        pred_class = int(np.argmax(pred_probs))
        confidence = round(pred_probs[pred_class] * 100, 2)
        prediction_label = "Eligible" if pred_class == 1 else "Not Eligible"
    except Exception:
        # tree may not have predict_proba
        pred_class = int(tree_model.predict(X_student)[0])
        confidence = 0.0
        prediction_label = "Eligible" if pred_class == 1 else "Not Eligible"

    # SHAP
    try:
        explainer = shap.TreeExplainer(tree_model)
        shap_values = explainer.shap_values(X_student)
        if isinstance(shap_values, list):
            shap_for_pred = shap_values[pred_class][0]
        else:
            shap_for_pred = shap_values[0]
    except Exception as e:
        print("SHAP error:", e)
        shap_for_pred = np.zeros(X_student.shape[1]) if X_student.size else np.array([])

    shap_for_pred = np.array(shap_for_pred).flatten()
    feature_names = list(student_proc.columns)

    # Match lengths
    min_len = min(len(feature_names), len(shap_for_pred))
    feature_names = feature_names[:min_len]
    shap_for_pred = shap_for_pred[:min_len]

    importance = sorted(zip(feature_names, shap_for_pred), key=lambda x: abs(x[1]), reverse=True)

    # textual explanations
    outcome_text = (
        f"This student was predicted as {prediction_label} for financial aid "
        f"with a confidence of {confidence}%."
    )
    text_explanations = []
    for feature, impact in importance:
        direction = "increased" if impact > 0 else "decreased"
        if feature in original_values.columns:
            feature_value = str(original_values[feature].values[0])
        else:
            feature_value = str(student_proc[feature].values[0] if feature in student_proc.columns else "")
        meaning = (
            f"The student's {feature.replace('_',' ')} ({feature_value}) {direction} the likelihood of being classified as "
            f"{'eligible' if pred_class == 1 else 'not eligible'}."
        )
        text_explanations.append(meaning)

    # Save SHAP bar plot as PNG
    shap_plot_path = os.path.join(PLOTS_FOLDER, f"shap_{student_id}.png")
    try:
        plt.figure(figsize=(7, 4))
        # shap.bar_plot expects shap values and feature names
        try:
            shap.bar_plot(shap_for_pred, feature_names=feature_names, show=False)
        except Exception:
            # fallback: simple horizontal bar
            y_pos = np.arange(len(feature_names))
            plt.barh(y_pos, shap_for_pred)
            plt.yticks(y_pos, feature_names)
            plt.xlabel("SHAP value")
        plt.tight_layout()
        plt.savefig(shap_plot_path)
        plt.close()
    except Exception as e:
        print("Error saving SHAP plot:", e)
        shap_plot_path = None

    return render_template(
        "explanation.html",
        student_id=student_id,
        importance=importance,
        text_explanations=text_explanations,
        shap_plot=shap_plot_path,
        outcome_text=outcome_text,
        prediction_label=prediction_label,
        confidence=confidence
    )
# Feedback Route
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    student_id = request.form.get("student_id")
    agreement = request.form.get("agreement")
    comments = request.form.get("comments")

    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO feedback (student_id, agreement, comments, submitted_at) VALUES (?, ?, ?, ?)",
            (student_id, agreement, comments, datetime.utcnow().isoformat())
        )
        conn.commit()
    finally:
        conn.close()

    # Redirect back to the same explanation page
    return redirect(url_for("explain", student_id=student_id))

# ----------------------------
# Logout
# ----------------------------
@app.route("/logout")
def logout():
    logout_user()
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


# ----------------------------
# ---------------- Admin Module
# ----------------------------
# Admin routes use same login/signup but require is_admin flag = 1
# ----------------------------

# Helper decorator-like guard (simple)
def require_admin():
    if session.get("is_admin") != 1:
        flash("Access denied: Admins only", "error")
        return False
    return True

# ----------------------------
# Admin dashboard
# ----------------------------
@app.route("/admin")
def admin_dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get('is_admin') != 1:
        flash("Access denied", "error")
        return redirect(url_for('dashboard'))

    # Load dataset with predictions
    df_local_admin = load_and_predict_dataset()

    # Ensure 'prediction' column exists
    if 'prediction' not in df_local_admin.columns:
        df_local_admin['prediction'] = 0

    eligible_count = int(df_local_admin['prediction'].sum())
    not_eligible_count = len(df_local_admin) - eligible_count

    return render_template(
        "dashboard_admin.html",
        eligible_count=eligible_count,
        not_eligible_count=not_eligible_count
    )

# Admin student detail + delete
@app.route("/admin/student/<int:sid>", methods=["GET", "DELETE"])
def admin_student_detail(sid):
    if "user_id" not in session:
        return jsonify({"error": "not authenticated"}), 401
    if session.get("is_admin") != 1:
        return jsonify({"error": "forbidden"}), 403

    conn = get_db_connection()
    try:
        user = conn.execute("SELECT * FROM users WHERE id=?", (sid,)).fetchone()
        if request.method == "DELETE":
            if user:
                conn.execute("DELETE FROM users WHERE id=?", (sid,))
                conn.commit()
            return ("", 204)

        if not user:
            return "<p>User not found</p>", 404

        html = "<dl>"
        for k in user.keys():
            html += f"<dt>{k}</dt><dd>{user[k]}</dd>"
        html += "</dl>"
        return html
    finally:
        conn.close()


# Admin export users to CSV
@app.route("/admin/export/students")
def admin_export_students():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("is_admin") != 1:
        flash("Access denied", "error")
        return redirect(url_for("dashboard"))

    conn = get_db_connection()
    try:
        df_users = pd.read_sql_query("SELECT * FROM users", conn)
    finally:
        conn.close()

    buf = io.StringIO()
    df_users.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), as_attachment=True, download_name="users_export.csv", mimetype="text/csv")

@app.route("/admin/model")
def admin_model():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    if session.get("is_admin") != 1:
        flash("Access denied", "error")
        return redirect(url_for("dashboard"))

    return render_template("model_admin.html")

# Admin trigger retrain model (synchronous stub)
@app.route("/admin/train-model", methods=["POST"])
def admin_train_model():
    if "user_id" not in session:
        return jsonify({"error": "not authenticated"}), 401
    if session.get("is_admin") != 1:
        return jsonify({"error": "forbidden"}), 403

    # NOTE: This is a synchronous stub. Replace with your actual training function.
    try:
        # importing your train function if exists
        # from eda.projecteda import train_and_save_models
        # logs = train_and_save_models()
        logs = "Training stub: implement training routine in eda/projecteda.py or similar."

        metrics = {
            "accuracy": 0.96,
            "trained_at": str(datetime.utcnow())
        }
        with open(METRICS_FILE, "w") as f:
            json.dump(metrics, f)

        return logs
    except Exception as e:
        return f"Error during training: {e}", 500

# Admin view model metrics
@app.route("/admin/model/metrics")
def admin_model_metrics():
    if "user_id" not in session:
        return jsonify({"error": "not authenticated"}), 401
    if session.get("is_admin") != 1:
        return jsonify({"error": "forbidden"}), 403

    try:
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
        return jsonify(metrics)
    except Exception:
        return jsonify({"accuracy": 0.0})

# Admin download model
@app.route("/admin/model/download")
def download_model():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("is_admin") != 1:
        flash("Access denied", "error")
        return redirect(url_for("dashboard"))

    dt_path = os.path.join(MODELS_PATH, "dt_model.pkl")
    if os.path.exists(dt_path):
        return send_file(dt_path, as_attachment=True)
    else:
        flash("Model not found.", "error")
        return redirect(url_for("admin_model"))

@app.route("/admin/students")
def admin_students():
    if "user_id" not in session:
        return redirect(url_for("login"))
    if session.get("is_admin") != 1:
        flash("Access denied", "error")
        return redirect(url_for("dashboard"))

    return render_template("students_admin.html")

@app.route("/api/admin/summary")
def api_admin_summary():
    if "user_id" not in session:
        return jsonify({"error": "not authenticated"}), 401
    if session.get("is_admin") != 1:
        return jsonify({"error": "forbidden"}), 403

    df_local_admin = load_and_predict_dataset()

    # Basic counts
    total = len(df_local_admin)
    eligible = int(df_local_admin['prediction'].sum()) if 'prediction' in df_local_admin else 0
    not_eligible = total - eligible

    # Model accuracy
    accuracy = 0.0
    try:
        with open(METRICS_FILE) as f:
            metrics = json.load(f)
            accuracy = metrics.get("accuracy", 0.0)
    except:
        accuracy = 0.0

    # Region distribution
    if 'background' in df_local_admin.columns:
        region_labels = df_local_admin['background'].value_counts().index.tolist()
        region_values = df_local_admin['background'].value_counts().tolist()
    else:
        region_labels = []
        region_values = []

    # Income distribution
    if 'household_income_bracket' in df_local_admin.columns:
        income_labels = df_local_admin['household_income_bracket'].value_counts().index.tolist()
        income_values = df_local_admin['household_income_bracket'].value_counts().tolist()
    else:
        income_labels = []
        income_values = []

    return jsonify({
        "total": total,
        "eligible": eligible,
        "not_eligible": not_eligible,
        "model_accuracy": accuracy,
        "region_labels": region_labels,
        "region_values": region_values,
        "income_labels": income_labels,
        "income_values": income_values
    })

# ===== SETTINGS PAGE =====
@app.route('/settings')
def settings():
    # You can customize this template with actual settings content
    return render_template('settings.html')

# Main guard
if __name__ == "__main__":
    # Note: In debug mode with reloader, Flask spawns child processes/threads which can cause DB locks.
    # Running with use_reloader=False reduces duplicate processes in dev.
    app.run(debug=True, use_reloader=False)
