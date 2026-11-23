"""Toyota Fortuner AutoAssist: Flask app (login, roles, work orders, document management, RAG chat).

Sections: setup -> security (login/sessions/CSRF) -> dashboards -> vehicles -> chat -> work orders ->
history -> documents -> users -> safety/settings -> errors -> start-up.
"""
import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")            # keeps FAISS / NumPy stable on Windows

import hashlib
import hmac
import logging
import smtplib
import ssl
import re
import secrets
import threading
from datetime import datetime, timedelta
from email.message import EmailMessage
from logging.handlers import RotatingFileHandler
from pathlib import Path

import click
from dotenv import load_dotenv
from markupsafe import Markup
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template, request,
                   send_from_directory, session, url_for)
from sqlalchemy import func, inspect, or_
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError, OperationalError, SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

import faultcodes
import rag
from llm import PLACEHOLDER_KEYS, ask_assistant
from models import (ADMIN, DEFAULT_SETTINGS, ENGINEER, ROLES, TECH, USER_STATUSES, Approval, ChatHistory,
                    Document, FaultCode, FaultCodeFile, OtpChallenge, OrderLog, RepairHistory, Setting, User, Vehicle, WorkOrder, db, get_setting,
                    set_setting)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
load_dotenv(BASE_DIR / ".env")

# ---------------------------------------------------------------------------
# Setup: configuration comes from .env (nothing secret is written in the code)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
_file_log = RotatingFileHandler(BASE_DIR / "app.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
_file_log.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(_file_log)
log = logging.getLogger("autoassist")


def database_uri():
    if os.getenv("DATABASE_URL"):                 # optional full URL (the tests use SQLite)
        return os.environ["DATABASE_URL"]
    return URL.create("mysql+pymysql", username=os.getenv("DB_USER", "root"),
                      password=os.getenv("DB_PASSWORD", ""), host=os.getenv("DB_HOST", "localhost"),
                      port=int(os.getenv("DB_PORT", "3306")), database=os.getenv("DB_NAME", "autoassist"),
                      query={"charset": "utf8mb4"}).render_as_string(hide_password=False)


def secret_key():
    key = os.getenv("SECRET_KEY", "").strip()
    if len(key) < 16 or key.lower().startswith("change"):
        log.warning("SECRET_KEY is not set in .env; using a temporary key (everyone is logged out on restart).")
        return secrets.token_hex(32)
    return key


app = Flask(__name__)
app.config.update(
    SQLALCHEMY_DATABASE_URI=database_uri(),
    SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True, "pool_recycle": 280},
    SECRET_KEY=secret_key(),
    MAX_CONTENT_LENGTH=100 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=os.getenv("SESSION_COOKIE_SECURE", "0") == "1",   # set to 1 when served over HTTPS
    PERMANENT_SESSION_LIFETIME=timedelta(days=14),                           # used only by "Remember me"
    PROCESS_IN_BACKGROUND=True,
    OCR_ENABLED=os.getenv("OCR_ENABLED", "1") == "1",
)
db.init_app(app)
UPLOAD_DIR.mkdir(exist_ok=True)

NAV = {  # (label, endpoint, Bootstrap icon)
    TECH: [("Dashboard", "dashboard", "speedometer2"), ("Vehicle Registry", "vehicles", "truck-front"),
           ("AutoAssist Chat", "chat", "chat-dots"),
           ("Fault Codes", "fault_codes", "exclamation-diamond"),
           ("My Work Orders", "orders", "clipboard-check")],
    ENGINEER: [("Dashboard", "dashboard", "speedometer2"), ("Work Order Review", "review", "card-checklist"),
               ("High-Risk Approval", "approvals", "shield-check"), ("Diagnosis and Closure", "closure", "wrench-adjustable")],
    ADMIN: [("Dashboard", "dashboard", "speedometer2"), ("Vehicle Registry", "vehicles", "truck-front"),
            ("Document Management", "documents", "file-earmark-pdf"),
            ("Fault Code Management", "fault_code_admin", "file-earmark-spreadsheet"), ("User Management", "users", "people"),
            ("Safety Rules", "safety", "shield-exclamation"), ("System Settings", "settings", "gear")],
}
# Pages that belong under another menu item (so the right item stays highlighted).
NAV_ALIAS = {"order_detail": {TECH: "orders", ENGINEER: "review"}}

READY = "Ready for engineering review"
TECH_STATUSES = ["Open", "Under review", "In progress", "On hold", READY]        # what a technician can choose
ORDER_STATUSES = ["Open", "Under review", "In progress", "On hold", READY, "Awaiting approval",
                  "Returned for evidence", "In diagnosis", "Completed", "Closed"]
RISKS = ["Unassessed", "Low", "Medium", "High"]                                  # vehicle criticality / work order risk
REVIEW_RISKS = ["Low", "Medium", "High"]                                          # the engineer's risk classification
VEHICLE_STATUSES = ["Available", "In service", "Unavailable"]
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
USERNAME_RE = re.compile(r"^[a-z0-9_.-]{3,30}$")
MAX_FAILED, LOCK_MINUTES = 5, 10
OTP_TTL_SECONDS = 120
OTP_RESEND_SECONDS = 30
OTP_MAX_ATTEMPTS = 5


def clip(value, size):
    return (value or "").strip()[:size]


def password_problem(password):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        return "Password must contain at least one letter and one number."
    return None


def ai_key_ready():
    return os.getenv("OPENROUTER_API_KEY", "").strip().lower() not in PLACEHOLDER_KEYS


def mail_settings_ready():
    return bool(os.getenv("MAIL_USERNAME", "").strip() and os.getenv("MAIL_PASSWORD", "").strip())


def otp_digest(user_id, purpose, code):
    """Return a keyed digest so a database-only leak does not reveal six-digit OTPs."""
    secret = app.config["SECRET_KEY"].encode("utf-8")
    payload = f"{user_id}:{purpose}:{code}".encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def otp_cooldown_remaining(user, purpose, now=None):
    """Seconds remaining before a new OTP can be requested for this account/purpose."""
    now = now or datetime.now()
    previous = (OtpChallenge.query.filter_by(user_id=user.id, purpose=purpose)
                .order_by(OtpChallenge.created_at.desc()).first())
    if not previous:
        return 0
    elapsed = (now - previous.created_at).total_seconds()
    return max(0, int(OTP_RESEND_SECONDS - elapsed + 0.999))


def send_email_otp(user, purpose):
    """Create a two-minute OTP challenge and email it. The raw OTP is never stored."""
    if not mail_settings_ready():
        log.error("Gmail OTP is not configured. Set MAIL_USERNAME and MAIL_PASSWORD in .env.")
        return None

    now = datetime.now()
    previous = (OtpChallenge.query.filter_by(user_id=user.id, purpose=purpose)
                .order_by(OtpChallenge.created_at.desc()).first())
    if previous and (now - previous.created_at).total_seconds() < OTP_RESEND_SECONDS:
        if not previous.consumed_at and previous.expires_at > now:
            return previous  # reuse the active code; do not send repeated emails
        return None

    code = f"{secrets.randbelow(1_000_000):06d}"
    OtpChallenge.query.filter_by(user_id=user.id, purpose=purpose).delete(synchronize_session=False)
    challenge = OtpChallenge(
        user_id=user.id,
        purpose=purpose,
        code_hash=otp_digest(user.id, purpose, code),
        created_at=now,
        expires_at=now + timedelta(seconds=OTP_TTL_SECONDS),
        attempts=0,
    )
    db.session.add(challenge)
    db.session.commit()

    username = os.getenv("MAIL_USERNAME", "").strip()
    password = os.getenv("MAIL_PASSWORD", "").replace(" ", "").strip()
    host = os.getenv("MAIL_SERVER", "smtp.gmail.com").strip()
    sender = os.getenv("MAIL_DEFAULT_SENDER", username).strip() or username
    purpose_text = "sign-in" if purpose == "login" else "password reset"
    message = EmailMessage()
    message["Subject"] = f"Toyota Fortuner AutoAssist – {purpose_text} verification code"
    message["From"] = sender
    message["To"] = user.email
    message.set_content(
        f"Hello {user.full_name},\n\n"
        f"Your Toyota Fortuner AutoAssist {purpose_text} code is: {code}\n\n"
        "This code expires in 2 minutes and can be used only once. "
        "Do not share it with anyone. If you did not request this code, ignore this email.\n\n"
        "Toyota Fortuner AutoAssist"
    )
    try:
        port = int(os.getenv("MAIL_PORT", "465"))
        if not 1 <= port <= 65535:
            raise ValueError("MAIL_PORT must be between 1 and 65535")
        timeout = 20
        use_ssl = os.getenv("MAIL_USE_SSL", "1" if port == 465 else "0") == "1"
        if use_ssl:
            with smtplib.SMTP_SSL(host, port, timeout=timeout, context=ssl.create_default_context()) as server:
                server.login(username, password)
                server.send_message(message)
        else:
            with smtplib.SMTP(host, port, timeout=timeout) as server:
                server.ehlo()
                server.starttls(context=ssl.create_default_context())
                server.ehlo()
                server.login(username, password)
                server.send_message(message)
        return challenge
    except Exception:
        log.exception("Gmail OTP delivery failed for user id %s.", user.id)
        db.session.delete(challenge)
        db.session.commit()
        return None


def check_otp_challenge(challenge, submitted_code):
    """Return an error message or None. Successful codes are consumed immediately."""
    now = datetime.now()
    if not challenge or challenge.consumed_at:
        return "This verification code is no longer valid. Request a new code."
    if challenge.expires_at <= now:
        challenge.consumed_at = now
        db.session.commit()
        return "This verification code has expired. Request a new code."
    if challenge.attempts >= OTP_MAX_ATTEMPTS:
        challenge.consumed_at = now
        db.session.commit()
        return "Too many incorrect attempts. Request a new code."

    code = (submitted_code or "").strip()
    expected = otp_digest(challenge.user_id, challenge.purpose, code)
    if not (len(code) == 6 and code.isdigit() and secrets.compare_digest(expected, challenge.code_hash)):
        challenge.attempts += 1
        if challenge.attempts >= OTP_MAX_ATTEMPTS:
            challenge.consumed_at = now
        db.session.commit()
        remaining = max(0, OTP_MAX_ATTEMPTS - challenge.attempts)
        return ("Too many incorrect attempts. Request a new code." if remaining == 0
                else f"Incorrect code. {remaining} attempt(s) remaining.")

    challenge.consumed_at = now
    db.session.commit()
    return None


def masked_email(email):
    name, separator, domain = (email or "").partition("@")
    if not separator:
        return "your registered email"
    return (name[:1] + "***@" + domain) if name else "***@" + domain


def current_otp_challenge(purpose):
    if session.get("otp_purpose") != purpose:
        return None
    challenge_id = session.get("otp_challenge_id")
    challenge = db.session.get(OtpChallenge, challenge_id) if challenge_id else None
    return challenge if challenge and challenge.purpose == purpose else None


# ---------------------------------------------------------------------------
# Security: sessions, CSRF, role checks, headers
# ---------------------------------------------------------------------------
@app.before_request
def load_user_and_check_csrf():
    g.user = None
    uid = session.get("uid")
    if uid:
        user = db.session.get(User, uid)
        if user and user.status == "active":
            g.user = user
        else:
            session.clear()                               # disabled or deleted while logged in
    if "csrf" not in session:
        session["csrf"] = secrets.token_hex(16)
    if request.method == "POST":
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token", "")
        if not secrets.compare_digest(sent, session["csrf"]):
            abort(400, "Your session expired. Please reload the page and try again.")


@app.after_request
def security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    if request.endpoint != "static" and (g.get("user") or request.endpoint in {
        "login", "register", "verify_login_otp", "forgot_password", "verify_reset_otp", "reset_password", "resend_otp"
    }):
        response.headers["Cache-Control"] = "no-store"     # avoid caching signed-in pages and OTP/password screens
    return response


@app.context_processor
def inject_globals():
    role = g.user.role if g.user else None
    endpoint = request.endpoint or ""
    active = NAV_ALIAS.get(endpoint, {}).get(role, endpoint)
    token = session.get("csrf", "")
    return {"csrf_token": token, "csrf_input": Markup(f'<input type="hidden" name="csrf_token" value="{token}">'), "me": g.user, "nav": NAV.get(role, []), "nav_active": active,
            "TECH": TECH, "ENGINEER": ENGINEER, "ADMIN": ADMIN, "TECH_STATUSES": TECH_STATUSES, "READY": READY}


@app.template_filter("dt")
def format_datetime(value):
    return value.strftime("%d %b %Y, %H:%M") if value else "-"


def login_required(*roles):
    def decorator(view):
        from functools import wraps

        @wraps(view)
        def wrapper(*args, **kwargs):
            if not g.user:
                if request.path.startswith("/api/"):
                    return jsonify({"error": "Please log in again."}), 401
                return redirect(url_for("login", next=request.path))
            if roles and g.user.role not in roles:
                log.warning("Blocked: %s (%s) tried %s", g.user.username, g.user.role, request.path)
                abort(403)
            return view(*args, **kwargs)
        return wrapper
    return decorator


def safe_next(target):
    return target if target and target.startswith("/") and not target.startswith("//") else None


@app.get("/")
def index():
    return redirect(url_for("dashboard") if g.user else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if session.get("otp_purpose") == "login" and current_otp_challenge("login"):
        return redirect(url_for("verify_login_otp"))
    if request.method == "POST":
        ident = clip(request.form.get("identifier"), 120).lower()
        password = request.form.get("password", "")
        user = User.query.filter(or_(User.username == ident, User.email == ident)).first() if ident else None
        now = datetime.now()
        if user and user.locked_until and user.locked_until > now:
            flash("Too many failed attempts. Please wait a few minutes and try again.", "error")
            log.warning("Login blocked (locked): %s", user.username)
        elif user and check_password_hash(user.password_hash, password):
            if user.status == "pending":
                flash("Your account is waiting for administrator approval.", "info")
            elif user.status != "active":
                flash("This account is disabled. Please contact the administrator.", "error")
            else:
                challenge = send_email_otp(user, "login")
                if challenge:
                    session.clear()
                    session["csrf"] = secrets.token_hex(16)
                    session["otp_challenge_id"] = challenge.id
                    session["otp_purpose"] = "login"
                    session["otp_next"] = safe_next(request.args.get("next")) or ""
                    session["otp_remember"] = request.form.get("remember") == "on"
                    flash("A six-digit verification code was sent to your registered email. It expires in 2 minutes.", "success")
                    return redirect(url_for("verify_login_otp"))
                wait_seconds = otp_cooldown_remaining(user, "login")
                if wait_seconds:
                    flash(f"Please wait {wait_seconds} seconds before requesting another sign-in code.", "info")
                else:
                    flash("We could not send the verification email. Check the Gmail SMTP settings in .env and try again.", "error")
        else:
            check_password_hash(generate_password_hash("dummy-value"), password)   # similar timing for unknown users
            if user:
                user.failed_attempts += 1
                if user.failed_attempts >= MAX_FAILED:
                    user.locked_until, user.failed_attempts = now + timedelta(minutes=LOCK_MINUTES), 0
                db.session.commit()
            log.warning("Login failed for '%s'", ident[:40])
            flash("Incorrect username/email or password.", "error")
    return render_template("auth.html", mode="login")


@app.route("/verify-login-otp", methods=["GET", "POST"])
def verify_login_otp():
    if g.user:
        return redirect(url_for("dashboard"))
    challenge = current_otp_challenge("login")
    if not challenge or challenge.consumed_at or challenge.expires_at <= datetime.now():
        session.pop("otp_challenge_id", None)
        session.pop("otp_purpose", None)
        flash("Your sign-in verification code expired or is no longer valid. Please sign in again.", "error")
        return redirect(url_for("login"))
    user = db.session.get(User, challenge.user_id)
    if not user or user.status != "active":
        session.clear()
        flash("This account is not active. Contact the administrator.", "error")
        return redirect(url_for("login"))
    if request.method == "POST":
        error = check_otp_challenge(challenge, request.form.get("otp"))
        if not error:
            user.failed_attempts, user.locked_until, user.last_login = 0, None, datetime.now()
            db.session.commit()
            destination = safe_next(session.get("otp_next")) or url_for("dashboard")
            remember = bool(session.get("otp_remember"))
            session.clear()
            session["csrf"] = secrets.token_hex(16)
            session["uid"] = user.id
            session["login_at"] = datetime.now().replace(microsecond=0).isoformat()
            session.permanent = remember
            log.info("Login completed after email OTP: %s (%s)", user.username, user.role)
            return redirect(destination)
        flash(error, "error")
        if challenge.consumed_at or challenge.expires_at <= datetime.now() or challenge.attempts >= OTP_MAX_ATTEMPTS:
            session.pop("otp_challenge_id", None)
            session.pop("otp_purpose", None)
            return redirect(url_for("login"))
    return render_template(
        "auth.html", mode="verify_login", email=masked_email(user.email),
        otp_expires_at=int(challenge.expires_at.timestamp() * 1000),
    )


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        email = clip(request.form.get("email"), 120).lower()
        user = User.query.filter_by(email=email).first() if EMAIL_RE.match(email) else None
        session.pop("otp_challenge_id", None)
        session.pop("otp_purpose", None)
        session.pop("password_reset_user_id", None)
        session.pop("password_reset_authorized_at", None)
        if user and user.status == "active":
            challenge = send_email_otp(user, "password_reset")
            if challenge:
                session["otp_challenge_id"] = challenge.id
                session["otp_purpose"] = "password_reset"
            else:
                log.warning("Password reset OTP could not be sent for user id %s.", user.id)
        # Same response for known and unknown email addresses to avoid account enumeration.
        flash("If an active account uses that email, a verification code has been sent. The code expires in 2 minutes.", "info")
        return redirect(url_for("verify_reset_otp"))
    return render_template("auth.html", mode="forgot_password")


@app.route("/verify-reset-otp", methods=["GET", "POST"])
def verify_reset_otp():
    if g.user:
        return redirect(url_for("dashboard"))
    challenge = current_otp_challenge("password_reset")
    if challenge and (challenge.consumed_at or challenge.expires_at <= datetime.now()):
        session.pop("otp_challenge_id", None)
        session.pop("otp_purpose", None)
        challenge = None
        flash("Your password-reset code expired. Request a new code.", "error")
    if request.method == "POST":
        if not challenge:
            flash("No active reset code was found. Request a new code.", "error")
            return redirect(url_for("forgot_password"))
        error = check_otp_challenge(challenge, request.form.get("otp"))
        if not error:
            user = db.session.get(User, challenge.user_id)
            if user and user.status == "active":
                session.pop("otp_challenge_id", None)
                session.pop("otp_purpose", None)
                session["password_reset_user_id"] = user.id
                session["password_reset_authorized_at"] = datetime.now().timestamp()
                return redirect(url_for("reset_password"))
            error = "This account is not active. Contact the administrator."
        flash(error, "error")
        if challenge.consumed_at or challenge.expires_at <= datetime.now() or challenge.attempts >= OTP_MAX_ATTEMPTS:
            session.pop("otp_challenge_id", None)
            session.pop("otp_purpose", None)
            return redirect(url_for("forgot_password"))
    return render_template(
        "auth.html", mode="verify_reset",
        email=(masked_email(challenge.user.email) if challenge else "your registered email"),
        otp_expires_at=(int(challenge.expires_at.timestamp() * 1000) if challenge else 0),
    )


@app.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    if g.user:
        return redirect(url_for("dashboard"))
    user_id = session.get("password_reset_user_id")
    authorized_at = session.get("password_reset_authorized_at", 0)
    if not user_id or datetime.now().timestamp() - float(authorized_at) > 600:
        session.pop("password_reset_user_id", None)
        session.pop("password_reset_authorized_at", None)
        flash("Please verify a new password-reset code first.", "error")
        return redirect(url_for("forgot_password"))
    user = db.session.get(User, user_id)
    if not user or user.status != "active":
        session.clear()
        flash("This account is not active. Contact the administrator.", "error")
        return redirect(url_for("login"))
    if request.method == "POST":
        password, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        error = "The two passwords do not match." if password != confirm else password_problem(password)
        if error:
            flash(error, "error")
        else:
            user.password_hash = generate_password_hash(password)
            user.failed_attempts, user.locked_until = 0, None
            db.session.query(OtpChallenge).filter_by(user_id=user.id).delete(synchronize_session=False)
            db.session.commit()
            session.clear()
            flash("Your password has been updated. Sign in with your new password.", "success")
            return redirect(url_for("login"))
    return render_template("auth.html", mode="reset_password")


@app.post("/resend-otp")
def resend_otp():
    purpose = session.get("otp_purpose")
    if purpose not in {"login", "password_reset"}:
        flash("Request a new verification code first.", "error")
        return redirect(url_for("login"))
    previous = current_otp_challenge(purpose)
    if not previous:
        flash("Your verification session expired. Request a new code.", "error")
        return redirect(url_for("login" if purpose == "login" else "forgot_password"))
    if (datetime.now() - previous.created_at).total_seconds() < OTP_RESEND_SECONDS:
        flash("Please wait 30 seconds before requesting another code.", "info")
        return redirect(url_for("verify_login_otp" if purpose == "login" else "verify_reset_otp"))
    user = db.session.get(User, previous.user_id)
    if not user or user.status != "active":
        flash("This account is not active.", "error")
        return redirect(url_for("login"))
    challenge = send_email_otp(user, purpose)
    if challenge:
        session["otp_challenge_id"] = challenge.id
        flash("A new verification code has been sent. The previous code is no longer valid.", "success")
    else:
        flash("The email could not be sent. Check Gmail SMTP settings in .env.", "error")
    return redirect(url_for("verify_login_otp" if purpose == "login" else "verify_reset_otp"))


def admin_exists():
    return User.query.filter_by(role=ADMIN, status="active").count() > 0


@app.route("/register", methods=["GET", "POST"])
def register():
    """Anyone can ask for an account and choose the role they need, but the account is created PENDING
    with the normal Technician permissions. Only an administrator can approve it and grant the role.
    Exception: while the system has no administrator at all, the first account becomes the administrator."""
    if g.user:
        return redirect(url_for("dashboard"))
    first_user = not admin_exists()
    form = {}
    if request.method == "POST":
        form = {k: clip(request.form.get(k), 120) for k in ("full_name", "email", "username")}
        form["email"], form["username"] = form["email"].lower(), form["username"].lower()
        password, confirm = request.form.get("password", ""), request.form.get("confirm", "")
        form["role"] = request.form.get("role", "")
        error = None
        if not first_user and form["role"] not in ROLES:
            error = "Please choose the role you need."
        elif len(form["full_name"]) < 2:
            error = "Please enter your full name."
        elif not EMAIL_RE.match(form["email"]):
            error = "Please enter a valid email address."
        elif not USERNAME_RE.match(form["username"]):
            error = "Username must be 3-30 characters: letters, numbers, dot, dash or underscore."
        elif password != confirm:
            error = "The two passwords do not match."
        else:
            error = password_problem(password)
        if not error and User.query.filter(or_(User.email == form["email"], User.username == form["username"])).first():
            error = "That email or username is already registered."
        if error:
            flash(error, "error")
        else:
            db.session.add(User(full_name=form["full_name"], email=form["email"], username=form["username"],
                                password_hash=generate_password_hash(password),
                                role=ADMIN if first_user else TECH, status="active" if first_user else "pending",
                                requested_role=ADMIN if first_user else form["role"]))
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                flash("That email or username is already registered.", "error")
                return render_template("auth.html", mode="register", form=form, roles=ROLES, first_user=first_user)
            log.info("New registration (%s): %s", "first administrator" if first_user else "pending", form["username"])
            if first_user:
                flash("Administrator account created. You can sign in now.", "success")
            else:
                flash("Account created. An administrator must approve it before you can sign in.", "success")
            return redirect(url_for("login"))
    return render_template("auth.html", mode="register", form=form, roles=ROLES, first_user=first_user)


@app.post("/logout")
def logout():
    if g.user:
        log.info("Logout: %s", g.user.username)
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Dashboards
# ---------------------------------------------------------------------------
@app.get("/dashboard")
@login_required()
def dashboard():
    role, stats, orders, extra = g.user.role, [], [], {}
    if role == TECH:
        mine = WorkOrder.query.filter_by(technician_id=g.user.id)
        stats = [(mine.filter(WorkOrder.status != "Closed").count(), "Open work orders", "clipboard-check"),
                 (mine.filter(WorkOrder.approval == "Pending").count(), "Waiting for approval", "hourglass-split"),
                 (mine.filter(WorkOrder.status == "Closed").count(), "Closed cases", "check2-circle"),
                 (ChatHistory.query.filter_by(user_id=g.user.id).count(), "Questions asked", "chat-dots")]
        orders = mine.order_by(WorkOrder.id.desc()).limit(6).all()
    elif role == ENGINEER:
        count = lambda *cond: WorkOrder.query.filter(*cond).count()
        stats = [(count(WorkOrder.status == READY), "Ready for review", "folder2-open"),
                 (count(WorkOrder.approval == "Pending"), "Approval requests", "shield-check"),
                 (count(WorkOrder.status == "In diagnosis"), "Active diagnoses", "wrench-adjustable"),
                 (count(WorkOrder.status == "Closed"), "Closed cases", "check2-circle")]
        orders = WorkOrder.query.order_by(WorkOrder.id.desc()).limit(8).all()
    else:
        docs = Document.query.order_by(Document.id.desc()).all()
        stats = [(Vehicle.query.count(), "Vehicles", "truck-front"),
                 (sum(1 for d in docs if d.status == "Indexed"), "Indexed documents", "file-earmark-check"),
                 (sum(d.chunk_count or 0 for d in docs), "Searchable passages", "search"),
                 (User.query.filter_by(status="active").count(), "Active users", "people")]
        extra = {"docs": docs[:6], "rag": rag.rag_status(), "pending_users": User.query.filter_by(status="pending").count(),
                 "failed_docs": sum(1 for d in docs if d.status in ("Failed", "Needs OCR")), "ai_ready": ai_key_ready()}
    return render_template("dashboard.html", stats=stats, orders=orders, **extra)


# ---------------------------------------------------------------------------
# Vehicle Registry: a plain record of the vehicles the workshop has registered.
# It is NOT linked to the chat or to the manuals (those work from model + generation).
# ---------------------------------------------------------------------------
@app.get("/vehicles")
@login_required(TECH, ADMIN)
def vehicles():
    q = request.args.get("q", "").strip()
    query = Vehicle.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Vehicle.asset_id.like(like), Vehicle.model.like(like), Vehicle.generation.like(like)))
    return render_template("vehicles.html", rows=query.order_by(Vehicle.asset_id).all(), q=q,
                           statuses=VEHICLE_STATUSES, risks=RISKS)


@app.post("/vehicles/save")
@login_required(TECH, ADMIN)
def vehicle_save():
    """Technicians and administrators can register a NEW vehicle. Only administrators can edit one."""
    vid = request.form.get("id", type=int)
    vehicle = db.session.get(Vehicle, vid) if vid else None
    if vid and not vehicle:
        abort(404)
    if vehicle and g.user.role != ADMIN:
        abort(403)
    f = request.form
    d = {k: clip(f.get(k), 80) for k in ("asset_id", "model", "generation", "year", "engine", "variant", "location")}
    d["generation"] = d["generation"][:40]
    d["criticality"] = f.get("criticality") if f.get("criticality") in RISKS else "Unassessed"
    d["status"] = f.get("status") if f.get("status") in VEHICLE_STATUSES else "Available"
    other = Vehicle.query.filter(Vehicle.asset_id == d["asset_id"], Vehicle.id != (vid or 0)).first()
    if not d["asset_id"] or not d["model"] or not d["generation"]:
        flash("Asset ID, model and generation are required.", "error")
    elif other:
        flash("That asset ID is already registered.", "error")
    else:
        vehicle = vehicle or Vehicle()
        for key, value in d.items():
            setattr(vehicle, key, value)
        db.session.add(vehicle)
        db.session.commit()
        log.info("Vehicle saved: %s by %s", vehicle.asset_id, g.user.username)
        flash("Vehicle registered." if not vid else "Vehicle updated.", "success")
    return redirect(url_for("vehicles"))


@app.post("/vehicles/<int:vid>/delete")
@login_required(ADMIN)
def vehicle_delete(vid):
    vehicle = db.get_or_404(Vehicle, vid)
    try:
        db.session.delete(vehicle)
        db.session.commit()
        log.info("Vehicle deleted: %s by %s", vehicle.asset_id, g.user.username)
        flash("Vehicle deleted.", "success")
    except IntegrityError:
        db.session.rollback()
        flash("This vehicle is used by work orders or history, so it cannot be deleted. Set its status to Unavailable instead.", "error")
    return redirect(url_for("vehicles"))


# ---------------------------------------------------------------------------
# AutoAssist chat (technician): the car model + generation chosen in the chat decide which manuals are searched
# ---------------------------------------------------------------------------
def chat_catalog():
    """Models and generations that have at least one INDEXED vehicle-specific manual (what the dropdowns offer)."""
    rows = db.session.query(Document.model, Document.generation).filter(
        Document.status == "Indexed", Document.applicability == "vehicle_specific").distinct().all()
    models = {}
    for model, generation in sorted(rows, key=lambda r: ((r[0] or "").lower(), (r[1] or "").lower())):
        entry = models.setdefault(rag.norm_key(model), {"model": model, "generations": {}})
        entry["generations"].setdefault(rag.norm_key(generation), generation)
    return [{"model": e["model"], "generations": list(e["generations"].values())} for e in models.values()]


def match_catalog(catalog, model, generation):
    """The stored spelling (model, generation) if this exact pair has indexed manuals, otherwise None."""
    for entry in catalog:
        if rag.norm_key(entry["model"]) == rag.norm_key(model):
            for gen in entry["generations"]:
                if rag.norm_key(gen) == rag.norm_key(generation):
                    return entry["model"], gen
    return None


def pair_docs(model, generation):
    """Indexed manuals for this exact model + generation (metadata filter, same rule the search uses)."""
    docs = Document.query.filter(Document.status == "Indexed", Document.applicability == "vehicle_specific").all()
    return [d for d in docs if rag.norm_key(d.model) == rag.norm_key(model) and rag.norm_key(d.generation) == rag.norm_key(generation)]


def chat_started_at():
    """AutoAssist starts fresh at every login: only questions asked since this login are shown.
    Older questions and answers stay in the chat_history table (they are not deleted)."""
    try:
        return datetime.fromisoformat(session.get("login_at"))
    except (TypeError, ValueError):                      # a session that was opened before this change
        now = datetime.now().replace(microsecond=0)
        session["login_at"] = now.isoformat()
        return now


@app.get("/chat")
@login_required(TECH)
def chat():
    catalog = chat_catalog()
    model, generation = request.args.get("model", ""), request.args.get("generation", "")
    if not model and session.get("chat_pair"):                     # continue with the last selection
        model, generation = session["chat_pair"]
    selected = match_catalog(catalog, model, generation) if model and generation else None
    messages, docs, vehicles_for_order = [], [], []
    if selected:
        session["chat_pair"] = list(selected)
        messages = (ChatHistory.query.filter(ChatHistory.user_id == g.user.id, ChatHistory.created_at >= chat_started_at(),
                    func.lower(ChatHistory.model) == selected[0].lower(), func.lower(ChatHistory.generation) == selected[1].lower())
                    .order_by(ChatHistory.id.desc()).limit(30).all())[::-1]
        docs = pair_docs(*selected)
        vehicles_for_order = [v for v in Vehicle.query.order_by(Vehicle.asset_id).all()
                              if rag.norm_key(v.model) == rag.norm_key(selected[0]) and rag.norm_key(v.generation) == rag.norm_key(selected[1])]
    else:
        session.pop("chat_pair", None)
    return render_template("chat.html", catalog=catalog, selected=selected, messages=messages, docs=docs,
                           order_vehicles=vehicles_for_order, ai_ready=ai_key_ready())


@app.post("/api/ask")
@login_required(TECH)
def api_ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Type a question first."}), 400
    if len(question) > 1500:
        return jsonify({"error": "Please keep the question under 1,500 characters."}), 400
    selected = match_catalog(chat_catalog(), str(payload.get("model") or ""), str(payload.get("generation") or ""))
    if not selected:                                              # also stops anyone from asking about an unlisted vehicle
        return jsonify({"error": "Select a car model and generation that has an indexed manual."}), 400
    model, generation = selected

    rules = "\n".join(f"- {risk} risk: {get_setting('safety_' + risk)}" for risk in ("High", "Medium", "Low"))
    try:
        top_k = int(get_setting("top_k", "5"))
    except ValueError:
        top_k = 5
    # Recent chat for THIS user + vehicle (since login) so follow-ups such as "Why does it happen?" are understood.
    recent = (ChatHistory.query.filter(ChatHistory.user_id == g.user.id, ChatHistory.created_at >= chat_started_at(),
              func.lower(ChatHistory.model) == model.lower(), func.lower(ChatHistory.generation) == generation.lower())
              .order_by(ChatHistory.id.desc()).limit(3).all())[::-1]
    history = [{"question": r.question, "answer": r.answer} for r in recent]
    # Only manuals that are Indexed in MySQL for this exact model + generation may be used as evidence.
    allowed_ids = {d.id for d in pair_docs(model, generation)}
    try:
        result = ask_assistant(question, model, generation, rules, top_k=top_k, history=history, allowed_doc_ids=allowed_ids)
    except Exception:
        log.exception("Chat failed")
        result = {"answer": "Something went wrong while searching the manuals. Please try again.",
                  "sources": [], "mode": "error"}
    entry = ChatHistory(user_id=g.user.id, model=model, generation=generation, question=question,
                        answer=result["answer"], sources=result["sources"], mode=result["mode"])
    db.session.add(entry)
    try:
        db.session.commit()
        result["id"] = entry.id
    except SQLAlchemyError:
        db.session.rollback()
        log.exception("Could not save chat history")
    log.info("Chat: user=%s %s/%s mode=%s query=%r", g.user.username, model, generation, result["mode"], result.get("query"))
    return jsonify(result)


# ---------------------------------------------------------------------------
# Work orders
# ---------------------------------------------------------------------------
def tech_statuses(order):
    """What a technician may choose. Approving and closing stay with the engineer; 'In progress' stays locked
    for a High-risk case until the engineer approves it."""
    statuses = list(TECH_STATUSES)
    if order.risk == "High" and order.approval != "Approved":
        statuses.remove("In progress")
    return statuses


def tech_can_change_status(order):
    """The status belongs to the engineer while a case is awaiting approval, in diagnosis, completed or closed."""
    return order.status in TECH_STATUSES or order.status == "Returned for evidence"


def add_log(order, text):
    db.session.add(OrderLog(order=order, user_id=g.user.id, text=text[:1200]))


def order_list(*conditions):
    """Work orders filtered by the ?q= search and ?status= filter in the URL."""
    query = WorkOrder.query.outerjoin(Vehicle).filter(*conditions)
    q, status = request.args.get("q", "").strip(), request.args.get("status", "")
    if q:
        like = f"%{q}%"
        query = query.filter(or_(WorkOrder.code.like(like), WorkOrder.issue.like(like), WorkOrder.model.like(like),
                                 Vehicle.asset_id.like(like)))
    if status in ORDER_STATUSES:
        query = query.filter(WorkOrder.status == status)
    return query.order_by(WorkOrder.id.desc()).all()


def order_payload(o):
    """The fields the approval and closure pop-ups show (sent to the page as JSON, shown with textContent)."""
    return {"code": o.code, "vehicle": f"{o.model}, {o.generation}" + (f" · {o.vehicle.asset_id}" if o.vehicle else ""),
            "issue": o.issue, "technician": o.technician.full_name, "priority": o.priority, "status": o.status,
            "risk": o.risk, "approval": o.approval, "findings": o.findings or "", "actions_taken": o.actions_taken or "",
            "parts": o.parts or "", "hours": o.hours or 0, "review_gap": o.review_gap or "", "scope": o.scope or "",
            "diagnosis": o.diagnosis or "", "verification": o.verification or "",
            "outcome": o.outcome or "", "closed": o.status == "Closed",
            "detail_url": url_for("order_detail", code=o.code), "approval_url": url_for("order_approval", code=o.code),
            "closure_url": url_for("order_closure", code=o.code)}


def orders_page(rows, title, subtitle, new=False, show_approval=False, popup=None):
    """popup: 'approval' or 'closure' makes a click on an order open a pop-up instead of the detail page."""
    vehicles_list = Vehicle.query.order_by(Vehicle.asset_id).all() if new else []
    payloads = {o.code: order_payload(o) for o in rows} if popup else {}
    return render_template("orders.html", orders=rows, title=title, subtitle=subtitle, show_new=new,
                           show_approval=show_approval, statuses=ORDER_STATUSES, vehicles=vehicles_list,
                           popup=popup, payloads=payloads)


@app.get("/orders")
@login_required(TECH)
def orders():
    return orders_page(order_list(WorkOrder.technician_id == g.user.id), "My Work Orders",
                       "Your cases: findings, actions, parts and time.", new=True)


@app.get("/review")
@login_required(ENGINEER)
def review():
    return orders_page(order_list(), "Work Order Review",
                       "Open a case to review the technician's findings, classify the risk and note any evidence gap.")


@app.get("/closure")
@login_required(ENGINEER)
def closure():
    """Cases that are ready for a final diagnosis: reviewed (Low/Medium), approved High-risk, or already closed."""
    rows = order_list(or_(WorkOrder.status.in_(["In diagnosis", "Completed", "Closed"]),
                          (WorkOrder.risk == "High") & (WorkOrder.approval == "Approved")))
    return orders_page(rows, "Diagnosis and Closure", "Open a case to record the final diagnosis and verification, then close it.",
                       popup="closure")


@app.get("/approvals")
@login_required(ENGINEER)
def approvals():
    return orders_page(order_list(WorkOrder.risk == "High"), "High-Risk Approval",
                       "Open a case to approve, reject or return it. Work cannot start until you approve.",
                       show_approval=True, popup="approval")


def read_tech_fields(form, status_default):
    """The five technician fields. Returns (values, error)."""
    status = form.get("status") or status_default
    if status not in TECH_STATUSES:
        return None, "Choose a valid case status."
    try:
        hours = round(max(0.0, float(form.get("hours") or 0)), 2)
    except ValueError:
        return None, "Hours spent must be a number."
    if hours > 1000:
        return None, "Hours spent looks too large."
    values = {"status": status, "findings": clip(form.get("findings"), 2200), "actions_taken": clip(form.get("actions_taken"), 2200),
              "parts": clip(form.get("parts"), 300) or "None documented", "hours": hours}
    if status == READY and not values["findings"]:
        return None, "Write your findings before sending the case for engineering review."
    return values, None


@app.post("/orders/new")
@login_required(TECH)
def order_new():
    """Create a work order.
    From the chat: model + generation come from the chat answer, a registered vehicle is optional.
    From My Work Orders: a registered vehicle is chosen and its model + generation are used.
    The technician also records the case status, findings, actions taken, parts used and hours spent."""
    issue = clip(request.form.get("issue"), 2200)
    vehicle = db.session.get(Vehicle, request.form.get("vehicle_id", type=int) or 0)
    chat_entry = None
    chat_id = request.form.get("chat_id", type=int)
    if chat_id:
        chat_entry = db.session.get(ChatHistory, chat_id)
        if chat_entry and chat_entry.user_id != g.user.id:
            chat_entry = None
    if chat_entry:
        model, generation = chat_entry.model, chat_entry.generation
        if vehicle and not (rag.norm_key(vehicle.model) == rag.norm_key(model) and rag.norm_key(vehicle.generation) == rag.norm_key(generation)):
            vehicle = None                                   # a registered vehicle must be the same model + generation
    elif vehicle:
        model, generation = vehicle.model, vehicle.generation
    else:
        model = generation = None
    if not model or not issue:
        flash("Choose a registered vehicle and describe the reported issue.", "error")
        return redirect(request.referrer or url_for("orders"))
    values, error = read_tech_fields(request.form, "Open")
    if error:
        flash(error, "error")
        return redirect(request.referrer or url_for("orders"))
    order = WorkOrder(code="TMP-" + secrets.token_hex(4), model=model, generation=generation,
                      vehicle_id=vehicle.id if vehicle else None, technician_id=g.user.id,
                      chat_id=chat_entry.id if chat_entry else None, issue=issue,
                      priority="High" if request.form.get("priority") == "High" else "Normal", **values)
    db.session.add(order)
    db.session.flush()
    order.code = f"WO-{1000 + order.id}"
    add_log(order, "Created and submitted for engineer review" if order.status == READY else f"Created with status {order.status}")
    db.session.commit()
    log.info("Work order %s created by %s", order.code, g.user.username)
    flash(f"{order.code} created.", "success")
    return redirect(url_for("order_detail", code=order.code))


def get_order(code):
    order = WorkOrder.query.filter_by(code=code).first_or_404()
    if g.user.role == TECH and order.technician_id != g.user.id:
        abort(403)
    return order


@app.get("/orders/<code>")
@login_required(TECH, ENGINEER)
def order_detail(code):
    order = get_order(code)
    is_eng = g.user.role == ENGINEER
    related = []
    if is_eng:
        related = (RepairHistory.query.filter(
            func.lower(RepairHistory.model) == (order.model or "").lower(),
            func.lower(RepairHistory.generation) == (order.generation or "").lower(),
            or_(RepairHistory.work_order_id.is_(None), RepairHistory.work_order_id != order.id))
            .order_by(RepairHistory.id.desc()).limit(5).all())
    statuses = tech_statuses(order)
    if order.status not in statuses:
        statuses = [order.status] + statuses
    return render_template("order_detail.html", order=order, statuses=statuses, risks=REVIEW_RISKS, is_eng=is_eng,
                           editable=order.status != "Closed", status_locked=not tech_can_change_status(order),
                           related=related)


@app.post("/orders/<code>/update")
@login_required(TECH)
def order_update(code):
    """Technician progress update: status, findings, actions taken, parts used, hours spent."""
    order = get_order(code)
    if order.status == "Closed":
        abort(403)
    locked = not tech_can_change_status(order)
    values, error = read_tech_fields(request.form, order.status if not locked else "Open")
    if locked:
        if values:
            values["status"] = order.status                  # the engineer owns the status at this stage
    elif not error and values["status"] != order.status and values["status"] not in tech_statuses(order):
        error = ("High-risk cases need an engineer approval before work can start."
                 if values["status"] == "In progress" else "You cannot set that status.")
    if error:
        flash(error, "error")
        return redirect(url_for("order_detail", code=code))
    changed = values["status"] != order.status
    for field, value in values.items():
        setattr(order, field, value)
    if changed and order.status == READY:
        add_log(order, "Submitted for engineer review")
    else:
        add_log(order, f"Technician update saved. Status {order.status}" + (f", {order.hours:g} h" if order.hours else ""))
    db.session.commit()
    flash("Update saved." if order.status != READY or not changed else "Update saved and sent for engineering review.", "success")
    return redirect(url_for("order_detail", code=code))


@app.post("/orders/<code>/review")
@login_required(ENGINEER)
def order_review(code):
    """Engineer review: risk classification + review findings / evidence gap. High risk goes to High-Risk Approval."""
    order = WorkOrder.query.filter_by(code=code).first_or_404()
    risk, gap = request.form.get("risk", ""), clip(request.form.get("review_gap"), 2200)
    if order.status == "Closed":
        abort(403)
    if risk not in ("Low", "Medium", "High"):
        flash("Choose a risk classification: Low, Medium or High.", "error")
        return redirect(url_for("order_detail", code=code))
    order.risk, order.review_gap = risk, gap
    if risk == "High":
        if order.approval == "Approved":
            order.status = "In diagnosis"
            note = "Review saved. Risk High (already approved)"
        else:
            order.approval, order.status = "Pending", "Awaiting approval"
            note = "Review saved. Risk High, sent to High-Risk Approval"
    else:
        order.approval, order.status = "Not requested", "In diagnosis"
        note = f"Review saved. Risk {risk}, moved to Diagnosis and Closure"
    add_log(order, note + (f". Evidence gap: {gap[:300]}" if gap else ""))
    db.session.commit()
    log.info("Review of %s: risk %s by %s", order.code, risk, g.user.username)
    flash("Sent to High-Risk Approval." if risk == "High" and order.approval == "Pending" else "Review saved.", "success")
    return redirect(request.referrer if (request.referrer or "").endswith(("/review", "/approvals")) else url_for("order_detail", code=code))


@app.post("/orders/<code>/approval")
@login_required(ENGINEER)
def order_approval(code):
    """Pop-up decision on a High-risk case: Approved / Rejected / Returned for evidence."""
    order = WorkOrder.query.filter_by(code=code, risk="High").first_or_404()
    decision, scope = request.form.get("decision", ""), clip(request.form.get("scope"), 2200)
    notes = clip(request.form.get("comment"), 1000)
    if order.approval != "Pending":
        flash(f"{order.code} is not waiting for a decision (approval: {order.approval}).", "error")
    elif decision not in ("Approved", "Rejected", "Returned for evidence") or not notes:
        flash("Choose a decision and write your review notes.", "error")
    elif decision == "Approved" and not scope:
        flash("Describe the exact action scope being approved.", "error")
    else:
        order.approval = decision
        if decision == "Approved":
            order.scope, order.status = scope, "In progress"        # the technician can start the approved work
        else:
            order.status = "On hold" if decision == "Rejected" else "Returned for evidence"
        db.session.add(Approval(order=order, engineer_id=g.user.id, decision=decision, scope=scope, comment=notes))
        add_log(order, f"{decision}: {notes}")
        db.session.commit()
        log.info("Approval on %s: %s by %s", order.code, decision, g.user.username)
        flash(f"Decision recorded: {decision}.", "success")
    return redirect(url_for("approvals"))


@app.post("/orders/<code>/closure")
@login_required(ENGINEER)
def order_closure(code):
    """Pop-up on Diagnosis and Closure: 'save' keeps the notes, 'close' closes the case and writes repair history."""
    order = WorkOrder.query.filter_by(code=code).first_or_404()
    action = request.form.get("action", "save")
    diagnosis, verification = clip(request.form.get("diagnosis"), 2200), clip(request.form.get("verification"), 2200)
    outcome = clip(request.form.get("outcome"), 2200)
    if order.status == "Closed":
        flash(f"{order.code} is already closed.", "error")
    elif action == "close" and (not diagnosis or not verification):
        flash("Closing a case needs a final diagnosis and verification notes.", "error")
    elif action == "close" and order.risk == "Unassessed":
        flash("Review the case and classify its risk before closing it.", "error")
    elif action == "close" and order.risk == "High" and order.approval != "Approved":
        flash("A High-risk case needs an approval before it can be closed.", "error")
    else:
        order.diagnosis, order.verification, order.outcome = diagnosis or None, verification or None, outcome or None
        if action == "close":
            order.status, order.closed_at = "Closed", datetime.now()
            db.session.add(RepairHistory(model=order.model, generation=order.generation, vehicle_id=order.vehicle_id,
                                         work_order_id=order.id, recorded_by=g.user.id,
                                         summary=f"{order.issue[:300]} | Diagnosis: {diagnosis[:500]}",
                                         outcome=(outcome or verification)[:1000], is_synthetic=False))
            add_log(order, "Closed with final diagnosis and verification")
            log.info("Work order %s closed by %s", order.code, g.user.username)
            flash(f"{order.code} closed and added to repair history.", "success")
        else:
            add_log(order, "Diagnosis saved")
            flash("Diagnosis saved.", "success")
        db.session.commit()
    return redirect(url_for("closure"))


# ---------------------------------------------------------------------------
# Related repair history (by car model + generation)
# ---------------------------------------------------------------------------
def known_pairs():
    """Every (model, generation) the system knows about, keyed by normalised name. Manuals come first so
    their spelling wins; this keeps 'Fortuner Hilux' and 'fortuner hilux' as ONE entry."""
    pairs = {}
    sources = (db.session.query(Document.model, Document.generation).filter(Document.applicability == "vehicle_specific"),
               db.session.query(Vehicle.model, Vehicle.generation), db.session.query(RepairHistory.model, RepairHistory.generation))
    for query in sources:
        for model, generation in query.distinct().all():
            if model and generation:
                pairs.setdefault((rag.norm_key(model), rag.norm_key(generation)), (model, generation))
    return sorted(pairs.values(), key=lambda p: (p[0].lower(), p[1].lower()))


@app.get("/history")
@login_required(ENGINEER, ADMIN)
def history():
    pair = request.args.get("pair", "")
    model, generation = pair.split("||", 1) if "||" in pair else ("", "")
    query = RepairHistory.query
    if model:                                       # "related" = same model and generation
        query = query.filter(func.lower(RepairHistory.model) == model.lower(),
                             func.lower(RepairHistory.generation) == generation.lower())
    rows = query.order_by(RepairHistory.id.desc()).limit(100).all()
    return render_template("history.html", rows=rows, selected=(model, generation) if model else None, pairs=known_pairs())


@app.post("/history/add")
@login_required(ENGINEER, ADMIN)
def history_add():
    model, generation = clip(request.form.get("model"), 80), clip(request.form.get("generation"), 40)
    summary = clip(request.form.get("summary"), 2000)
    if not model or not generation or not summary:
        flash("Enter the car model, generation and a repair summary.", "error")
        return redirect(url_for("history"))
    model, generation = {(rag.norm_key(m), rag.norm_key(gn)): (m, gn) for m, gn in known_pairs()}.get(
        (rag.norm_key(model), rag.norm_key(generation)), (model, generation))
    db.session.add(RepairHistory(model=model, generation=generation, summary=summary,
                                 outcome=clip(request.form.get("outcome"), 1000),
                                 is_synthetic=request.form.get("synthetic") == "on", recorded_by=g.user.id))
    db.session.commit()
    flash("History record added.", "success")
    return redirect(url_for("history", pair=f"{model}||{generation}"))


# ---------------------------------------------------------------------------
# Administrator: documents (upload + automatic indexing)
# ---------------------------------------------------------------------------
def process_document(doc_id):
    """Index one PDF and store the result (status, pages, chunk count) in MySQL."""
    with app.app_context():
        doc = db.session.get(Document, doc_id)
        if not doc:
            return
        try:
            doc.status, doc.error_message = "Processing", None
            db.session.commit()
            meta = {"document_type": doc.document_type, "model": doc.model, "generation": doc.generation,
                    "year_range": doc.year_range, "synthetic": doc.synthetic}
            result = rag.index_document(doc.id, BASE_DIR / doc.file_path, meta, use_ocr=app.config["OCR_ENABLED"])
            doc.status, doc.page_count, doc.chunk_count = result["status"], result["pages"], result["chunks"]
            doc.error_message = result["message"] or None
            doc.processed_at = datetime.now()
            if result["status"] == "Indexed":
                doc.synthetic = doc.synthetic or any(c.get("synthetic") for c in rag.load_chunks() if c.get("doc_id") == doc.id)
            db.session.commit()
            log.info("Document %s -> %s (%s pages, %s chunks) %s", doc.filename, doc.status, doc.page_count,
                     doc.chunk_count, doc.error_message or "")
        except Exception as exc:
            db.session.rollback()
            log.exception("Processing failed for document %s", doc_id)
            doc = db.session.get(Document, doc_id)
            if doc:
                doc.status, doc.error_message = "Failed", f"Unexpected error ({type(exc).__name__}). See app.log."
                db.session.commit()


def schedule_processing(doc_id):
    if app.config["PROCESS_IN_BACKGROUND"]:
        threading.Thread(target=process_document, args=(doc_id,), daemon=True).start()
    else:
        process_document(doc_id)


@app.get("/documents")
@login_required(ADMIN)
def documents():
    q = request.args.get("q", "").strip()
    query = Document.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Document.filename.like(like), Document.model.like(like), Document.generation.like(like)))
    pairs = known_pairs()
    return render_template("documents.html", docs=query.order_by(Document.id.desc()).all(), q=q,
                           models=sorted({m for m, _ in pairs}), generations=sorted({gn for _, gn in pairs}),
                           document_types=rag.DOCUMENT_TYPES, rag=rag.rag_status(), ocr_ready=rag.tesseract_available())


@app.get("/documents/status")
@login_required(ADMIN)
def documents_status():
    return jsonify([{"id": d.id, "status": d.status, "chunks": d.chunk_count, "pages": d.page_count}
                    for d in Document.query.all()])


@app.post("/documents/upload")
@login_required(ADMIN)
def upload_documents():
    """Each PDF is described by: car model, generation, document type and years. That metadata is what the
    chat dropdown and the search filter use. Processing starts automatically."""
    files = [f for f in request.files.getlist("pdfs") if f and f.filename]
    if not files:
        flash("Choose at least one PDF file.", "error")
        return redirect(url_for("documents"))
    types, models_in = request.form.getlist("document_type"), request.form.getlist("model")
    gens_in, years = request.form.getlist("generation"), request.form.getlist("year_range")
    spelling = {(rag.norm_key(m), rag.norm_key(gn)): (m, gn) for m, gn in known_pairs()}
    started = 0
    for n, upload in enumerate(files):
        name = secure_filename(upload.filename) or f"document_{n + 1}.pdf"
        pick = lambda values: values[n] if n < len(values) else ""
        meta, error = rag.clean_metadata({"document_type": pick(types), "model": clip(pick(models_in), 80),
                                          "generation": clip(pick(gens_in), 40), "year_range": clip(pick(years), 40)})
        data = upload.read()
        if error:
            flash(f"{name}: {error}", "error")
        elif not name.lower().endswith(".pdf") or not data.startswith(b"%PDF"):
            flash(f"{name}: this is not a valid PDF file.", "error")
        elif len(data) < 200:
            flash(f"{name}: the PDF is empty.", "error")
        else:
            digest = hashlib.sha256(data).hexdigest()
            twin = Document.query.filter_by(sha256=digest).first()
            if twin:
                flash(f"{name}: this exact file is already uploaded as '{twin.filename}' ({twin.status}).", "error")
                continue
            if meta["applicability"] == "vehicle_specific":       # reuse the existing spelling of a known model/generation
                meta["model"], meta["generation"] = spelling.get((rag.norm_key(meta["model"]), rag.norm_key(meta["generation"])),
                                                                 (meta["model"], meta["generation"]))
            stem, target, counter = Path(name).stem, UPLOAD_DIR / name, 1
            while target.exists():
                counter += 1
                target = UPLOAD_DIR / f"{stem}_{counter}.pdf"
            target.write_bytes(data)
            doc = Document(filename=target.name, file_path=f"uploads/{target.name}", sha256=digest,
                           document_type=meta["document_type"], model=meta["model"], generation=meta["generation"],
                           year_range=meta["year_range"], applicability=meta["applicability"],
                           synthetic=False, status="Pending", uploaded_by=g.user.id)
            db.session.add(doc)
            db.session.commit()
            log.info("Uploaded %s (%s / %s) by %s", doc.filename, doc.model, doc.generation, g.user.username)
            schedule_processing(doc.id)
            started += 1
    if started:
        flash(f"{started} PDF(s) uploaded. Processing started automatically; the status updates on this page.", "success")
    return redirect(url_for("documents"))


@app.post("/documents/<int:doc_id>/reprocess")
@login_required(ADMIN)
def document_reprocess(doc_id):
    doc = db.get_or_404(Document, doc_id)
    if doc.status in ("Pending", "Processing"):
        flash("This document is already being processed.", "info")
    elif not (BASE_DIR / doc.file_path).exists():
        flash("The original PDF file is missing from the uploads folder.", "error")
    else:
        doc.status, doc.error_message = "Pending", None
        db.session.commit()
        schedule_processing(doc.id)
        flash("Reprocessing started.", "success")
    return redirect(url_for("documents"))


@app.post("/documents/<int:doc_id>/delete")
@login_required(ADMIN)
def document_delete(doc_id):
    doc = db.get_or_404(Document, doc_id)
    if doc.status == "Processing":
        flash("Wait until processing finishes before removing this document.", "error")
        return redirect(url_for("documents"))
    rag.remove_document(doc.id)
    (BASE_DIR / doc.file_path).unlink(missing_ok=True)
    name = doc.filename
    db.session.delete(doc)
    db.session.commit()
    log.info("Document removed: %s by %s", name, g.user.username)
    flash(f"{name} was removed from the index and the uploads folder.", "success")
    return redirect(url_for("documents"))


@app.get("/documents/<int:doc_id>/file")
@login_required(ADMIN)
def document_file(doc_id):
    doc = db.get_or_404(Document, doc_id)
    return send_from_directory(UPLOAD_DIR, Path(doc.file_path).name, mimetype="application/pdf")


# ---------------------------------------------------------------------------
# Fault codes: administrators upload CSV / PDF lists, technicians look a code up
# ---------------------------------------------------------------------------
def code_applies(entry, pair):
    """A code row with no model applies to every vehicle; otherwise it must match the chosen model (+ generation)."""
    if not pair or not entry.model:
        return True
    if rag.norm_key(entry.model) != rag.norm_key(pair[0]):
        return False
    return not entry.generation or rag.norm_key(entry.generation) == rag.norm_key(pair[1])


@app.get("/fault-codes")
@login_required(TECH)
def fault_codes():
    q = request.args.get("code", "").strip()[:80]
    chosen = request.args.get("vehicle", "")
    pair = tuple(chosen.split("||", 1)) if "||" in chosen else None
    pairs = known_pairs()
    if pair and not any(rag.norm_key(pair[0]) == rag.norm_key(m) and rag.norm_key(pair[1]) == rag.norm_key(gn) for m, gn in pairs):
        pair = None
    results, similar, mode = [], [], None
    if q:
        code = faultcodes.normalize_code(q)
        if code:
            mode = "code"
            found = FaultCode.query.filter_by(code=code).order_by(FaultCode.id).all()
            results = [e for e in found if code_applies(e, pair)]
            results.sort(key=lambda e: 0 if e.model else 1)           # vehicle-specific entries first
            if not results:
                similar = [c for (c,) in db.session.query(FaultCode.code).filter(FaultCode.code.like(code[:3] + "%"))
                           .distinct().order_by(FaultCode.code).limit(12).all()]
        else:
            mode = "keyword"
            like = "%" + q.replace("%", "").replace("_", "") + "%"
            found = (FaultCode.query.filter(or_(FaultCode.description.like(like), FaultCode.system.like(like), FaultCode.causes.like(like)))
                     .order_by(FaultCode.code).limit(60).all())
            results = [e for e in found if code_applies(e, pair)][:30]
    total = FaultCode.query.count()
    return render_template("fault_codes.html", q=q, mode=mode, results=results, similar=similar, total=total,
                           pairs=pairs, chosen=f"{pair[0]}||{pair[1]}" if pair else "")


FAULT_DIR = BASE_DIR / "fault_code_files"


@app.get("/fault-codes/manage")
@login_required(ADMIN)
def fault_code_admin():
    pairs = known_pairs()
    files = FaultCodeFile.query.order_by(FaultCodeFile.id.desc()).all()
    return render_template("fault_code_admin.html", files=files, total=sum(f.code_count or 0 for f in files),
                           models=sorted({m for m, _ in pairs}), generations=sorted({gn for _, gn in pairs}))


@app.get("/fault-codes/template.csv")
@login_required(ADMIN)
def fault_code_template():
    return app.response_class(faultcodes.SAMPLE_CSV, mimetype="text/csv",
                              headers={"Content-Disposition": "attachment; filename=fault_codes_template.csv"})


@app.post("/fault-codes/upload")
@login_required(ADMIN)
def fault_code_upload():
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Choose at least one CSV or PDF file.", "error")
        return redirect(url_for("fault_code_admin"))
    model, generation = clip(request.form.get("model"), 80), clip(request.form.get("generation"), 40)
    if generation and not model:
        flash("Enter the car model too, or leave both empty for a list that applies to every vehicle.", "error")
        return redirect(url_for("fault_code_admin"))
    FAULT_DIR.mkdir(exist_ok=True)
    for upload in files:
        name = secure_filename(upload.filename) or "fault_codes"
        kind = name.rsplit(".", 1)[-1].lower() if "." in name else ""
        data = upload.read()
        if kind not in ("csv", "pdf"):
            flash(f"{name}: only CSV and PDF files are accepted.", "error")
            continue
        if kind == "pdf" and not data.startswith(b"%PDF"):
            flash(f"{name}: this is not a valid PDF.", "error")
            continue
        digest = hashlib.sha256(data).hexdigest()
        existing = FaultCodeFile.query.filter_by(sha256=digest).first()
        if existing:
            flash(f"{name}: already uploaded as {existing.filename}.", "error")
            continue
        rows, message = faultcodes.parse_csv(data) if kind == "csv" else faultcodes.parse_pdf(data)
        if not rows:
            flash(f"{name}: {message}", "error")
            continue
        stored = f"{digest[:12]}_{name}"
        (FAULT_DIR / stored).write_bytes(data)
        record = FaultCodeFile(filename=name, file_path=f"fault_code_files/{stored}", file_type=kind, sha256=digest,
                               model=model or None, generation=generation or None, code_count=len(rows), uploaded_by=g.user.id)
        db.session.add(record)
        try:
            db.session.flush()
            db.session.execute(db.insert(FaultCode), [
                {"file_id": record.id, "code": r["code"], "description": r["description"], "system": r["system"] or None,
                 "causes": r["causes"] or None, "action": r["action"] or None, "page": r["page"],
                 "model": (r["model"] or model) or None, "generation": (r["generation"] or (generation if not r["model"] else "")) or None}
                for r in rows])
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            (FAULT_DIR / stored).unlink(missing_ok=True)
            log.exception("Could not save fault codes from %s", name)
            flash(f"{name}: could not be saved. Please try again.", "error")
            continue
        log.info("Fault codes: %s uploaded by %s (%d codes)", name, g.user.username, len(rows))
        flash(f"{name}: {len(rows)} fault codes added." + (f" {message}" if message else ""), "success")
    return redirect(url_for("fault_code_admin"))


@app.post("/fault-codes/<int:file_id>/delete")
@login_required(ADMIN)
def fault_code_delete(file_id):
    record = db.get_or_404(FaultCodeFile, file_id)
    name, path = record.filename, BASE_DIR / record.file_path
    db.session.delete(record)
    db.session.commit()
    path.unlink(missing_ok=True)
    log.info("Fault code list %s deleted by %s", name, g.user.username)
    flash(f"{name} and its fault codes were removed.", "success")
    return redirect(url_for("fault_code_admin"))


@app.get("/fault-codes/<int:file_id>/file")
@login_required(ADMIN)
def fault_code_file(file_id):
    record = db.get_or_404(FaultCodeFile, file_id)
    return send_from_directory(FAULT_DIR, Path(record.file_path).name, as_attachment=record.file_type == "csv")


# ---------------------------------------------------------------------------
# Administrator: users, safety rules, settings
# ---------------------------------------------------------------------------
@app.get("/users")
@login_required(ADMIN)
def users():
    q = request.args.get("q", "").strip()
    query = User.query
    if q:
        like = f"%{q}%"
        query = query.filter(or_(User.username.like(like), User.email.like(like), User.full_name.like(like)))
    rows = query.order_by(User.status.desc(), User.id).all()           # "pending" users first
    return render_template("users.html", rows=rows, q=q, roles=ROLES, statuses=USER_STATUSES)


def active_admins_left(excluding=None):
    query = User.query.filter_by(role=ADMIN, status="active")
    return query.filter(User.id != excluding.id).count() if excluding else query.count()


@app.post("/users/save")
@login_required(ADMIN)
def user_save():
    uid = request.form.get("id", type=int)
    user = db.session.get(User, uid) if uid else None
    if uid and not user:
        abort(404)
    f = request.form
    full_name, email, username = clip(f.get("full_name"), 120), clip(f.get("email"), 120).lower(), clip(f.get("username"), 40).lower()
    role, status, password = f.get("role"), f.get("status", "active"), f.get("password", "")
    error = None
    if len(full_name) < 2 or not EMAIL_RE.match(email) or not USERNAME_RE.match(username):
        error = "Enter a full name, a valid email and a username (3-30 letters, numbers, . - _)."
    elif role not in ROLES or status not in USER_STATUSES:
        error = "Invalid role or status."
    elif (not user or password) and password_problem(password):
        error = password_problem(password)
    elif User.query.filter(or_(User.email == email, User.username == username), User.id != (uid or 0)).first():
        error = "That email or username is already used by another account."
    elif user and user.role == ADMIN and (role != ADMIN or status != "active") and active_admins_left(user) == 0:
        error = "There must be at least one active administrator."
    elif user and user.id == g.user.id and (role != ADMIN or status != "active"):
        error = "You cannot remove your own administrator access."
    if error:
        flash(error, "error")
        return redirect(url_for("users"))
    user = user or User()
    user.full_name, user.email, user.username, user.role, user.status = full_name, email, username, role, status
    if password:
        user.password_hash = generate_password_hash(password)
    db.session.add(user)
    db.session.commit()
    log.info("User saved: %s (%s, %s) by %s", username, role, status, g.user.username)
    flash("User saved.", "success")
    return redirect(url_for("users"))


@app.post("/users/<int:uid>/approve")
@login_required(ADMIN)
def user_approve(uid):
    user = db.get_or_404(User, uid)
    role = request.form.get("role")
    if role not in ROLES:
        flash("Choose a role.", "error")
    else:
        user.role, user.status = role, "active"
        db.session.commit()
        log.info("User approved: %s as %s by %s", user.username, role, g.user.username)
        flash(f"{user.username} approved as {role}.", "success")
    return redirect(url_for("users"))


@app.post("/users/<int:uid>/delete")
@login_required(ADMIN)
def user_delete(uid):
    user = db.get_or_404(User, uid)
    if user.id == g.user.id:
        flash("You cannot delete your own account.", "error")
    elif user.role == ADMIN and user.status == "active" and active_admins_left(user) == 0:
        flash("There must be at least one active administrator.", "error")
    else:
        try:
            db.session.delete(user)
            db.session.commit()
            log.info("User deleted: %s by %s", user.username, g.user.username)
            flash("User deleted.", "success")
        except IntegrityError:
            db.session.rollback()
            flash("This user has work orders, chats or documents on record. Disable the account instead.", "error")
    return redirect(url_for("users"))


@app.route("/safety", methods=["GET", "POST"])
@login_required(ADMIN)
def safety():
    if request.method == "POST":
        risk, note = request.form.get("risk", ""), clip(request.form.get("note"), 600)
        if risk in ("High", "Medium", "Low") and note:
            set_setting("safety_" + risk, note)
            db.session.commit()
            flash(f"{risk} risk rule saved.", "success")
        else:
            flash("The rule text cannot be empty.", "error")
        return redirect(url_for("safety"))
    rules = [(risk, get_setting("safety_" + risk)) for risk in ("High", "Medium", "Low")]
    return render_template("settings.html", section="safety", rules=rules)


@app.route("/settings", methods=["GET", "POST"])
@login_required(ADMIN)
def settings():
    if request.method == "POST":
        if request.form.get("top_k") in ("3", "5", "8", "10"):
            set_setting("top_k", request.form["top_k"])
            db.session.commit()
            flash("Settings saved.", "success")
        return redirect(url_for("settings"))
    try:
        db.session.execute(db.text("SELECT 1"))
        db_ok = True
    except SQLAlchemyError:
        db_ok = False
    return render_template("settings.html", section="settings", top_k=get_setting("top_k", "5"), ai_ready=ai_key_ready(),
                           llm_model=os.getenv("LLM_MODEL", "meta-llama/llama-3.3-70b-instruct"), db_ok=db_ok,
                           db_name=db.engine.url.database, rag=rag.rag_status(), ocr_ready=rag.tesseract_available())


# ---------------------------------------------------------------------------
# Error pages (no technical details or secrets are shown to the user)
# ---------------------------------------------------------------------------
def error_page(code, title, message):
    if request.path.startswith("/api/"):
        return jsonify({"error": message}), code
    return render_template("error.html", code=code, title=title, message=message), code


@app.errorhandler(400)
def bad_request(e):
    return error_page(400, "Request not accepted", getattr(e, "description", "") or "Please go back and try again.")


@app.errorhandler(403)
def forbidden(_e):
    return error_page(403, "Access denied", "Your role does not have permission to open this page.")


@app.errorhandler(404)
def not_found(_e):
    return error_page(404, "Page not found", "That page or record does not exist.")


@app.errorhandler(413)
def too_large(_e):
    return error_page(413, "File too large", "The upload is larger than 100 MB. Upload fewer or smaller files.")


@app.errorhandler(OperationalError)
def database_down(e):
    db.session.rollback()
    log.error("Database connection problem: %s", type(e).__name__)
    return error_page(503, "Database unavailable", "The application cannot reach MySQL right now. Please try again shortly.")


@app.errorhandler(Exception)
def server_error(e):
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return error_page(e.code, e.name, e.description)
    db.session.rollback()
    log.exception("Unexpected error on %s", request.path)
    return error_page(500, "Something went wrong", "An unexpected error occurred. It has been logged for the administrator.")


# ---------------------------------------------------------------------------
# Start-up: create tables, link existing PDFs/index to MySQL records, first-admin command
# ---------------------------------------------------------------------------
def sync_rag_records():
    """Keep MySQL and the on-disk RAG store consistent after a restart or a first run."""
    stuck = Document.query.filter(Document.status.in_(["Pending", "Processing"])).all()
    for doc in stuck:
        doc.status, doc.error_message = "Failed", "Processing was interrupted by a restart. Press Reprocess."

    # Repair the link between the search index and MySQL. Chunks are "orphans" when no document record
    # matches them: an index from the earlier version (no ids), a database that was recreated, or a half-finished upload.
    by_name = {d.filename: d for d in Document.query.all()}
    valid = {d.id: d.filename for d in by_name.values()}
    chunks = rag.load_chunks()
    indexed_ids = {c.get("doc_id") for c in chunks if valid.get(c.get("doc_id")) == c.get("source")}
    orphans = {}
    for chunk in chunks:
        if valid.get(chunk.get("doc_id")) != chunk.get("source"):
            orphans.setdefault(chunk.get("source"), []).append(chunk)
    mapping, drop = {}, set()
    for source, group in orphans.items():
        path = UPLOAD_DIR / (source or "")
        record = by_name.get(source)
        if record and record.id in indexed_ids:
            drop.add(source)                                   # this PDF is already indexed: never keep a second copy
        elif record:
            mapping[source] = record.id                        # record exists, the chunks just carry a wrong id
        elif source and path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if Document.query.filter_by(sha256=digest).first():
                drop.add(source)                               # identical file already registered under another name
                continue
            first = group[0]
            doc = Document(filename=source, file_path=f"uploads/{source}", sha256=digest,
                           document_type=first.get("document_type") or "other", model=first.get("model"),
                           generation=first.get("generation"), year_range=first.get("year_range") or "Not specified",
                           applicability=first.get("applicability") or "vehicle_specific",
                           synthetic=bool(first.get("synthetic")), page_count=len({c.get("page") for c in group}),
                           chunk_count=len(group), status="Indexed", processed_at=datetime.now())
            db.session.add(doc)
            db.session.flush()
            mapping[source] = doc.id
            log.info("Linked existing PDF to the database: %s", source)
        else:
            drop.add(source)                                   # no PDF and no record: nothing to show as evidence
            log.warning("Removed search data of a missing PDF: %s", source)
    if mapping:
        rag.tag_chunks(mapping, valid)
    if drop:
        valid_after = {**valid, **{doc_id: name for name, doc_id in mapping.items()}}
        rag.keep_chunks(lambda c: c.get("source") not in drop or valid_after.get(c.get("doc_id")) == c.get("source"))

    have = {c.get("doc_id") for c in rag.load_chunks()}
    for doc in Document.query.filter_by(status="Indexed"):
        if doc.id not in have:
            doc.status, doc.error_message = "Failed", "Search data is missing from rag_store. Press Reprocess."
    db.session.commit()


def upgrade_schema():
    """create_all() never changes an existing table, so databases made by an earlier version are upgraded here."""
    def columns(table):
        return {c["name"] for c in inspect(db.engine).get_columns(table)}

    if "requested_role" not in columns("users"):
        db.session.execute(db.text("ALTER TABLE users ADD COLUMN requested_role VARCHAR(30)"))
        db.session.commit()
        log.info("Database upgraded: added users.requested_role")
    if "actions_taken" not in columns("work_orders"):          # technician "Actions taken" + engineer "evidence gap"
        db.session.execute(db.text("ALTER TABLE work_orders ADD COLUMN actions_taken TEXT"))
        db.session.execute(db.text("ALTER TABLE work_orders ADD COLUMN review_gap TEXT"))
        db.session.execute(db.text("UPDATE work_orders SET actions_taken = outcome WHERE outcome IS NOT NULL"))
        db.session.commit()
        log.info("Database upgraded: added work_orders.actions_taken and work_orders.review_gap")
    # Work orders, history and chats now keep the car model + generation themselves (the registry vehicle is optional).
    for table in ("work_orders", "repair_history", "chat_history"):
        have = columns(table)
        if "model" in have:
            continue
        db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN model VARCHAR(80)"))
        db.session.execute(db.text(f"ALTER TABLE {table} ADD COLUMN generation VARCHAR(40)"))
        if "vehicle_id" in have:
            for field in ("model", "generation"):
                db.session.execute(db.text(f"UPDATE {table} SET {field} = (SELECT v.{field} FROM vehicles v WHERE v.id = {table}.vehicle_id)"))
            if db.engine.dialect.name == "mysql":
                db.session.execute(db.text(f"ALTER TABLE {table} MODIFY vehicle_id INT NULL"))
        db.session.commit()
        log.info("Database upgraded: %s now stores model and generation", table)


def init_database():
    try:
        with app.app_context():
            db.create_all()
            upgrade_schema()
            for key, value in DEFAULT_SETTINGS.items():
                if db.session.get(Setting, key) is None:
                    db.session.add(Setting(key=key, value=value))
            db.session.commit()
            sync_rag_records()
    except SQLAlchemyError as exc:
        log.error("Cannot connect to the database: %s", type(exc).__name__)
        raise SystemExit("\nCould not connect to MySQL. Check that MySQL is running, the database exists, and "
                         "DB_HOST / DB_USER / DB_PASSWORD / DB_NAME in your .env file are correct.\n")


@app.cli.command("create-admin")
@click.option("--full-name", prompt="Full name")
@click.option("--email", prompt="Email")
@click.option("--username", prompt="Username")
@click.password_option()
def create_admin(full_name, email, username, password):
    """Create an administrator account (use this once to create the first admin)."""
    email, username = email.strip().lower(), username.strip().lower()
    if not EMAIL_RE.match(email) or not USERNAME_RE.match(username) or password_problem(password):
        raise click.ClickException(password_problem(password) or "Invalid email or username.")
    if User.query.filter(or_(User.email == email, User.username == username)).first():
        raise click.ClickException("That email or username already exists.")
    db.session.add(User(full_name=full_name.strip(), email=email, username=username,
                        password_hash=generate_password_hash(password), role=ADMIN, status="active"))
    db.session.commit()
    click.echo(f"Administrator '{username}' created. You can log in now.")


init_database()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
