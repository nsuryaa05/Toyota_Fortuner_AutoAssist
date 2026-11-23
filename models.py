"""Database tables (SQLAlchemy). MySQL keeps the app records; the PDFs and the FAISS index stay on disk.

Tables: users, vehicles, documents, work_orders, order_log, approvals, repair_history,
chat_history, system_settings, fault_code_files, fault_codes, otp_challenges.
"""
from datetime import datetime

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

db = SQLAlchemy()


@event.listens_for(Engine, "connect")
def sqlite_enforce_foreign_keys(dbapi_connection, _record):
    """MySQL enforces foreign keys by itself; SQLite (used only by the tests) needs this switch."""
    if dbapi_connection.__class__.__module__.startswith("sqlite3"):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

TECH, ENGINEER, ADMIN = "Technician", "Maintenance Engineer", "Administrator"
ROLES = [TECH, ENGINEER, ADMIN]
USER_STATUSES = ["pending", "active", "disabled"]


class User(db.Model):
    __tablename__ = "users"
    id = db.Column(db.Integer, primary_key=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(40), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)      # Werkzeug hash, never the password
    role = db.Column(db.String(30), nullable=False, default=TECH)
    requested_role = db.Column(db.String(30))                      # role asked for at sign-up (admin decides)
    status = db.Column(db.String(20), nullable=False, default="pending")   # pending / active / disabled
    failed_attempts = db.Column(db.Integer, nullable=False, default=0)
    locked_until = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.now)
    last_login = db.Column(db.DateTime)


class OtpChallenge(db.Model):
    """Short-lived, one-use email OTP challenge. Only a hash of the code is stored."""
    __tablename__ = "otp_challenges"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    purpose = db.Column(db.String(20), nullable=False, index=True)  # login / password_reset
    code_hash = db.Column(db.String(64), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.now)
    expires_at = db.Column(db.DateTime, nullable=False, index=True)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    consumed_at = db.Column(db.DateTime)
    user = db.relationship("User")


class Vehicle(db.Model):
    __tablename__ = "vehicles"
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(db.String(30), unique=True, nullable=False)
    model = db.Column(db.String(80), nullable=False, index=True)
    generation = db.Column(db.String(40), nullable=False)
    year = db.Column(db.String(40))
    engine = db.Column(db.String(80))
    variant = db.Column(db.String(80))
    location = db.Column(db.String(80))
    criticality = db.Column(db.String(20), default="Unassessed")
    status = db.Column(db.String(20), default="Available")


class Document(db.Model):
    """One uploaded PDF. The file lives in uploads/, its search data in rag_store/ (not in MySQL)."""
    __tablename__ = "documents"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)           # relative to the project folder
    sha256 = db.Column(db.String(64), unique=True, nullable=False)  # duplicate detection
    document_type = db.Column(db.String(30), nullable=False)
    model = db.Column(db.String(80))                                # empty for a general DTC reference
    generation = db.Column(db.String(40))
    year_range = db.Column(db.String(40), default="Not specified")
    applicability = db.Column(db.String(30), nullable=False)        # vehicle_specific / general_dtc_reference
    synthetic = db.Column(db.Boolean, default=False)
    page_count = db.Column(db.Integer, default=0)
    chunk_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), nullable=False, default="Pending")   # Pending/Processing/Indexed/Needs OCR/Failed
    error_message = db.Column(db.Text)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    uploaded_at = db.Column(db.DateTime, default=datetime.now)
    processed_at = db.Column(db.DateTime)
    uploader = db.relationship("User")


class ChatHistory(db.Model):
    __tablename__ = "chat_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    model = db.Column(db.String(80), nullable=False, index=True)        # the car model + generation chosen in the chat
    generation = db.Column(db.String(40), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    sources = db.Column(db.JSON)                                    # [{"document":..., "page":...}]
    mode = db.Column(db.String(30))                                 # ok / no_match / api_error ...
    created_at = db.Column(db.DateTime, default=datetime.now)
    user = db.relationship("User")


class WorkOrder(db.Model):
    __tablename__ = "work_orders"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(20), unique=True, nullable=False)
    model = db.Column(db.String(80), index=True)                        # car model + generation the case is about
    generation = db.Column(db.String(40))
    vehicle_id = db.Column(db.Integer, db.ForeignKey("vehicles.id"), index=True)   # optional: a registered vehicle
    technician_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    chat_id = db.Column(db.Integer, db.ForeignKey("chat_history.id"))   # the AI answer this order came from
    issue = db.Column(db.Text, nullable=False)
    priority = db.Column(db.String(20), default="Normal")
    status = db.Column(db.String(30), default="Under review")
    risk = db.Column(db.String(20), default="Unassessed")
    approval = db.Column(db.String(30), default="Not requested")
    scope = db.Column(db.Text)
    findings = db.Column(db.Text)                                   # technician: what was found
    actions_taken = db.Column(db.Text)                              # technician: what was done
    review_gap = db.Column(db.Text)                                 # engineer: review findings / evidence gap
    outcome = db.Column(db.Text)                                    # engineer: outcome / additional notes (closure)
    parts = db.Column(db.String(300), default="None documented")
    hours = db.Column(db.Float, default=0)
    diagnosis = db.Column(db.Text)
    verification = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.now)
    closed_at = db.Column(db.DateTime)
    vehicle = db.relationship("Vehicle")
    technician = db.relationship("User")
    chat = db.relationship("ChatHistory")
    log = db.relationship("OrderLog", backref="order", order_by="OrderLog.id")
    approvals = db.relationship("Approval", backref="order", order_by="Approval.id")


class OrderLog(db.Model):
    __tablename__ = "order_log"
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False, index=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"))
    text = db.Column(db.Text, nullable=False)
    at = db.Column(db.DateTime, default=datetime.now)
    user = db.relationship("User")


class Approval(db.Model):
    __tablename__ = "approvals"
    id = db.Column(db.Integer, primary_key=True)
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"), nullable=False, index=True)
    engineer_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    decision = db.Column(db.String(30), nullable=False)             # Approved / Rejected / Returned for evidence
    scope = db.Column(db.Text)
    evidence = db.Column(db.Text)
    comment = db.Column(db.Text)
    decided_at = db.Column(db.DateTime, default=datetime.now)
    engineer = db.relationship("User")


class RepairHistory(db.Model):
    __tablename__ = "repair_history"
    id = db.Column(db.Integer, primary_key=True)
    model = db.Column(db.String(80), index=True)
    generation = db.Column(db.String(40))
    vehicle_id = db.Column(db.Integer, db.ForeignKey("vehicles.id"), index=True)   # optional
    work_order_id = db.Column(db.Integer, db.ForeignKey("work_orders.id"))
    summary = db.Column(db.Text, nullable=False)
    outcome = db.Column(db.Text)
    is_synthetic = db.Column(db.Boolean, default=False)             # sample / unverified record
    recorded_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    recorded_at = db.Column(db.DateTime, default=datetime.now)
    vehicle = db.relationship("Vehicle")
    work_order = db.relationship("WorkOrder")


class FaultCodeFile(db.Model):
    """One uploaded fault-code list (CSV or PDF). Its rows live in fault_codes."""
    __tablename__ = "fault_code_files"
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(500), nullable=False)           # relative to the project folder
    file_type = db.Column(db.String(10), nullable=False)            # csv / pdf
    sha256 = db.Column(db.String(64), unique=True, nullable=False)  # duplicate detection
    model = db.Column(db.String(80))                                # empty = applies to every vehicle
    generation = db.Column(db.String(40))
    code_count = db.Column(db.Integer, default=0)
    uploaded_by = db.Column(db.Integer, db.ForeignKey("users.id"))
    uploaded_at = db.Column(db.DateTime, default=datetime.now)
    uploader = db.relationship("User")
    codes = db.relationship("FaultCode", backref="file", cascade="all, delete-orphan")


class FaultCode(db.Model):
    __tablename__ = "fault_codes"
    id = db.Column(db.Integer, primary_key=True)
    file_id = db.Column(db.Integer, db.ForeignKey("fault_code_files.id"), nullable=False, index=True)
    code = db.Column(db.String(20), nullable=False, index=True)     # normalised: upper case, no spaces or dashes
    description = db.Column(db.Text, nullable=False)                # what the code means
    system = db.Column(db.String(120))
    causes = db.Column(db.Text)
    action = db.Column(db.Text)                                     # recommended check / fix
    model = db.Column(db.String(80))                                # copied from the row or the file; empty = general
    generation = db.Column(db.String(40))
    page = db.Column(db.Integer)                                    # PDF page the entry was read from


class Setting(db.Model):
    __tablename__ = "system_settings"
    key = db.Column(db.String(60), primary_key=True)
    value = db.Column(db.Text, nullable=False)


DEFAULT_SETTINGS = {
    "top_k": "5",
    "safety_High": "Require applicable evidence, exact action scope and qualified engineer review before any work.",
    "safety_Medium": "Use source-backed precautions only; escalate when the manual does not cover the question.",
    "safety_Low": "Evidence is still required; never invent technical steps.",
}


def get_setting(key, default=""):
    row = db.session.get(Setting, key)
    return row.value if row else DEFAULT_SETTINGS.get(key, default)


def set_setting(key, value):
    row = db.session.get(Setting, key)
    if row:
        row.value = value
    else:
        db.session.add(Setting(key=key, value=value))
