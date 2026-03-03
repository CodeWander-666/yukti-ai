"""
Main Streamlit application for Yukti AI.
Handles:
- User authentication (login/signup)
- Chat interface for normal users
- Advanced admin dashboard with real‑time metrics, CRUD, analytics
- Error handling and logging
- Integration with all other modules
"""

import os
import sys
import time
import json
import logging
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# Third‑party imports
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Local imports (all must be present)
try:
    from config import (
        BASE_DIR,
        VECTORDB_PATH,
        DB_PATH,
        RETRIEVAL_CACHE_TTL,
        TASK_POLL_INTERVAL,
        ZHIPU_API_KEY,
        GOOGLE_API_KEY,
    )
except ImportError as e:
    st.error(f"Configuration error: {e}")
    st.stop()

try:
    from langchain_helper import (
        create_vector_db,
        get_kb_detailed_status,
        get_document_count,
        check_kb_status,
    )
except ImportError as e:
    st.error(f"Knowledge base module error: {e}")
    st.stop()

try:
    from think import think
except ImportError as e:
    st.error(f"Reasoning engine error: {e}")
    st.stop()

try:
    from model_manager import (
        get_available_models,
        MODELS,
        get_active_tasks,
        get_task_status,
        ZHIPU_AVAILABLE,
        GEMINI_AVAILABLE,
        load_model,
    )
except ImportError as e:
    st.error(f"Model manager error: {e}")
    st.stop()

try:
    from language_detector import detect_language
except ImportError as e:
    st.warning(f"Language detector unavailable: {e}")
    def detect_language(text):
        return {"language": "en", "method": "fallback", "explicit_instruction": None}

try:
    from ui_helpers import (
        render_task,
        show_error_toast,
        show_success_toast,
        confirm_dialog,
        metric_card,
        log_viewer,
    )
except ImportError as e:
    st.error(f"UI helpers error: {e}")
    st.stop()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page configuration (must be first Streamlit command)
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Yukti AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------------------
# Database setup (users, activity, metrics)
# ----------------------------------------------------------------------
def init_db():
    """Initialize database tables if they don't exist."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # User activity log
        c.execute('''CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            model_key TEXT,
            prompt_length INTEGER,
            response_time_ms INTEGER,
            success BOOLEAN,
            error TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')

        # Model performance metrics (aggregated)
        c.execute('''CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key TEXT,
            period TEXT,
            period_start TIMESTAMP,
            total_requests INTEGER,
            avg_response_time_ms INTEGER,
            error_count INTEGER,
            UNIQUE(model_key, period, period_start)
        )''')

        # System metrics snapshots
        c.execute('''CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cpu REAL,
            memory REAL,
            disk REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # Knowledge base snapshots
        c.execute('''CREATE TABLE IF NOT EXISTS kb_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_count INTEGER,
            index_size INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        # Admin action log
        c.execute('''CREATE TABLE IF NOT EXISTS admin_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(admin_id) REFERENCES users(id)
        )''')

        # Create indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_timestamp ON user_activity(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_period ON model_metrics(model_key, period, period_start)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_kb_metrics_timestamp ON kb_metrics(timestamp)")

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.exception("Database initialization failed")
        st.error(f"Database error: {e}")
        st.stop()

# Run database init
init_db()

# ----------------------------------------------------------------------
# Auto‑create admin user from environment variables (if no users exist)
# ----------------------------------------------------------------------
if os.getenv("ADMIN_PASSWORD"):
    admin_username = os.getenv("ADMIN_USERNAME", "admin")
    admin_pass = os.getenv("ADMIN_PASSWORD")
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        # Check if any user exists
        c.execute("SELECT COUNT(*) FROM users")
        if c.fetchone()[0] == 0:
            import bcrypt
            hashed = bcrypt.hashpw(admin_pass.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            c.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
                (admin_username, hashed)
            )
            conn.commit()
            logger.info(f"Created default admin user: {admin_username}")
        conn.close()
    except Exception as e:
        logger.error(f"Failed to create default admin: {e}")

# ----------------------------------------------------------------------
# Password hashing (bcrypt)
# ----------------------------------------------------------------------
try:
    import bcrypt
except ImportError:
    st.error("bcrypt is required. Please add it to requirements.txt")
    st.stop()

def hash_password(password: str) -> str:
    """Return bcrypt hash of password."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password: str, hashed: str) -> bool:
    """Verify password against hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# ----------------------------------------------------------------------
# Authentication functions
# ----------------------------------------------------------------------
def authenticate(username: str, password: str) -> Tuple[bool, Optional[int], bool]:
    """
    Authenticate user.
    Returns (success, user_id, is_admin)
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id, password_hash, is_admin FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        if row and check_password(password, row[1]):
            return True, row[0], bool(row[2])
        return False, None, False
    except Exception as e:
        logger.exception(f"Authentication error for user {username}")
        return False, None, False

def register_user(username: str, password: str) -> Tuple[bool, str]:
    """Register a new user. Returns (success, message)."""
    if not username or not password:
        return False, "Username and password required"
    if len(username) < 3 or len(password) < 6:
        return False, "Username must be ≥3 chars, password ≥6 chars"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            return False, "Username already exists"
        hashed = hash_password(password)
        c.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)", (username, hashed, 0))
        conn.commit()
        conn.close()
        return True, "User created successfully"
    except Exception as e:
        logger.exception("Registration error")
        return False, f"Database error: {e}"

def log_user_activity(user_id: int, model_key: str, prompt_length: int,
                      response_time_ms: int, success: bool, error: str = None):
    """Log a user interaction for analytics."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "INSERT INTO user_activity (user_id, model_key, prompt_length, response_time_ms, success, error) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, model_key, prompt_length, response_time_ms, 1 if success else 0, error)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log user activity: {e}")

def log_admin_action(admin_id: int, action: str, details: str = ""):
    """Log an admin action for audit trail."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "INSERT INTO admin_actions (admin_id, action, details) VALUES (?, ?, ?)",
            (admin_id, action, details)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log admin action: {e}")

# ----------------------------------------------------------------------
# System metrics (requires psutil)
# ----------------------------------------------------------------------
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed; system metrics will be limited.")

def get_system_metrics() -> Dict[str, float]:
    """Return current CPU, memory, disk usage as percentages."""
    if not PSUTIL_AVAILABLE:
        return {"cpu": 0, "memory": 0, "disk": 0}
    try:
        return {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent,
        }
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        return {"cpu": 0, "memory": 0, "disk": 0}

def record_system_metrics():
    """Take a snapshot of system metrics and store in DB."""
    metrics = get_system_metrics()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "INSERT INTO system_metrics (cpu, memory, disk) VALUES (?, ?, ?)",
            (metrics["cpu"], metrics["memory"], metrics["disk"])
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record system metrics: {e}")

def record_kb_metrics():
    """Take a snapshot of knowledge base metrics and store in DB."""
    count = get_document_count()
    if count is None:
        return
    try:
        # Get index size (if exists)
        index_size = 0
        if VECTORDB_PATH.exists():
            for f in VECTORDB_PATH.glob("*"):
                if f.is_file():
                    index_size += f.stat().st_size
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute(
            "INSERT INTO kb_metrics (doc_count, index_size) VALUES (?, ?)",
            (count, index_size)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record KB metrics: {e}")

# ----------------------------------------------------------------------
# Admin dashboard functions (CRUD)
# ----------------------------------------------------------------------
def get_all_users() -> List[Dict[str, Any]]:
    """Return list of all users (excluding password hash)."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id, username, is_admin, created_at FROM users ORDER BY id")
        rows = c.fetchall()
        conn.close()
        return [
            {"id": r[0], "username": r[1], "is_admin": bool(r[2]), "created_at": r[3]}
            for r in rows
        ]
    except Exception as e:
        logger.exception("Failed to fetch users")
        return []

def update_user(user_id: int, username: str, password: Optional[str], is_admin: bool) -> Tuple[bool, str]:
    """Update user details. If password is None/empty, keep existing."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if password:
            hashed = hash_password(password)
            c.execute(
                "UPDATE users SET username = ?, password_hash = ?, is_admin = ? WHERE id = ?",
                (username, hashed, 1 if is_admin else 0, user_id)
            )
        else:
            c.execute(
                "UPDATE users SET username = ?, is_admin = ? WHERE id = ?",
                (username, 1 if is_admin else 0, user_id)
            )
        conn.commit()
        conn.close()
        return True, "User updated"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        logger.exception("Failed to update user")
        return False, str(e)

def delete_user(user_id: int) -> Tuple[bool, str]:
    """Delete a user and their activity logs (cascade)."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("DELETE FROM user_activity WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True, "User deleted"
    except Exception as e:
        logger.exception("Failed to delete user")
        return False, str(e)

def create_user(username: str, password: str, is_admin: bool) -> Tuple[bool, str]:
    """Create a new user."""
    if not username or not password:
        return False, "Username and password required"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            return False, "Username already exists"
        hashed = hash_password(password)
        c.execute(
            "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
            (username, hashed, 1 if is_admin else 0)
        )
        conn.commit()
        conn.close()
        return True, "User created"
    except Exception as e:
        logger.exception("Failed to create user")
        return False, str(e)

# ----------------------------------------------------------------------
# Analytics functions
# ----------------------------------------------------------------------
def get_model_performance(days: int = 7) -> pd.DataFrame:
    """Aggregate model performance over last N days."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT
                model_key,
                DATE(timestamp) as day,
                COUNT(*) as requests,
                AVG(response_time_ms) as avg_response,
                SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as errors
            FROM user_activity
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY model_key, day
            ORDER BY day DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get model performance")
        return pd.DataFrame()

def get_user_activity_summary(days: int = 30) -> pd.DataFrame:
    """Daily active users over last N days."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT
                DATE(timestamp) as day,
                COUNT(DISTINCT user_id) as active_users,
                COUNT(*) as total_queries
            FROM user_activity
            WHERE timestamp >= datetime('now', '-{days} days')
            GROUP BY day
            ORDER BY day
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get user activity summary")
        return pd.DataFrame()

def get_system_metrics_history(hours: int = 24) -> pd.DataFrame:
    """System metrics over last N hours."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT
                datetime(timestamp) as time,
                cpu,
                memory,
                disk
            FROM system_metrics
            WHERE timestamp >= datetime('now', '-{hours} hours')
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get system metrics history")
        return pd.DataFrame()

def get_kb_history(days: int = 30) -> pd.DataFrame:
    """Knowledge base size history."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT
                datetime(timestamp) as time,
                doc_count,
                index_size
            FROM kb_metrics
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get KB history")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.is_admin = False
    st.session_state.admin_mode = False
    st.session_state.messages = []
    st.session_state.tasks = {}
    st.session_state.conversation_language = None
    st.session_state.kb_status = get_kb_detailed_status()
    st.session_state.clear_input = False
    st.session_state.processing = False
    st.session_state.uploaded_file = None

# ----------------------------------------------------------------------
# Login / Signup UI (if not logged in)
# ----------------------------------------------------------------------
if not st.session_state.logged_in:
    st.markdown("# 🚀 Yukti AI")
    st.markdown("## Please log in to continue")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            if submitted:
                success, user_id, is_admin = authenticate(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.session_state.is_admin = is_admin
                    st.session_state.admin_mode = is_admin  # admin sees dashboard
                    show_success_toast(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username")
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submitted = st.form_submit_button("Sign Up")
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, msg = register_user(new_username, new_password)
                    if success:
                        show_success_toast("Account created! Please log in.")
                        st.rerun()
                    else:
                        st.error(msg)
    st.stop()  # Do not proceed to main app

# ----------------------------------------------------------------------
# Main app (after login)
# ----------------------------------------------------------------------
# Sidebar (common for both admin and normal users)
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    if st.button("🚪 Logout", use_container_width=True):
        # Log out
        for key in ["logged_in", "user_id", "username", "is_admin", "admin_mode", "messages", "tasks"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    if st.session_state.is_admin:
        # Admin toggle: switch between dashboard and chat
        admin_mode = st.checkbox("🛡️ Admin Mode", value=st.session_state.admin_mode)
        if admin_mode != st.session_state.admin_mode:
            st.session_state.admin_mode = admin_mode
            st.rerun()

    # Model selector (visible to both, but admin can also see it in dashboard)
    st.markdown("## 🧠 Model")
    try:
        model_options = get_available_models()
        if not model_options:
            st.warning("No models available. Check API keys.")
            model_options = ["Yukti‑Flash"]
    except Exception as e:
        st.error(f"Failed to fetch models: {e}")
        logger.exception("get_available_models")
        model_options = ["Yukti‑Flash"]

    display_names = [
        f"{m} – {MODELS.get(m, {}).get('description', 'No description')}"
        for m in model_options
    ]
    selected_display = st.selectbox(
        label="Select model",
        options=display_names,
        index=0,
        key="model_display",
        label_visibility="collapsed"
    )
    selected_model = model_options[display_names.index(selected_display)]
    st.session_state.selected_model = selected_model

    # File upload (only if model supports it)
    uploaded_file = None
    if selected_model in ["Yukti‑Video", "Yukti‑Image", "Yukti‑Audio"]:
        st.markdown("### 📎 Attach File")
        if selected_model == "Yukti‑Video":
            uploaded_file = st.file_uploader("Upload image (optional)", type=["png", "jpg", "jpeg"])
        elif selected_model == "Yukti‑Image":
            uploaded_file = st.file_uploader("Upload reference image (optional)", type=["png", "jpg", "jpeg"])
        elif selected_model == "Yukti‑Audio":
            uploaded_file = st.file_uploader("Upload audio (optional)", type=["mp3", "wav"])

    st.divider()

    # Knowledge base status
    st.markdown("### 📚 Knowledge Base")
    kb_status = st.session_state.kb_status
    if kb_status["ready"]:
        st.markdown(f"✅ **Active** ({kb_status['document_count']} docs)")
    else:
        st.markdown("❌ **Not Ready**")
        if kb_status["error"]:
            st.caption(f"Error: {kb_status['error']}")

    if st.button("🔄 Update KB", use_container_width=True):
        with st.spinner("Rebuilding..."):
            success = create_vector_db()
            if success:
                st.session_state.kb_status = get_kb_detailed_status()
                show_success_toast("Knowledge base updated")
                st.rerun()
            else:
                st.error("Update failed")

    st.divider()

    # Tasks (only if Zhipu available)
    if ZHIPU_AVAILABLE:
        st.markdown("### 📋 Tasks")
        if st.button("⟳ Refresh Tasks", use_container_width=True):
            st.rerun()
        try:
            active_tasks = get_active_tasks()
            for task_id, variant, status, progress in active_tasks:
                if task_id not in st.session_state.tasks:
                    st.session_state.tasks[task_id] = {
                        "variant": variant,
                        "status": status,
                        "progress": progress,
                        "result_url": None,
                        "error": None
                    }
                else:
                    st.session_state.tasks[task_id].update({
                        "status": status,
                        "progress": progress
                    })
        except Exception as e:
            st.warning(f"Could not fetch tasks: {e}")

        for task_id in list(st.session_state.tasks.keys()):
            render_task(task_id, st.session_state.tasks[task_id])
        st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        show_success_toast("Chat cleared")
        st.rerun()

# ----------------------------------------------------------------------
# Routing: Admin Dashboard or Normal Chat
# ----------------------------------------------------------------------
if st.session_state.admin_mode:
    # -------------------- Admin Dashboard --------------------
    st.title("🛡️ Admin Dashboard")
    st.caption(f"Logged in as {st.session_state.username} (Admin)")

    # Record metrics periodically (every 5 minutes, but we'll just do it on dashboard load)
    # For real‑time, we'll refresh within the dashboard
    record_system_metrics()
    record_kb_metrics()

    # Create tabs
    tabs = st.tabs([
        "📊 Overview",
        "👥 Users",
        "📈 Analytics",
        "📚 Knowledge Base",
        "📋 Tasks",
        "🤖 Agentic Insights",
        "⚙️ System"
    ])

    # --- Overview Tab ---
    with tabs[0]:
        st.subheader("Real‑time System Status")
        # Real‑time metrics placeholder
        metrics_placeholder = st.empty()
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            metric_card("Total Users", len(get_all_users()))
        with col2:
            metric_card("Knowledge Base Docs", get_document_count() or 0)
        with col3:
            active_tasks_count = len(get_active_tasks())
            metric_card("Active Tasks", active_tasks_count)
        with col4:
            # Last update time from KB index
            if VECTORDB_PATH.exists():
                mtime = datetime.fromtimestamp(VECTORDB_PATH.stat().st_mtime)
                metric_card("Last KB Update", mtime.strftime("%Y-%m-%d %H:%M"))
            else:
                metric_card("Last KB Update", "Never")

        # Real‑time system metrics loop (auto‑refresh)
        while st.session_state.admin_mode:
            with metrics_placeholder.container():
                sys_metrics = get_system_metrics()
                cols = st.columns(3)
                cols[0].metric("CPU Usage", f"{sys_metrics['cpu']:.1f}%")
                cols[1].metric("Memory Usage", f"{sys_metrics['memory']:.1f}%")
                cols[2].metric("Disk Usage", f"{sys_metrics['disk']:.1f}%")
            time.sleep(2)
            st.rerun()

    # --- Users Tab (CRUD) ---
    with tabs[1]:
        st.subheader("User Management")

        # Add new user form
        with st.expander("➕ Add New User"):
            with st.form("add_user_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_confirm = st.text_input("Confirm Password", type="password")
                new_is_admin = st.checkbox("Admin privileges")
                if st.form_submit_button("Create User"):
                    if new_password != new_confirm:
                        st.error("Passwords do not match")
                    else:
                        ok, msg = create_user(new_username, new_password, new_is_admin)
                        if ok:
                            log_admin_action(st.session_state.user_id, "CREATE_USER", new_username)
                            show_success_toast("User created")
                            st.rerun()
                        else:
                            st.error(msg)

        # User table
        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users)
            st.dataframe(df_users, use_container_width=True)

            # Edit/Delete for each user
            for user in users:
                if user["id"] == st.session_state.user_id:
                    continue  # cannot edit yourself here (use profile later)
                with st.expander(f"Edit {user['username']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_username = st.text_input("Username", value=user["username"], key=f"user_{user['id']}_name")
                        new_is_admin = st.checkbox("Admin", value=user["is_admin"], key=f"user_{user['id']}_admin")
                    with col2:
                        new_password = st.text_input("New Password (leave blank to keep)", type="password", key=f"user_{user['id']}_pass")
                        if st.button("Update", key=f"user_{user['id']}_update"):
                            ok, msg = update_user(user["id"], new_username, new_password or None, new_is_admin)
                            if ok:
                                log_admin_action(st.session_state.user_id, "UPDATE_USER", new_username)
                                show_success_toast("User updated")
                                st.rerun()
                            else:
                                st.error(msg)
                    if st.button("Delete", key=f"user_{user['id']}_delete"):
                        if confirm_dialog("Delete User", f"Delete {user['username']}? This cannot be undone."):
                            ok, msg = delete_user(user["id"])
                            if ok:
                                log_admin_action(st.session_state.user_id, "DELETE_USER", user["username"])
                                show_success_toast("User deleted")
                                st.rerun()
                            else:
                                st.error(msg)
        else:
            st.info("No users found.")

    # --- Analytics Tab ---
    with tabs[2]:
        st.subheader("Model Performance (Last 7 Days)")
        perf_df = get_model_performance(7)
        if not perf_df.empty:
            fig = px.line(perf_df, x="day", y="requests", color="model_key", title="Daily Requests per Model")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.line(perf_df, x="day", y="avg_response", color="model_key", title="Avg Response Time (ms)")
            st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.bar(perf_df, x="day", y="errors", color="model_key", title="Daily Errors")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No performance data yet.")

        st.subheader("User Activity")
        activity_df = get_user_activity_summary(30)
        if not activity_df.empty:
            fig4 = px.line(activity_df, x="day", y="active_users", title="Daily Active Users")
            st.plotly_chart(fig4, use_container_width=True)
            fig5 = px.line(activity_df, x="day", y="total_queries", title="Daily Queries")
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.info("No activity data yet.")

        st.subheader("System Metrics (Last 24h)")
        sys_df = get_system_metrics_history(24)
        if not sys_df.empty:
            fig6 = px.line(sys_df, x="time", y=["cpu", "memory", "disk"], title="System Resources")
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.info("No system metrics yet.")

    # --- Knowledge Base Tab ---
    with tabs[3]:
        st.subheader("Knowledge Base Analytics")
        col1, col2 = st.columns(2)
        with col1:
            metric_card("Current Documents", get_document_count() or 0)
        with col2:
            if VECTORDB_PATH.exists():
                size_mb = sum(f.stat().st_size for f in VECTORDB_PATH.glob("*") if f.is_file()) / (1024*1024)
                metric_card("Index Size", f"{size_mb:.2f} MB")
            else:
                metric_card("Index Size", "N/A")

        kb_df = get_kb_history(30)
        if not kb_df.empty:
            fig = px.line(kb_df, x="time", y="doc_count", title="Document Count Over Time")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No KB history yet. Rebuild to start tracking.")

        if st.button("🔄 Rebuild Knowledge Base Now"):
            if confirm_dialog("Rebuild KB", "This may take a few minutes. Continue?"):
                with st.spinner("Rebuilding..."):
                    success = create_vector_db()
                    if success:
                        st.session_state.kb_status = get_kb_detailed_status()
                        record_kb_metrics()
                        log_admin_action(st.session_state.user_id, "REBUILD_KB")
                        show_success_toast("Knowledge base rebuilt")
                        st.rerun()
                    else:
                        st.error("Rebuild failed")

    # --- Tasks Tab ---
    with tabs[4]:
        st.subheader("Active Tasks")
        active = get_active_tasks()
        if active:
            for task_id, variant, status, progress in active:
                with st.container():
                    cols = st.columns([2, 2, 2, 2])
                    cols[0].write(f"**{variant}**")
                    cols[1].write(f"`{task_id[:8]}`")
                    cols[2].write(status)
                    if status == 'processing':
                        cols[3].progress(progress/100, text=f"{progress}%")
                    else:
                        cols[3].write("—")
        else:
            st.info("No active tasks.")

        st.subheader("Task History")
        # Could add a query for completed/failed tasks

    # --- Agentic Insights Tab ---
    with tabs[5]:
        st.subheader("AI‑Powered Insights")

        # Anomaly detection (simple)
        perf_df = get_model_performance(1)  # last 24h
        if not perf_df.empty:
            avg_error_rate = perf_df['errors'].sum() / max(perf_df['requests'].sum(), 1)
            if avg_error_rate > 0.1:
                st.warning(f"⚠️ High error rate ({avg_error_rate:.1%}) in the last 24h.")

        # Load prediction (simple linear regression)
        activity_df = get_user_activity_summary(14)
        if len(activity_df) > 5:
            # Very simple forecast: average of last 7 days
            last_7_avg = activity_df.tail(7)['total_queries'].mean()
            st.info(f"📈 Forecast: ~{int(last_7_avg)} queries per day next week.")

        # Recommendations
        st.subheader("Recommendations")
        recs = []

        # Check knowledge base age
        if VECTORDB_PATH.exists():
            mtime = datetime.fromtimestamp(VECTORDB_PATH.stat().st_mtime)
            if (datetime.now() - mtime) > timedelta(days=7):
                recs.append(("Knowledge base is over 7 days old. Consider rebuilding.", "Rebuild Now", "REBUILD_KB"))

        # Check system load
        sys_metrics = get_system_metrics()
        if sys_metrics["cpu"] > 80:
            recs.append(("CPU usage >80%. Consider scaling up.", None, None))
        if sys_metrics["memory"] > 80:
            recs.append(("Memory usage >80%. Consider increasing RAM.", None, None))

        # Check concurrency limits
        active_tasks = get_active_tasks()
        if len(active_tasks) > 10:
            recs.append(("High number of active tasks. Check for stuck tasks.", "Clear Stale Tasks", "CLEAR_TASKS"))

        if recs:
            for rec, action, action_code in recs:
                cols = st.columns([4, 1])
                cols[0].write(f"• {rec}")
                if action:
                    if cols[1].button(action, key=f"rec_{hash(rec)}"):
                        if action_code == "REBUILD_KB":
                            st.session_state.need_rebuild = True
                            st.rerun()
                        elif action_code == "CLEAR_TASKS":
                            if confirm_dialog("Clear Tasks", "Delete all stale tasks?"):
                                # Implement clear logic
                                show_success_toast("Tasks cleared (not implemented)")
        else:
            st.success("All systems nominal.")

    # --- System Tab ---
    with tabs[6]:
        st.subheader("System Control")
        if st.button("📥 Export Analytics Data"):
            # Export all analytics tables as CSV
            import io
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                get_model_performance(30).to_excel(writer, sheet_name="Model Performance")
                get_user_activity_summary(30).to_excel(writer, sheet_name="User Activity")
                get_system_metrics_history(24).to_excel(writer, sheet_name="System Metrics")
                get_kb_history(30).to_excel(writer, sheet_name="KB History")
            st.download_button(
                "Download Export",
                data=output.getvalue(),
                file_name=f"yukti_analytics_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        st.subheader("Log Viewer")
        log_viewer("updater.log", max_lines=50)

    # Stop here so chat doesn't render
    st.stop()

else:
    # -------------------- Normal Chat Interface --------------------
    # Display chat messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "media" in msg:
                for media in msg["media"]:
                    if media["type"] == "image":
                        st.image(media["url"], width=300)
                    elif media["type"] == "audio":
                        st.audio(media["url"])
                    elif media["type"] == "video":
                        st.video(media["url"])

    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Process message
        start_time = time.time()
        st.session_state.processing = True

        # Get uploaded file if any
        uploaded_file_obj = None
        if uploaded_file is not None:
            # Save to temp file
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                uploaded_file_obj = tmp.name

        # Detect language
        lang_info = detect_language(prompt)
        target_lang = lang_info['language']
        if lang_info['explicit_instruction']:
            st.session_state.conversation_language = lang_info['explicit_instruction']
        elif st.session_state.conversation_language is None:
            st.session_state.conversation_language = target_lang

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            try:
                model_key = st.session_state.selected_model
                extra_kwargs = {}
                if uploaded_file_obj:
                    extra_kwargs["image_url"] = uploaded_file_obj

                if model_key in ["Yukti‑Flash", "Yukti‑Quantum"]:
                    if not check_kb_status():
                        answer = "Knowledge base not ready. Please update it first."
                        response_placeholder.markdown(answer)
                        result = {"type": "sync", "answer": answer}
                    else:
                        history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[-10:]
                        ]
                        result = think(prompt, history, model_key, language=st.session_state.conversation_language)
                else:
                    # Generation models
                    model = load_model(model_key)
                    if model_key == "Yukti‑Audio":
                        audio_path = model.invoke(prompt, voice="female", language=st.session_state.conversation_language, **extra_kwargs)
                        with open(audio_path, "rb") as f:
                            audio_bytes = f.read()
                        st.audio(audio_bytes, format="audio/wav")
                        st.download_button("📥 Download Audio", data=audio_bytes, file_name="yukti_audio.wav")
                        answer = "Audio generated."
                        result = {"type": "sync", "answer": answer, "media": [{"type": "audio", "url": audio_path}]}
                    elif model_key == "Yukti‑Image":
                        image_url = model.invoke(prompt, **extra_kwargs)
                        st.image(image_url, width=300)
                        st.download_button("📥 Download Image", data=requests.get(image_url).content, file_name="yukti_image.png")
                        answer = "Image generated."
                        result = {"type": "sync", "answer": answer, "media": [{"type": "image", "url": image_url}]}
                    elif model_key == "Yukti‑Video":
                        task_id = model.invoke(prompt, language=st.session_state.conversation_language, **extra_kwargs)
                        st.session_state.tasks[task_id] = {
                            "variant": "Yukti‑Video",
                            "status": "submitted",
                            "progress": 0,
                            "result_url": None,
                            "error": None
                        }
                        answer = f"Video task started: `{task_id}`"
                        result = {"type": "async", "task_id": task_id}
                    else:
                        history = [
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages[-10:]
                        ]
                        result = think(prompt, history, model_key, language=st.session_state.conversation_language)

                if result.get("type") == "sync":
                    answer = result.get("answer", "")
                    if result.get("monologue"):
                        with st.expander("Show reasoning"):
                            st.markdown(result["monologue"])
                    response_placeholder.markdown(answer)
                    # Log activity
                    elapsed = int((time.time() - start_time) * 1000)
                    log_user_activity(
                        st.session_state.user_id,
                        model_key,
                        len(prompt),
                        elapsed,
                        True
                    )
                    # Store in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "media": result.get("media", [])
                    })
                elif result.get("type") == "async":
                    st.info(f"Task {result['task_id']} submitted. Check sidebar for progress.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("Chat error")
                # Log failure
                elapsed = int((time.time() - start_time) * 1000)
                log_user_activity(
                    st.session_state.user_id,
                    st.session_state.selected_model,
                    len(prompt),
                    elapsed,
                    False,
                    str(e)
                )

        st.session_state.processing = False
        st.rerun()
