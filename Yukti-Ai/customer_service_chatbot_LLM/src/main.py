"""
Main Streamlit application for Yukti AI.
Includes hardcoded admin credentials (admin1234/admin1234) and a full admin dashboard.
Knowledge base updates automatically in the background (manual update button removed).
All tabs display real-time data with comprehensive metrics and insights.
Video tasks are isolated per user.
Web scraping sources can be configured in the admin panel.
"""

import os
import sys
import time
import logging
import sqlite3
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import bcrypt
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Local imports
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
except ImportError:
    BASE_DIR = PROJECT_ROOT
    VECTORDB_PATH = BASE_DIR / "faiss_index"
    DB_PATH = BASE_DIR / "yukti_tasks.db"
    RETRIEVAL_CACHE_TTL = 3600
    TASK_POLL_INTERVAL = 5
    ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

try:
    from langchain_helper import (
        create_vector_db,
        get_kb_detailed_status,
        get_document_count,
        check_kb_status,
    )
except ImportError:
    def create_vector_db(): return False
    def get_kb_detailed_status(): return {"ready": False, "error": "Not implemented"}
    def get_document_count(): return None
    def check_kb_status(): return False

try:
    from think import think
except ImportError:
    def think(*args, **kwargs): return {"type": "sync", "answer": "Think module not available"}

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
except ImportError:
    ZHIPU_AVAILABLE = False
    GEMINI_AVAILABLE = False
    MODELS = {}
    def get_available_models(): return []
    def get_active_tasks(user_id=None): return []
    def get_task_status(*args): return None
    def load_model(*args): return None

try:
    from language_detector import detect_language
except ImportError:
    def detect_language(text): return {"language": "en", "method": "fallback", "explicit_instruction": None}

# Import UI helpers
try:
    from ui_helpers import render_task, show_error_toast, show_success_toast, confirm_dialog, metric_card, log_viewer
except ImportError:
    def render_task(*args): pass
    def show_error_toast(msg): st.error(msg)
    def show_success_toast(msg): st.success(msg)
    def confirm_dialog(title, msg): return st.checkbox(msg)
    def metric_card(label, value, delta=None): st.metric(label, value, delta)
    def log_viewer(path, max_lines): st.info("Log viewer not available")

# Import knowledge updater components
try:
    from knowledge_updater.builder import rebuild_index
    from knowledge_updater.config import UPLOADS_PATH, SOURCES as KB_SOURCES
    KB_UPDATER_AVAILABLE = True
except ImportError:
    KB_UPDATER_AVAILABLE = False
    def rebuild_index(): return False
    # Fallback definition to prevent NameError in updater thread
    UPLOADS_PATH = BASE_DIR / "data" / "uploads"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------
st.set_page_config(page_title="Yukti AI", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# ----------------------------------------------------------------------
# Database initialization (ensure tasks table has user_id)
# ----------------------------------------------------------------------
def init_db():
    """Create all tables if they don't exist."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()

        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

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

        c.execute('''CREATE TABLE IF NOT EXISTS system_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cpu REAL,
            memory REAL,
            disk REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS kb_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_count INTEGER,
            index_size INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS admin_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id INTEGER,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(admin_id) REFERENCES users(id)
        )''')

        c.execute('''CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            variant TEXT,
            model TEXT,
            status TEXT,
            progress INTEGER,
            result_url TEXT,
            error TEXT,
            created_at TIMESTAMP,
            completed_at TIMESTAMP,
            user_id INTEGER
        )''')

        # Indexes
        c.execute("CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_users_is_admin ON users(is_admin)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_user ON user_activity(user_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_user_activity_timestamp ON user_activity(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_model_metrics_period ON model_metrics(model_key, period, period_start)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_kb_metrics_timestamp ON kb_metrics(timestamp)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user_id ON tasks(user_id)")

        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.exception("Database initialization failed")
        st.error(f"Database error: {e}")
        st.stop()

init_db()

# ----------------------------------------------------------------------
# Ensure default admin user exists
# ----------------------------------------------------------------------
def ensure_admin_user():
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = 'admin1234'")
        if not c.fetchone():
            hashed = bcrypt.hashpw(b'admin1234', bcrypt.gensalt()).decode('utf-8')
            c.execute(
                "INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 1)",
                ('admin1234', hashed)
            )
            conn.commit()
            logger.info("Default admin user created (admin1234/admin1234)")
        conn.close()
    except Exception as e:
        logger.error(f"Failed to create default admin: {e}")

ensure_admin_user()

# ----------------------------------------------------------------------
# Authentication helpers
# ----------------------------------------------------------------------
def authenticate(username: str, password: str) -> Tuple[bool, Optional[int], bool]:
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id, password_hash, is_admin FROM users WHERE username = ?", (username,))
        row = c.fetchone()
        conn.close()
        if row and bcrypt.checkpw(password.encode('utf-8'), row[1].encode('utf-8')):
            return True, row[0], bool(row[2])
        return False, None, False
    except Exception as e:
        logger.exception("Authentication error")
        return False, None, False

def register_user(username: str, password: str) -> Tuple[bool, str]:
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
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        c.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, 0)", (username, hashed))
        conn.commit()
        conn.close()
        return True, "User created successfully"
    except Exception as e:
        logger.exception("Registration error")
        return False, f"Database error: {e}"

def log_user_activity(user_id, model_key, prompt_length, response_time_ms, success, error=None):
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
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("INSERT INTO admin_actions (admin_id, action, details) VALUES (?, ?, ?)", (admin_id, action, details))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to log admin action: {e}")

# ----------------------------------------------------------------------
# System metrics
# ----------------------------------------------------------------------
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

def get_system_metrics():
    if not PSUTIL_AVAILABLE:
        return {"cpu": 0, "memory": 0, "disk": 0}
    try:
        return {
            "cpu": psutil.cpu_percent(interval=0.1),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent,
        }
    except:
        return {"cpu": 0, "memory": 0, "disk": 0}

def record_system_metrics():
    metrics = get_system_metrics()
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("INSERT INTO system_metrics (cpu, memory, disk) VALUES (?, ?, ?)", (metrics["cpu"], metrics["memory"], metrics["disk"]))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record system metrics: {e}")

def record_kb_metrics():
    count = get_document_count()
    if count is None:
        return
    try:
        index_size = 0
        if VECTORDB_PATH.exists():
            for f in VECTORDB_PATH.glob("*"):
                if f.is_file():
                    index_size += f.stat().st_size
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("INSERT INTO kb_metrics (doc_count, index_size) VALUES (?, ?)", (count, index_size))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to record KB metrics: {e}")

# ----------------------------------------------------------------------
# Enhanced admin data functions (with error handling and media columns)
# ----------------------------------------------------------------------
def get_all_users():
    try:
        conn = sqlite3.connect(str(DB_PATH))
        df = pd.read_sql_query("SELECT id, username, is_admin, created_at FROM users ORDER BY id", conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to fetch users")
        return pd.DataFrame(columns=["id", "username", "is_admin", "created_at"])

def get_user_message_counts():
    """Return user stats including message counts and media generation counts."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = """
            SELECT
                u.id,
                u.username,
                u.is_admin,
                u.created_at,
                COUNT(a.id) AS total_messages,
                SUM(CASE WHEN a.success = 1 THEN 1 ELSE 0 END) AS successful,
                SUM(CASE WHEN a.success = 0 THEN 1 ELSE 0 END) AS failed,
                AVG(a.response_time_ms) AS avg_response_time,
                MAX(a.timestamp) AS last_active,
                (
                    SELECT model_key
                    FROM user_activity a2
                    WHERE a2.user_id = u.id
                    GROUP BY a2.model_key
                    ORDER BY COUNT(*) DESC
                    LIMIT 1
                ) AS most_used_model,
                COUNT(DISTINCT CASE WHEN t.variant = 'Yukti‑Image' THEN t.task_id END) AS images_created,
                COUNT(DISTINCT CASE WHEN t.variant = 'Yukti‑Video' THEN t.task_id END) AS videos_created,
                COUNT(DISTINCT CASE WHEN t.variant = 'Yukti‑Audio' THEN t.task_id END) AS audio_created
            FROM users u
            LEFT JOIN user_activity a ON u.id = a.user_id
            LEFT JOIN tasks t ON u.id = t.user_id
            GROUP BY u.id
            ORDER BY total_messages DESC
        """
        df = pd.read_sql_query(query, conn)
        conn.close()

        if not df.empty:
            numeric_cols = ['total_messages', 'successful', 'failed', 'avg_response_time',
                            'images_created', 'videos_created', 'audio_created']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            df['success_rate'] = (df['successful'] / df['total_messages'].replace(0, 1)) * 100

            df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
            df['last_active'] = pd.to_datetime(df['last_active']).dt.strftime('%Y-%m-%d %H:%M')
            df['avg_response_time'] = df['avg_response_time'].round(0).astype(int)
        else:
            df = pd.DataFrame(columns=[
                'id', 'username', 'is_admin', 'created_at', 'total_messages',
                'successful', 'failed', 'avg_response_time', 'last_active',
                'most_used_model', 'success_rate',
                'images_created', 'videos_created', 'audio_created'
            ])
        return df
    except Exception as e:
        logger.exception("get_user_message_counts failed")
        if st.session_state.get('debug_mode', False):
            st.error(f"🔴 Error in get_user_message_counts: {e}")
        return pd.DataFrame(columns=[
            'id', 'username', 'is_admin', 'created_at', 'total_messages',
            'successful', 'failed', 'avg_response_time', 'last_active',
            'most_used_model', 'success_rate',
            'images_created', 'videos_created', 'audio_created'
        ])

def update_user(user_id, username, password, is_admin):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        if password:
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            c.execute("UPDATE users SET username = ?, password_hash = ?, is_admin = ? WHERE id = ?",
                      (username, hashed, 1 if is_admin else 0, user_id))
        else:
            c.execute("UPDATE users SET username = ?, is_admin = ? WHERE id = ?",
                      (username, 1 if is_admin else 0, user_id))
        conn.commit()
        conn.close()
        return True, "User updated"
    except sqlite3.IntegrityError:
        return False, "Username already exists"
    except Exception as e:
        logger.exception("Update user failed")
        return False, str(e)

def delete_user(user_id):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("DELETE FROM user_activity WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM tasks WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()
        conn.close()
        return True, "User deleted"
    except Exception as e:
        logger.exception("Delete user failed")
        return False, str(e)

def create_user(username, password, is_admin):
    if not username or not password:
        return False, "Username and password required"
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username = ?", (username,))
        if c.fetchone():
            conn.close()
            return False, "Username already exists"
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        c.execute("INSERT INTO users (username, password_hash, is_admin) VALUES (?, ?, ?)",
                  (username, hashed, 1 if is_admin else 0))
        conn.commit()
        conn.close()
        return True, "User created"
    except Exception as e:
        logger.exception("Create user failed")
        return False, str(e)

def get_model_performance(days=30):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT model_key, DATE(timestamp) as day,
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
        return pd.DataFrame(columns=["model_key", "day", "requests", "avg_response", "errors"])

def get_user_activity_summary(days=30):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT DATE(timestamp) as day,
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
        return pd.DataFrame(columns=["day", "active_users", "total_queries"])

def get_system_metrics_history(hours=24):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT datetime(timestamp) as time, cpu, memory, disk
            FROM system_metrics
            WHERE timestamp >= datetime('now', '-{hours} hours')
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get system metrics history")
        return pd.DataFrame(columns=["time", "cpu", "memory", "disk"])

def get_kb_history(days=30):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        query = f"""
            SELECT datetime(timestamp) as time, doc_count, index_size
            FROM kb_metrics
            WHERE timestamp >= datetime('now', '-{days} days')
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get KB history")
        return pd.DataFrame(columns=["time", "doc_count", "index_size"])

def get_task_history(user_id=None, limit=100):
    try:
        conn = sqlite3.connect(str(DB_PATH))
        if user_id is None:
            query = f"""
                SELECT task_id, variant, model, status, progress, result_url, error, created_at, completed_at
                FROM tasks
                ORDER BY created_at DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn)
        else:
            query = f"""
                SELECT task_id, variant, model, status, progress, result_url, error, created_at, completed_at
                FROM tasks
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT {limit}
            """
            df = pd.read_sql_query(query, conn, params=(user_id,))
        conn.close()
        return df
    except Exception as e:
        logger.exception("Failed to get task history")
        return pd.DataFrame(columns=["task_id", "variant", "model", "status", "progress", "result_url", "error", "created_at", "completed_at"])

def get_database_stats():
    stats = {}
    try:
        conn = sqlite3.connect(str(DB_PATH))
        c = conn.cursor()
        tables = ["users", "user_activity", "tasks", "system_metrics", "kb_metrics"]
        for table in tables:
            c.execute(f"SELECT COUNT(*) FROM {table}")
            stats[table] = c.fetchone()[0]
        conn.close()
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
    return stats

# ----------------------------------------------------------------------
# Web scraping configuration management
# ----------------------------------------------------------------------
def load_web_sources():
    """Load web scraping sources from JSON file (if exists)."""
    config_file = PROJECT_ROOT / "knowledge_updater" / "web_sources.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load web sources: {e}")
    # Default structure
    return {"websites": []}

def save_web_sources(sources):
    """Save web scraping sources to JSON file."""
    config_file = PROJECT_ROOT / "knowledge_updater" / "web_sources.json"
    try:
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(sources, f, indent=2)
        logger.info("Web sources saved to web_sources.json")
        return True
    except Exception as e:
        logger.error(f"Failed to save web sources: {e}")
        return False

# ----------------------------------------------------------------------
# Background knowledge base updater thread (FIXED)
# ----------------------------------------------------------------------
class KnowledgeBaseUpdater(threading.Thread):
    """Background thread that checks for changes and rebuilds index if needed."""
    def __init__(self, check_interval=2):
        super().__init__(daemon=True)
        self.check_interval = check_interval
        self.last_rebuild_time = 0
        self.last_known_mtimes = {}  # file -> mtime
        self.running = True

    def run(self):
        logger.info("Knowledge base auto-updater started.")
        while self.running:
            try:
                self._check_and_rebuild()
            except Exception as e:
                logger.exception("Error in KB updater thread")
            time.sleep(self.check_interval)

    def _check_and_rebuild(self):
        """Check if any source file has changed; if so, rebuild."""
        # Check dataset.csv (correct path)
        dataset_path = PROJECT_ROOT / "dataset" / "dataset.csv"
        if dataset_path.exists():
            mtime = dataset_path.stat().st_mtime
            if dataset_path not in self.last_known_mtimes or self.last_known_mtimes[dataset_path] != mtime:
                logger.info(f"Change detected in {dataset_path}. Scheduling rebuild.")
                self._trigger_rebuild()
                self.last_known_mtimes[dataset_path] = mtime
                return

        # Check uploads directory (UPLOADS_PATH now properly defined)
        if UPLOADS_PATH.exists():
            for file_path in UPLOADS_PATH.glob("*"):
                if file_path.is_file():
                    mtime = file_path.stat().st_mtime
                    if file_path not in self.last_known_mtimes or self.last_known_mtimes[file_path] != mtime:
                        logger.info(f"Change detected in {file_path}. Scheduling rebuild.")
                        self._trigger_rebuild()
                        self.last_known_mtimes[file_path] = mtime
                        return

    def _trigger_rebuild(self):
        """Rebuild index if at least 60 seconds passed since last rebuild."""
        now = time.time()
        if now - self.last_rebuild_time < 60:
            logger.info("Skipping rebuild – last rebuild too recent.")
            return
        logger.info("Auto-rebuilding knowledge base...")
        if rebuild_index():
            self.last_rebuild_time = now
            # Update session state status
            if 'kb_status' in st.session_state:
                st.session_state.kb_status = get_kb_detailed_status()
            # Record metrics
            record_kb_metrics()
            logger.info("Auto-rebuild completed.")
        else:
            logger.error("Auto-rebuild failed.")

    def stop(self):
        self.running = False

# Start background updater if not already running
if 'kb_updater' not in st.session_state:
    st.session_state.kb_updater = KnowledgeBaseUpdater(check_interval=2)
    st.session_state.kb_updater.start()

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
    st.session_state.last_metrics_record = 0
    st.session_state.debug_mode = False
    st.session_state.auto_refresh = False
    # Load web sources
    st.session_state.web_sources = load_web_sources()

# ----------------------------------------------------------------------
# Login / Signup UI
# ----------------------------------------------------------------------
if not st.session_state.logged_in:
    st.title("Yukti AI")
    st.subheader("Please log in to continue")
    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            if st.form_submit_button("Login"):
                success, user_id, is_admin = authenticate(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.session_state.is_admin = is_admin
                    st.session_state.admin_mode = is_admin
                    st.rerun()
                else:
                    st.error("Invalid username or password")

    with tab2:
        with st.form("signup_form"):
            new_username = st.text_input("Username")
            new_password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            if st.form_submit_button("Sign Up"):
                if new_password != confirm_password:
                    st.error("Passwords do not match")
                else:
                    ok, msg = register_user(new_username, new_password)
                    if ok:
                        st.success("Account created! Please log in.")
                        st.rerun()
                    else:
                        st.error(msg)
    st.stop()

# ----------------------------------------------------------------------
# Main app after login
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state.username}")
    if st.button("🚪 Logout", use_container_width=True):
        # Stop background updater before logout
        if 'kb_updater' in st.session_state:
            st.session_state.kb_updater.stop()
        for key in list(st.session_state.keys()):
            if key in ["logged_in", "user_id", "username", "is_admin", "admin_mode", "messages", "tasks", "debug_mode"]:
                del st.session_state[key]
        st.rerun()

    if st.session_state.is_admin:
        admin_mode = st.checkbox("🛡️ Admin Mode", value=st.session_state.admin_mode)
        if admin_mode != st.session_state.admin_mode:
            st.session_state.admin_mode = admin_mode
            st.rerun()

        debug_mode = st.checkbox("🔧 Debug Mode", value=st.session_state.get("debug_mode", False))
        if debug_mode != st.session_state.debug_mode:
            st.session_state.debug_mode = debug_mode
            st.rerun()

        auto_refresh = st.checkbox("🔄 Auto-refresh (every 10s)", value=st.session_state.get("auto_refresh", False))
        if auto_refresh != st.session_state.auto_refresh:
            st.session_state.auto_refresh = auto_refresh
            st.rerun()

    st.markdown("## 🧠 Model")
    model_options = get_available_models() or ["Yukti‑Flash"]
    display_names = [f"{m} – {MODELS.get(m, {}).get('description', 'No description')}" for m in model_options]
    selected_display = st.selectbox("Select model", display_names, index=0, label_visibility="collapsed")
    selected_model = model_options[display_names.index(selected_display)]
    st.session_state.selected_model = selected_model

    uploaded_file = None
    if selected_model in ["Yukti‑Video", "Yukti‑Image", "Yukti‑Audio"]:
        st.markdown("### 📎 Attach File")
        if selected_model == "Yukti‑Video":
            uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
        elif selected_model == "Yukti‑Image":
            uploaded_file = st.file_uploader("Upload reference image", type=["png", "jpg", "jpeg"])
        elif selected_model == "Yukti‑Audio":
            uploaded_file = st.file_uploader("Upload audio", type=["mp3", "wav"])

    st.divider()
    st.markdown("### 📚 Knowledge Base")
    kb_status = st.session_state.kb_status
    if kb_status["ready"]:
        st.markdown(f"✅ **Active** ({kb_status['document_count']} docs)")
    else:
        st.markdown("❌ **Not Ready**")
        if kb_status["error"]:
            st.caption(f"Error: {kb_status['error']}")
    # Manual update button removed – auto-update only
    st.divider()

    if ZHIPU_AVAILABLE:
        st.markdown("### 📋 Tasks")
        if st.button("⟳ Refresh Tasks", use_container_width=True):
            st.rerun()
        try:
            active_tasks = get_active_tasks(st.session_state.user_id)
            for task_id, variant, status, progress in active_tasks:
                if task_id not in st.session_state.tasks:
                    st.session_state.tasks[task_id] = {"variant": variant, "status": status, "progress": progress}
                else:
                    st.session_state.tasks[task_id].update({"status": status, "progress": progress})
        except Exception as e:
            if st.session_state.debug_mode:
                st.error(f"Task fetch error: {e}")
        for task_id, info in st.session_state.tasks.items():
            render_task(task_id, info)
        st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat cleared")
        st.rerun()

# ----------------------------------------------------------------------
# Admin dashboard or chat
# ----------------------------------------------------------------------
if st.session_state.admin_mode:
    # -------------------- Admin Dashboard --------------------
    st.title("🛡️ Admin Dashboard")
    st.caption(f"Logged in as {st.session_state.username} (Admin)")

    # Record metrics periodically
    now = time.time()
    if now - st.session_state.last_metrics_record > 60:
        record_system_metrics()
        record_kb_metrics()
        st.session_state.last_metrics_record = now

    # Database diagnostics
    with st.expander("🔍 Database Diagnostics", expanded=False):
        stats = get_database_stats()
        cols = st.columns(5)
        cols[0].metric("Users", stats.get("users", 0))
        cols[1].metric("User Activity", stats.get("user_activity", 0))
        cols[2].metric("Tasks", stats.get("tasks", 0))
        cols[3].metric("System Metrics", stats.get("system_metrics", 0))
        cols[4].metric("KB Metrics", stats.get("kb_metrics", 0))

    tabs = st.tabs(["📊 Overview", "👥 Users", "📈 Analytics", "📚 Knowledge Base", "📋 Tasks", "🤖 Insights", "⚙️ System"])

    # ----- Overview Tab -----
    with tabs[0]:
        st.subheader("System Status")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Users", len(get_all_users()))
        with col2:
            st.metric("KB Docs", get_document_count() or 0)
        with col3:
            st.metric("Active Tasks", len(get_active_tasks(st.session_state.user_id)))
        with col4:
            if VECTORDB_PATH.exists():
                mtime = datetime.fromtimestamp(VECTORDB_PATH.stat().st_mtime)
                st.metric("Last KB Update", mtime.strftime("%Y-%m-%d %H:%M"))
            else:
                st.metric("Last KB Update", "Never")

        # Live system metrics
        sys_metrics = get_system_metrics()
        cols = st.columns(3)
        cols[0].metric("CPU", f"{sys_metrics['cpu']:.1f}%")
        cols[1].metric("Memory", f"{sys_metrics['memory']:.1f}%")
        cols[2].metric("Disk", f"{sys_metrics['disk']:.1f}%")

    # ----- Users Tab -----
    with tabs[1]:
        st.subheader("User Management")

        with st.expander("➕ Add New User", expanded=False):
            with st.form("add_user_form"):
                col1, col2 = st.columns(2)
                with col1:
                    new_username = st.text_input("Username", key="add_user_username")
                    new_password = st.text_input("Password", type="password", key="add_user_pass")
                with col2:
                    confirm_password = st.text_input("Confirm Password", type="password", key="add_user_confirm")
                    new_is_admin = st.checkbox("Admin", key="add_user_admin")
                submitted = st.form_submit_button("Create User")
                if submitted:
                    if new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif not new_username or not new_password:
                        st.error("Username and password required")
                    else:
                        ok, msg = create_user(new_username, new_password, new_is_admin)
                        if ok:
                            log_admin_action(st.session_state.user_id, "CREATE_USER", new_username)
                            st.success("User created")
                            st.rerun()
                        else:
                            st.error(msg)

        users_df = get_user_message_counts()
        if users_df.empty:
            st.info("👥 No users found. Create one using the form above.")
        else:
            available_cols = users_df.columns.tolist()
            display_columns = ['username', 'is_admin', 'created_at', 'total_messages']
            optional_cols = ['success_rate', 'avg_response_time', 'last_active', 'most_used_model',
                             'images_created', 'videos_created', 'audio_created']
            for col in optional_cols:
                if col in available_cols:
                    display_columns.append(col)

            display_df = users_df[display_columns].copy()
            if 'success_rate' in display_df.columns:
                display_df['success_rate'] = display_df['success_rate'].round(1).astype(str) + '%'
            rename_map = {
                'username': 'Username',
                'is_admin': 'Admin',
                'created_at': 'Registered',
                'total_messages': 'Total Msgs',
                'success_rate': 'Success %',
                'avg_response_time': 'Avg Resp (ms)',
                'last_active': 'Last Active',
                'most_used_model': 'Fav Model',
                'images_created': 'Images',
                'videos_created': 'Videos',
                'audio_created': 'Audio'
            }
            display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})
            st.dataframe(display_df, width='stretch')

            if st.session_state.debug_mode:
                with st.expander("🔍 Raw User Data (Debug)"):
                    st.dataframe(users_df, width='stretch')

            st.markdown("### ✏️ Edit / Delete Users")
            for _, row in users_df.iterrows():
                if row['username'] == st.session_state.username:
                    continue
                with st.expander(f"{row['username']} (ID: {row['id']})"):
                    col1, col2 = st.columns(2)
                    with col1:
                        new_name = st.text_input("Username", value=row['username'], key=f"name_{row['id']}")
                        new_admin = st.checkbox("Admin", value=bool(row['is_admin']), key=f"admin_{row['id']}")
                    with col2:
                        new_pass = st.text_input("New Password (leave blank to keep)", type="password", key=f"pass_{row['id']}")
                        if st.button("Update", key=f"update_{row['id']}"):
                            ok, msg = update_user(row['id'], new_name, new_pass or None, new_admin)
                            if ok:
                                log_admin_action(st.session_state.user_id, "UPDATE_USER", new_name)
                                st.success("User updated")
                                st.rerun()
                            else:
                                st.error(msg)
                    if st.button("🗑️ Delete", key=f"delete_{row['id']}"):
                        if st.checkbox(f"Confirm delete {row['username']}", key=f"confirm_{row['id']}"):
                            ok, msg = delete_user(row['id'])
                            if ok:
                                log_admin_action(st.session_state.user_id, "DELETE_USER", row['username'])
                                st.success("User deleted")
                                st.rerun()
                            else:
                                st.error(msg)

    # ----- Analytics Tab -----
    with tabs[2]:
        st.subheader("Essential Performance Metrics")
        conn = sqlite3.connect(str(DB_PATH))
        total_queries = pd.read_sql_query("SELECT COUNT(*) as cnt FROM user_activity", conn)['cnt'][0]
        total_errors = pd.read_sql_query("SELECT COUNT(*) as cnt FROM user_activity WHERE success=0", conn)['cnt'][0]
        avg_latency = pd.read_sql_query("SELECT AVG(response_time_ms) as avg FROM user_activity", conn)['avg'][0] or 0
        resolution_rate = ((total_queries - total_errors) / total_queries * 100) if total_queries > 0 else 0
        fallback_rate = (total_errors / total_queries * 100) if total_queries > 0 else 0
        conn.close()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Latency (ms)", f"{avg_latency:.0f}")
        col2.metric("Resolution Rate", f"{resolution_rate:.1f}%")
        col3.metric("Fallback Rate", f"{fallback_rate:.1f}%")
        col4.metric("Task Completion (Video)", "N/A")

        st.subheader("Model Performance Over Time")
        perf = get_model_performance(30)
        if perf.empty:
            st.info("No model performance data yet.")
        else:
            fig1 = px.line(perf, x="day", y="avg_response", color="model_key",
                           title="Average Response Time (ms) per Model")
            st.plotly_chart(fig1, use_container_width=True)
            perf['error_rate'] = perf['errors'] / perf['requests'] * 100
            fig2 = px.line(perf, x="day", y="error_rate", color="model_key",
                           title="Error Rate (%) per Model")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("User Activity & Engagement")
        act = get_user_activity_summary(30)
        if act.empty:
            st.info("No user activity data yet.")
        else:
            fig3 = px.line(act, x="day", y="active_users", title="Daily Active Users")
            st.plotly_chart(fig3, use_container_width=True)
            fig4 = px.line(act, x="day", y="total_queries", title="Daily Queries")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("System Health")
        sys_hist = get_system_metrics_history(24)
        if sys_hist.empty:
            st.info("No system metrics yet.")
        else:
            fig5 = px.line(sys_hist, x="time", y=["cpu", "memory", "disk"],
                           title="System Resources (24h)")
            st.plotly_chart(fig5, use_container_width=True)

        st.subheader("Model Quality Radar")
        st.caption("(Placeholder – requires faithfulness, toxicity, etc.)")
        categories = ['Intelligence', 'Coherence', 'Relevance', 'Fluency']
        values = [85, 90, 88, 92]
        fig6 = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself'))
        fig6.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,100])))
        st.plotly_chart(fig6, use_container_width=True)

    # ----- Knowledge Base Tab (with Web Scraping) -----
    with tabs[3]:
        st.subheader("Knowledge Base Management")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", get_document_count() or 0)
        with col2:
            if VECTORDB_PATH.exists():
                size = sum(f.stat().st_size for f in VECTORDB_PATH.glob("*") if f.is_file()) / (1024*1024)
                st.metric("Index Size", f"{size:.2f} MB")
            else:
                st.metric("Index Size", "N/A")

        kb_hist = get_kb_history(30)
        if kb_hist.empty:
            st.info("📚 No knowledge base history yet. It will appear after the first rebuild.")
        else:
            if 'time' in kb_hist.columns and 'doc_count' in kb_hist.columns:
                fig = px.line(kb_hist, x="time", y="doc_count", title="Document Count Over Time")
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("### ⚙️ Auto‑Update Status")
        st.success("✅ Auto‑update is running in background (checks every 2 seconds, rebuilds if changes detected).")
        # Manual rebuild button is removed – only auto-update.

        with st.expander("📤 Upload Files", expanded=False):
            st.markdown("Upload CSV, TXT, or PDF files to add to the knowledge base. Changes will be picked up automatically.")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["csv", "txt", "pdf"],
                accept_multiple_files=True,
                key="kb_upload"
            )
            if uploaded_files and st.button("Upload Files"):
                for uploaded_file in uploaded_files:
                    file_path = UPLOADS_PATH / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    st.success(f"Saved {uploaded_file.name}")
                st.info("Auto‑update will detect changes within 2 seconds and rebuild.")

        with st.expander("🌐 RSS & API", expanded=False):
            st.markdown("Configure RSS feeds and APIs.")
            if "web_sources_config" not in st.session_state:
                st.session_state.web_sources_config = KB_SOURCES.copy()
            config = st.session_state.web_sources_config
            st.subheader("RSS Feeds")
            for i, feed in enumerate(config.get("rss", [])):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    feed["name"] = st.text_input(f"Name", value=feed["name"], key=f"rss_name_{i}")
                    feed["url"] = st.text_input(f"URL", value=feed["url"], key=f"rss_url_{i}")
                with cols[1]:
                    feed["enabled"] = st.checkbox("Enabled", value=feed.get("enabled", True), key=f"rss_enabled_{i}")
                with cols[2]:
                    if st.button("Remove", key=f"rss_remove_{i}"):
                        config["rss"].pop(i)
                        st.rerun()
            if st.button("➕ Add RSS Feed"):
                config["rss"].append({"name": "", "url": "", "enabled": True})
                st.rerun()
            st.subheader("APIs")
            for i, api in enumerate(config.get("api", [])):
                cols = st.columns([3, 1, 1])
                with cols[0]:
                    api["name"] = st.text_input(f"Name", value=api["name"], key=f"api_name_{i}")
                    api["url"] = st.text_input(f"URL", value=api["url"], key=f"api_url_{i}")
                with cols[1]:
                    api["enabled"] = st.checkbox("Enabled", value=api.get("enabled", True), key=f"api_enabled_{i}")
                with cols[2]:
                    if st.button("Remove", key=f"api_remove_{i}"):
                        config["api"].pop(i)
                        st.rerun()
            if st.button("➕ Add API"):
                config["api"].append({"name": "", "url": "", "enabled": True})
                st.rerun()
            if st.button("💾 Save RSS/API Sources"):
                st.success("Configuration saved (in memory). For persistence, implement JSON storage.")
                st.rerun()

        # ----- Website Crawling Section -----
        with st.expander("🕷️ Website Crawling", expanded=False):
            st.markdown("Configure websites to crawl entirely. The system will discover and index all pages from these sites.")
            st.caption("Enable/disable each source with the checkbox. Click 'Save Web Sources' to persist changes.")

            web_sources = st.session_state.web_sources
            websites = web_sources.get("websites", [])

            for i, site in enumerate(websites):
                cols = st.columns([2, 3, 1, 1, 1])
                with cols[0]:
                    site["name"] = st.text_input("Name", value=site.get("name", ""), key=f"ws_name_{i}")
                with cols[1]:
                    site["url"] = st.text_input("URL", value=site.get("url", ""), key=f"ws_url_{i}")
                with cols[2]:
                    site["max_pages"] = st.number_input("Max Pages", value=site.get("max_pages", 50),
                                                         min_value=1, max_value=1000, key=f"ws_pages_{i}")
                with cols[3]:
                    site["use_javascript"] = st.checkbox("JS Required", value=site.get("use_javascript", False),
                                                          key=f"ws_js_{i}")
                with cols[4]:
                    site["enabled"] = st.checkbox("Enable", value=site.get("enabled", False), key=f"ws_enable_{i}")

            if st.button("➕ Add Website"):
                websites.append({
                    "name": "",
                    "url": "",
                    "max_pages": 50,
                    "use_javascript": False,
                    "enabled": False
                })
                st.rerun()

            if st.button("💾 Save Web Sources"):
                web_sources["websites"] = websites
                if save_web_sources(web_sources):
                    st.session_state.web_sources = web_sources
                    st.success("Web sources saved! They will be used in the next knowledge base rebuild.")
                else:
                    st.error("Failed to save web sources. Check logs.")

    # ----- Tasks Tab -----
    with tabs[4]:
        st.subheader("Active Tasks")
        active = get_active_tasks(st.session_state.user_id)
        if active:
            for task_id, variant, status, prog in active:
                cols = st.columns([2,2,2,2])
                cols[0].write(f"**{variant}**")
                cols[1].write(f"`{task_id[:8]}`")
                cols[2].write(status)
                if status == 'processing':
                    cols[3].progress(prog/100)
                else:
                    cols[3].write("—")
        else:
            st.info("⏳ No active tasks.")
        st.subheader("Task History")
        hist = get_task_history(user_id=st.session_state.user_id, limit=100)
        if hist.empty:
            st.info("📋 No task history yet.")
        else:
            st.dataframe(hist, width='stretch')

    # ----- Insights Tab -----
    with tabs[5]:
        st.subheader("AI‑Powered Insights")
        conn = sqlite3.connect(str(DB_PATH))
        model_err = pd.read_sql_query("""
            SELECT model_key, COUNT(*) as total, SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) as errors
            FROM user_activity
            WHERE timestamp >= datetime('now', '-7 days')
            GROUP BY model_key
        """, conn)
        user_growth = pd.read_sql_query("""
            SELECT DATE(created_at) as day, COUNT(*) as new_users
            FROM users
            WHERE created_at >= datetime('now', '-30 days')
            GROUP BY day
        """, conn)
        conn.close()

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📈 Model Health")
            if not model_err.empty:
                for _, row in model_err.iterrows():
                    err_rate = row['errors'] / row['total'] * 100 if row['total'] > 0 else 0
                    if err_rate > 10:
                        st.warning(f"⚠️ {row['model_key']} error rate: {err_rate:.1f}%")
                    else:
                        st.success(f"✅ {row['model_key']} error rate: {err_rate:.1f}%")
            else:
                st.info("No model data last 7 days.")

        with col2:
            st.markdown("#### 👥 User Growth")
            if not user_growth.empty:
                total_new = user_growth['new_users'].sum()
                st.metric("New Users (30d)", total_new)
                if len(user_growth) > 7:
                    last_week = user_growth.tail(7)['new_users'].mean()
                    st.info(f"Avg {last_week:.1f} new users/day last week")
            else:
                st.info("No new users.")

        st.markdown("#### 🧠 Knowledge Base")
        if VECTORDB_PATH.exists():
            mtime = datetime.fromtimestamp(VECTORDB_PATH.stat().st_mtime)
            kb_age = (datetime.now() - mtime).days
            if kb_age > 7:
                st.warning(f"KB is {kb_age} days old. Consider rebuilding.")
            else:
                st.success(f"KB is {kb_age} days old (fresh).")
        else:
            st.info("KB not built yet.")

        st.markdown("#### 📊 System Recommendations")
        sys_metrics = get_system_metrics()
        recs = []
        if sys_metrics["cpu"] > 80:
            recs.append("⚠️ CPU >80% – consider scaling up.")
        if sys_metrics["memory"] > 80:
            recs.append("⚠️ Memory >80% – increase RAM.")
        if sys_metrics["disk"] > 85:
            recs.append("⚠️ Disk >85% – clean up.")
        if not recs:
            recs.append("✅ All systems nominal.")
        for r in recs:
            st.write(r)

        st.markdown("#### 🔮 Forecast (Next 7 Days)")
        st.caption("Based on recent activity trends")
        act = get_user_activity_summary(14)
        if not act.empty and len(act) > 7:
            avg_queries = act['total_queries'].tail(7).mean()
            st.info(f"Expected daily queries: ~{int(avg_queries)}")
        else:
            st.info("Insufficient data for forecast.")

    # ----- System Tab -----
    with tabs[6]:
        st.subheader("System Control")
        if st.button("📥 Export Analytics"):
            import io
            out = io.BytesIO()
            with pd.ExcelWriter(out, engine='xlsxwriter') as w:
                get_model_performance(30).to_excel(w, sheet_name="Model")
                get_user_activity_summary(30).to_excel(w, sheet_name="Activity")
                get_system_metrics_history(24).to_excel(w, sheet_name="System")
                get_kb_history(30).to_excel(w, sheet_name="KB")
                get_task_history(user_id=None, limit=100).to_excel(w, sheet_name="Tasks")
            st.download_button("Download", data=out.getvalue(), file_name=f"yukti_export_{datetime.now():%Y%m%d}.xlsx")
        st.subheader("Log Viewer")
        log_viewer("updater.log", 50)

    # Auto-refresh logic
    if st.session_state.auto_refresh:
        time.sleep(10)
        st.rerun()

    st.stop()

else:
    # -------------------- Chat Interface --------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "media" in msg:
                for m in msg["media"]:
                    if m["type"] == "image":
                        st.image(m["url"], width=300)
                    elif m["type"] == "audio":
                        st.audio(m["url"])
                    elif m["type"] == "video":
                        st.video(m["url"])

    if prompt := st.chat_input("Type your message..."):
        start = time.time()
        st.session_state.processing = True
        uploaded = None
        if uploaded_file:
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                uploaded = tmp.name

        lang_info = detect_language(prompt)
        if lang_info['explicit_instruction']:
            st.session_state.conversation_language = lang_info['explicit_instruction']
        elif st.session_state.conversation_language is None:
            st.session_state.conversation_language = lang_info['language']

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            try:
                model_key = st.session_state.selected_model
                extra = {}
                if uploaded:
                    extra["image_url"] = uploaded

                if model_key in ["Yukti‑Flash", "Yukti‑Quantum"]:
                    if not check_kb_status():
                        ans = "KB not ready. Please wait for auto‑update to complete."
                        placeholder.markdown(ans)
                        result = {"type": "sync", "answer": ans}
                    else:
                        hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                        result = think(prompt, hist, model_key, language=st.session_state.conversation_language)
                else:
                    model = load_model(model_key)
                    if model_key == "Yukti‑Audio":
                        path = model.invoke(prompt, voice="female", language=st.session_state.conversation_language, **extra)
                        with open(path, "rb") as f:
                            audio = f.read()
                        st.audio(audio)
                        st.download_button("Download Audio", audio, "audio.wav")
                        ans = "Audio generated."
                        result = {"type": "sync", "answer": ans, "media": [{"type": "audio", "url": path}]}
                    elif model_key == "Yukti‑Image":
                        url = model.invoke(prompt, **extra)
                        st.image(url, width=300)
                        st.download_button("Download Image", requests.get(url).content, "image.png")
                        ans = "Image generated."
                        result = {"type": "sync", "answer": ans, "media": [{"type": "image", "url": url}]}
                    elif model_key == "Yukti‑Video":
                        tid = model.invoke(prompt, language=st.session_state.conversation_language, user_id=st.session_state.user_id, **extra)
                        st.session_state.tasks[tid] = {"variant": "Yukti‑Video", "status": "submitted", "progress": 0}
                        ans = f"Task {tid} started."
                        result = {"type": "async", "task_id": tid}
                    else:
                        hist = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-10:]]
                        result = think(prompt, hist, model_key, language=st.session_state.conversation_language)

                if result.get("type") == "sync":
                    ans = result.get("answer", "")
                    if result.get("monologue"):
                        with st.expander("Reasoning"):
                            st.markdown(result["monologue"])
                    placeholder.markdown(ans)

                    # ---- Display sources (if any) ----
                    if result.get("sources"):
                        with st.expander("📚 Sources"):
                            for i, doc in enumerate(result["sources"]):
                                source_url = doc.metadata.get("source", "Unknown source")
                                st.markdown(f"{i+1}. [{source_url}]({source_url})")

                    log_user_activity(st.session_state.user_id, model_key, len(prompt),
                                      int((time.time()-start)*1000), True)
                    st.session_state.messages.append({"role": "assistant", "content": ans, "media": result.get("media", [])})
                elif result.get("type") == "async":
                    st.info(f"Task {result['task_id']} submitted. Check sidebar.")
                    st.session_state.messages.append({"role": "assistant", "content": ans})
            except Exception as e:
                st.error(f"Error: {e}")
                logger.exception("Chat error")
                log_user_activity(st.session_state.user_id, model_key, len(prompt),
                                  int((time.time()-start)*1000), False, str(e))

        st.session_state.processing = False
        st.rerun()
