"""
Reusable UI components for Yukti AI.
Includes task rendering, error toasts, confirmation dialogs, and metric cards.
Designed to keep main.py clean and maintainable.
"""

import time
import logging
from typing import Dict, Any, Optional, Callable

import streamlit as st
import requests

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Task Rendering
# ----------------------------------------------------------------------
def render_task(task_id: str, task_info: Dict[str, Any]) -> None:
    """
    Render a single task in the sidebar with download options.
    Updates in real-time based on task status.
    """
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{task_info['variant']}**  \n`{task_id[:8]}`")
            if task_info['status'] == 'processing':
                st.progress(task_info['progress'] / 100, text=f"{task_info['progress']}%")
            elif task_info['status'] == 'completed':
                st.success("✅ Completed")
            elif task_info['status'] == 'failed':
                st.error(f"❌ Failed: {task_info['error']}")
            else:
                st.info("⏳ Queued...")
        with cols[1]:
            if task_info['status'] == 'processing':
                if st.button("⟳", key=f"refresh_{task_id}"):
                    # Refresh will be handled by caller
                    pass
            elif task_info['status'] == 'completed' and task_info.get('result_url'):
                url = task_info['result_url']
                if task_info['variant'] == 'Yukti‑Video':
                    st.markdown(f"[🎬 Watch Video]({url})")
                    _download_button(url, f"yukti_video_{task_id[:8]}.mp4", "Video")
                else:  # Image
                    st.image(url, width=200)
                    _download_button(url, f"yukti_image_{task_id[:8]}.png", "Image")

def _download_button(url: str, filename: str, media_type: str) -> None:
    """Helper to create a download button for media."""
    try:
        with st.spinner("Preparing download..."):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            st.download_button(
                f"📥 Download {media_type}",
                data=response.iter_content(chunk_size=8192),
                file_name=filename,
                key=f"dl_{hash(url)}"
            )
    except Exception as e:
        st.error(f"Download failed: {e}")
        logger.exception(f"Download error for {url}")

# ----------------------------------------------------------------------
# Error Display
# ----------------------------------------------------------------------
def show_error_toast(message: str, icon: str = "❌") -> None:
    """
    Display a non‑blocking error toast (uses st.toast in newer Streamlit).
    Falls back to st.error if toast not available.
    """
    try:
        st.toast(message, icon=icon)
    except AttributeError:
        # Fallback for older Streamlit versions
        st.error(message)

def show_success_toast(message: str, icon: str = "✅") -> None:
    """Display a success toast."""
    try:
        st.toast(message, icon=icon)
    except AttributeError:
        st.success(message)

# ----------------------------------------------------------------------
# Confirmation Dialog
# ----------------------------------------------------------------------
def confirm_dialog(
    title: str,
    message: str,
    confirm_text: str = "Confirm",
    cancel_text: str = "Cancel",
    key: Optional[str] = None
) -> bool:
    """
    Display a confirmation dialog. Returns True if user confirms.
    Uses a popover to avoid cluttering the UI.
    """
    dialog_key = key or f"confirm_{hash(title)}"
    with st.popover(title):
        st.warning(message)
        col1, col2 = st.columns(2)
        with col1:
            if st.button(confirm_text, key=f"{dialog_key}_confirm"):
                return True
        with col2:
            if st.button(cancel_text, key=f"{dialog_key}_cancel"):
                return False
    return False

# ----------------------------------------------------------------------
# Metric Card with Delta
# ----------------------------------------------------------------------
def metric_card(
    label: str,
    value: Any,
    delta: Optional[float] = None,
    delta_color: str = "normal",
    help_text: Optional[str] = None
) -> None:
    """
    Display a metric card with optional delta and help tooltip.
    """
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color, help=help_text)

# ----------------------------------------------------------------------
# Real‑Time Metrics Updater (for admin dashboard)
# ----------------------------------------------------------------------
def realtime_metrics_placeholder(placeholder, get_metrics_func: Callable, interval: float = 2.0):
    """
    Continuously update a placeholder with metrics from get_metrics_func.
    Should be called inside a while loop in the admin dashboard.
    """
    with placeholder.container():
        metrics = get_metrics_func()
        cols = st.columns(len(metrics))
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                st.metric(label=key, value=value)
    time.sleep(interval)

# ----------------------------------------------------------------------
# Log Viewer
# ----------------------------------------------------------------------
def log_viewer(log_path: str, max_lines: int = 50) -> None:
    """
    Display the last max_lines lines of a log file in a scrollable text area.
    """
    try:
        with open(log_path, 'r') as f:
            lines = f.readlines()[-max_lines:]
            log_text = "".join(lines)
        st.code(log_text, language="bash")
    except FileNotFoundError:
        st.info("Log file not found.")
    except Exception as e:
        st.error(f"Error reading log: {e}")

# ----------------------------------------------------------------------
# Export all functions
# ----------------------------------------------------------------------
__all__ = [
    "render_task",
    "show_error_toast",
    "show_success_toast",
    "confirm_dialog",
    "metric_card",
    "realtime_metrics_placeholder",
    "log_viewer",
]
