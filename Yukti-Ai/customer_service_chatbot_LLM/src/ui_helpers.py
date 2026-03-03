import streamlit as st
import requests
import logging

logger = logging.getLogger(__name__)

def render_task(task_id: str, task_info: dict):
    """Render a single task in the sidebar with download options."""
    with st.container():
        cols = st.columns([3, 1])
        with cols[0]:
            st.markdown(f"**{task_info['variant']}**  \n`{task_id[:8]}`")
            if task_info['status'] == 'processing':
                st.progress(task_info['progress']/100, text=f"{task_info['progress']}%")
            elif task_info['status'] == 'completed':
                st.success("✅ Completed")
            elif task_info['status'] == 'failed':
                st.error(f"❌ Failed: {task_info['error']}")
            else:
                st.info("⏳ Queued...")
        with cols[1]:
            if task_info['status'] == 'processing':
                if st.button("⟳", key=f"refresh_{task_id}"):
                    # Refresh will be handled by the caller
                    pass
            elif task_info['status'] == 'completed' and task_info.get('result_url'):
                url = task_info['result_url']
                if task_info['variant'] == 'Yukti‑Video':
                    st.markdown(f"[🎬 Watch Video]({url})")
                    # Download with streaming to avoid memory issues
                    try:
                        with st.spinner("Downloading..."):
                            response = requests.get(url, stream=True)
                            response.raise_for_status()
                            st.download_button(
                                "📥 Download Video",
                                data=response.iter_content(chunk_size=8192),
                                file_name=f"yukti_video_{task_id[:8]}.mp4"
                            )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                        logger.exception("Video download error")
                else:  # Image
                    st.image(url, use_container_width=False, width=300)
                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()
                        st.download_button(
                            "📥 Download Image",
                            data=response.iter_content(chunk_size=8192),
                            file_name=f"yukti_image_{task_id[:8]}.png"
                        )
                    except Exception as e:
                        st.error(f"Download failed: {e}")
                        logger.exception("Image download error")
