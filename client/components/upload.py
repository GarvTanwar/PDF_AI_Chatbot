import streamlit as st
from utils.api import upload_pdfs_api

def render_uploader():
    st.sidebar.header("📤 Upload & Index Documents")

    # Step 1: Let Streamlit manage selection
    uploaded_files = st.sidebar.file_uploader(
        "Select Document(s)",
        type=["pdf", "txt", "doc", "docx", "xls", "xlsx", "ppt", "pptx"],
        accept_multiple_files=True,
        key="uploader"
    )

    # Step 2: Show what’s currently selected
    if uploaded_files:
        st.sidebar.write("**Files selected:**")
        for f in uploaded_files:
            st.sidebar.write(f"- {f.name}")

    # Step 3: Upload button only when there’s something selected
    if st.sidebar.button("Upload & Index Now", disabled=not uploaded_files):
        with st.spinner("Indexing… please wait"):
            try:
                resp = upload_pdfs_api(uploaded_files)
                if resp.status_code == 200:
                    st.sidebar.success("✅ Done! Vector store updated.")
                    st.session_state.pop("uploader", None)  # safer than del
                else:
                    st.sidebar.error(f"❌ Upload failed: {resp.text}")
            except Exception as e:
                st.sidebar.error(f"❌ Upload error: {e}")
