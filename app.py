import tempfile
import streamlit as st
from pathlib import Path

from qa_engine import QAEngine

def main():
    st.set_page_config(
        page_title="Document Q&A",
        page_icon="📄",
        layout="centered"
    )
    st.title("📄 Document Q&A")
    st.caption(
        "Upload a PDF and ask questions about its contents."
    )

    if "engine" not in st.session_state:
        st.session_state.engine: QAEngine | None = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history: list[dict] = []
    if "pdf_name" not in st.session_state:
        st.session_state.pdf_name: str = ""

    with st.sidebar:
        st.header("⚙️ Configuration")
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            help="Paste your OpenRouter key (e.g. sk-or-v1-…)"
        )

        st.divider()
        st.header("📂 Upload PDF")
        uploaded = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded and api_key:
            if uploaded.name != st.session_state.pdf_name:
                with st.spinner("Processing PDF..."):
                    with tempfile.NamedTemporaryFile(
                            suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = Path(tmp.name)

                    try:
                        engine = QAEngine(api_key=api_key)
                        engine.ingest(tmp_path)
                        st.session_state.engine = engine
                        st.session_state.pdf_name = uploaded.name
                        st.session_state.chat_history = []
                        st.success(f"✅ Your file **{uploaded.name}** is ready!")
                    except Exception as exc:
                        st.error(f"Failed to process PDF:\n\n{exc}")
                    finally:
                        tmp_path.unlink(missing_ok=True)
        elif uploaded and not api_key:
            st.warning("Enter your OpenRouter API key first.")

        if st.session_state.pdf_name:
            st.info(f"📑 Active document: **{st.session_state.pdf_name}**")
            if st.button("🗑️ Clear & upload new"):
                st.session_state.engine = None
                st.session_state.pdf_name = ""
                st.session_state.chat_history = []
                st.rerun()

    if not st.session_state.engine:
        st.info("👈 Upload your PDF and enter your API key in the sidebar to get started.")
        st.stop()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📚 Source chunks"):
                    for i, src in enumerate(msg["sources"], 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} - page {page}**")
                        st.caption(src.page_content[:500])

    if question := st.chat_input("Ask a question about the document..."):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = st.session_state.engine.ask(question)
                    answer = result["answer"]
                    sources = result["source_documents"]
                except Exception as exc:
                    answer = f"❌ Error: {exc}"
                    sources = []

            st.markdown(answer)
            if sources:
                with st.expander("📚 Source chunks"):
                    for i, src in enumerate(sources, 1):
                        page = src.metadata.get("page", "?")
                        st.markdown(f"**Chunk {i} - page {page}**")
                        st.caption(src.page_content[:500])

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )

if __name__ == "__main__":
    main()