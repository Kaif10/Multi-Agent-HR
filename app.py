import streamlit as st
import tempfile
from kubernetes_ai_agents import verify_resume_pipeline  # ⬅️ Replace or paste your actual pipeline here

st.set_page_config(page_title="CV Verifier", page_icon="📄", layout="wide")
st.title("📄CV Verifier – Multi-Agent Resume Scanner")

uploaded_file = st.file_uploader("Upload your CV (PDF only)", type=["pdf"])

if uploaded_file:
    if st.button("🔍 Analyze CV"):
        with st.spinner("Processing... Please wait (~20s)"):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                result = verify_resume_pipeline(tmp_path)

                st.success("✅ CV Analysis Complete")

                # 🎓 Education Analysis
                st.subheader("🎓 Education Analysis")
                for line in result["education_analysis"]:
                    st.markdown(f"• {line}")

                # 🛠️ Skills & Seniority Summary
                st.subheader("🛠️ Skills & Seniority Summary")
                st.markdown(result["skills_and_seniority"])

                # ✍️ Writing Style Analysis
                st.subheader("✍️ Writing Style Feedback")
                st.markdown(result["writing_style_analysis"])

                # 🧠 Final Verdict
                verdict = result["decision"]
                icon = {"Proceed": "✅", "Review": "⚠️", "Reject": "❌"}.get(verdict["verdict"], "ℹ️")
                st.subheader("📌 Final Decision")
                st.markdown(f"{icon} **{verdict['verdict']}** (Confidence: `{verdict['confidence']*100:.0f}%`)")
                st.info(verdict["reason"])

            except Exception as e:
                st.error(f"❌ Error running the pipeline: {e}")
