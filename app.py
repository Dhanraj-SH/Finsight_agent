import streamlit as st
import requests
from fpdf import FPDF


#Page config
st.set_page_config(
    page_title="Financial Sentiment AI",
    page_icon="📈",
    layout="wide"
)

# TITLE
st.title("📈 Financial Sentiment AI")
st.markdown(
    "Analyze financial market sentiment using "
    "FinBERT + LangGraph + Groq"
)

# SIDEBAR
st.sidebar.title("Recent Companies")
if "history" not in st.session_state:
    st.session_state.history = []

for company in st.session_state.history[-5:]:
    st.sidebar.write(f"• {company}")


# INPUT
company = st.text_input(
    "Enter Company Name",
    placeholder="Tesla"
)


# ANALYZE BUTTON
if st.button("Analyze"):
    if company.strip() == "":
        st.warning("Please enter a company name.")
    else:
        if company not in st.session_state.history:
            st.session_state.history.append(company)
        with st.spinner(
            "Running AI financial analysis..."
        ):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    json={
                        "company": company
                    }
                )
                data = response.json()

                # METRICS
                st.subheader("Market Sentiment")
                score = data["overall_sentiment"]
                if score > 0:
                    sentiment_label = "Positive 📈"
                    color = "green"

                elif score < 0:
                    sentiment_label = "Negative 📉"
                    color = "red"

                else:
                    sentiment_label = "Neutral ⚪"
                    color = "gray"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Sentiment Score",score)
                with col2:
                    st.markdown(
                        f"""
                        <h3 style='color:{color};'>
                        {sentiment_label}
                        </h3>
                        """,
                        unsafe_allow_html=True
                    )

                # PROGRESS BAR
                progress_value = min(max((score + 1) / 2, 0),1)
                st.progress(progress_value)

                # SUMMARY
                st.subheader("News Summary")
                summary = data["summary"]
                c1, c2, c3 = st.columns(3)
                c1.metric("Positive",summary["positive"])
                c2.metric("Negative",summary["negative"])
                c3.metric("Neutral",summary["neutral"])

                # EXECUTIVE SUMMARY
                st.subheader("Executive Summary")
                st.write(data["executive_summary"])

                # REPORT
                st.subheader("Detailed Report")
                with st.expander("View Full Report"):
                    st.markdown(data["report"])

                # PDF DOWNLOAD
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial",size=12)

                report_text = data["report"].replace("•", "-")

                for line in report_text.split("\n"):
                    pdf.multi_cell(0,10,txt=line)

                pdf_file = f"{company}_report.pdf"
                pdf.output(pdf_file)

                with open(pdf_file,"rb") as file:
                    st.download_button(
                        label="Download Report PDF",
                        data=file,
                        file_name=pdf_file,
                        mime="application/pdf"
                    )

            except Exception as e:
                st.error(str(e))