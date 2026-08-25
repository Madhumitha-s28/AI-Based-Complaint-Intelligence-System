import streamlit as st
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="Complaint Intelligence Dashboard",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2a44;
        margin-bottom: 0.2rem;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.2rem;
    }

    .section-card {
        background: white;
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }

    .model-pill-blue {
        display: inline-block;
        background-color: #e8f0fe;
        color: #2563eb;
        font-weight: 700;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .model-pill-green {
        display: inline-block;
        background-color: #eaf8ee;
        color: #16a34a;
        font-weight: 700;
        padding: 0.55rem 1rem;
        border-radius: 999px;
        font-size: 1rem;
        margin-bottom: 1rem;
    }

    .result-title {
        font-size: 1.9rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 1rem;
    }

    .result-row {
        font-size: 1.05rem;
        margin-bottom: 1.1rem;
        color: #1f2937;
        line-height: 1.6;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .priority-box {
        background-color: #eaf7ef;
        border: 1px solid #d1f0dc;
        color: #15803d;
        padding: 1rem;
        border-radius: 12px;
        font-size: 1.2rem;
        margin-top: 0.6rem;
        margin-bottom: 1rem;
        font-weight: 600;
    }

    .best-model-box {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-top: 1rem;
    }

    .divider-center {
        width: 2px;
        background: linear-gradient(to bottom, transparent, #d1d5db, transparent);
        min-height: 100%;
        margin: 0 auto;
    }

    .vs-badge {
        width: 56px;
        height: 56px;
        border-radius: 50%;
        border: 1px solid #d1d5db;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: #1f2937;
        margin: 1rem auto;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.92rem;
    }

    .insight-item {
        margin-bottom: 0.8rem;
        font-size: 1rem;
        color: #374151;
        line-height: 1.5;
    }
</style>
""", unsafe_allow_html=True)


svm_model = pickle.load(open("svm_issue_model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

svm_accuracy = pickle.load(open("accuracy.pkl", "rb"))
nb_accuracy = pickle.load(open("nb_accuracy.pkl", "rb"))

analyzer = SentimentIntensityAnalyzer()


def predict(model, text):
    vec = tfidf.transform([text])
    return model.predict(vec)[0]

def get_sentiment(text):
    score = analyzer.polarity_scores(text)

    if score['compound'] >= 0.05:
        return "Positive", 1
    elif score['compound'] <= -0.05:
        return "Negative", 3
    else:
        return "Neutral", 2

def get_issue_severity(issue):
    high = [
        "Cont'd attempts collect debt not owed",
        "Incorrect information on credit report",
        "Loan not serviced properly"
    ]

    medium = [
        "Credit card or prepaid card",
        "Mortgage",
        "Loan"
    ]

    if issue in high:
        return 3
    elif issue in medium:
        return 2
    else:
        return 1

def calculate_csi(issue_severity, sentiment_score):
    return round((0.6 * issue_severity) + (0.4 * sentiment_score), 2)

def get_priority(csi):
    if csi >= 2.5:
        return "High"
    elif csi >= 1.8:
        return "Medium"
    else:
        return "Low"

def get_action(priority):
    if priority == "High":
        return "🚨 Immediate escalation required"
    elif priority == "Medium":
        return "⚠️ Assign to support team"
    else:
        return "✅ Log and monitor"

def analyze_with_model(model, text):
    issue = predict(model, text)
    sentiment, sentiment_score = get_sentiment(text)
    issue_severity = get_issue_severity(issue)
    csi = calculate_csi(issue_severity, sentiment_score)
    priority = get_priority(csi)
    action = get_action(priority)

    return {
        "issue": issue,
        "sentiment": sentiment,
        "csi": csi,
        "priority": priority,
        "action": action
    }





import streamlit as st
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import plotly.express as px


st.set_page_config(
    page_title="Complaint Intelligence Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
/* Fix title being cut */
.block-container {
    padding-top: 3rem !important;
}

/* Force full visibility of headings */
h1 {
    margin-top: 0 !important;
    padding-top: 0.5rem !important;
    overflow: visible !important;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>
    .block-container {
        padding-top: 0.6rem !important;
        padding-bottom: 0.6rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 1500px !important;
    }

    .main {
        background-color: #f7f9fc;
    }

    section[data-testid="stSidebar"] {
        width: 300px !important;
        min-width: 300px !important;
        max-width: 300px !important;
        background-color: #f7f9fc;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 800;
        color: #1e2b4a;
        margin-bottom: 0.1rem;
        line-height: 1.2;
        white-space: normal !important;
        overflow: visible !important;
        word-break: break-word;
    }

    .app-subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 0.8rem;
    }

    .section-card {
        background: white;
        padding: 0.9rem 1rem;
        border-radius: 16px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 10px rgba(0,0,0,0.04);
        margin-bottom: 0.9rem;
    }

    .model-pill-blue {
        display: inline-block;
        background-color: #e8f0fe;
        color: #2563eb;
        font-weight: 700;
        padding: 0.5rem 0.95rem;
        border-radius: 999px;
        font-size: 0.98rem;
        margin-bottom: 0.6rem;
    }

    .model-pill-green {
        display: inline-block;
        background-color: #eaf8ee;
        color: #16a34a;
        font-weight: 700;
        padding: 0.5rem 0.95rem;
        border-radius: 999px;
        font-size: 0.98rem;
        margin-bottom: 0.6rem;
    }

    .result-title {
        font-size: 1.25rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.7rem;
    }

    .result-row {
        font-size: 1rem;
        margin-bottom: 0.8rem;
        color: #1f2937;
        line-height: 1.5;
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .priority-box-low, .priority-box-medium, .priority-box-high {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0.8rem;
        font-weight: 700;
    }

    .priority-box-low {
        background-color: #eaf7ef;
        border: 1px solid #d1f0dc;
        color: #15803d;
    }

    .priority-box-medium {
        background-color: #fff7e6;
        border: 1px solid #fde7b0;
        color: #b45309;
    }

    .priority-box-high {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #b91c1c;
    }

    .divider-wrap {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 360px;
    }

    .divider-line {
        width: 2px;
        background: #d1d5db;
        flex-grow: 1;
        min-height: 120px;
    }

    .vs-badge {
        width: 52px;
        height: 52px;
        border-radius: 50%;
        border: 1px solid #d1d5db;
        background: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 800;
        font-size: 1.05rem;
        color: #1f2937;
        margin: 0.45rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }

    .insight-item {
        margin-bottom: 0.65rem;
        font-size: 0.98rem;
        color: #374151;
        line-height: 1.45;
    }

    .compact-title {
        font-size: 1.05rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        color: #1f2937;
    }

    div[data-testid="stTextArea"] textarea {
        min-height: 95px !important;
    }

    div[data-testid="stButton"] > button {
        height: 44px !important;
        white-space: nowrap !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        padding: 0 18px !important;
    }
</style>
""", unsafe_allow_html=True)

svm_model = pickle.load(open("svm_issue_model.pkl", "rb"))
nb_model = pickle.load(open("nb_model.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
svm_accuracy = pickle.load(open("accuracy.pkl", "rb"))
nb_accuracy = pickle.load(open("nb_accuracy.pkl", "rb"))

analyzer = SentimentIntensityAnalyzer()

def predict(model, text):
    vec = tfidf.transform([text])
    return model.predict(vec)[0]

def get_sentiment(text):
    score = analyzer.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive", 1
    elif score['compound'] <= -0.05:
        return "Negative", 3
    return "Neutral", 2

def get_issue_severity(issue):
    high = [
        "Cont'd attempts collect debt not owed",
        "Incorrect information on credit report",
        "Loan not serviced properly"
    ]
    medium = [
        "Credit card or prepaid card",
        "Mortgage",
        "Loan"
    ]
    if issue in high:
        return 3
    elif issue in medium:
        return 2
    return 1

def calculate_csi(issue_severity, sentiment_score):
    return round((0.6 * issue_severity) + (0.4 * sentiment_score), 2)

def get_priority(csi):
    if csi >= 2.5:
        return "High"
    elif csi >= 1.8:
        return "Medium"
    return "Low"

def get_action(priority):
    if priority == "High":
        return "🚨 Immediate escalation required"
    elif priority == "Medium":
        return "⚠️ Assign to support team"
    return "✅ Log and monitor"

def get_priority_class(priority):
    if priority == "High":
        return "priority-box-high"
    elif priority == "Medium":
        return "priority-box-medium"
    return "priority-box-low"

def analyze_with_model(model, text):
    issue = predict(model, text)
    sentiment, sentiment_score = get_sentiment(text)
    issue_severity = get_issue_severity(issue)
    csi = calculate_csi(issue_severity, sentiment_score)
    priority = get_priority(csi)
    action = get_action(priority)
    return {
        "issue": issue,
        "sentiment": sentiment,
        "csi": csi,
        "priority": priority,
        "action": action
    }


if "complaint_text" not in st.session_state:
    st.session_state.complaint_text = ""


with st.sidebar:
    st.markdown("## 🧠 Complaint Intelligence")
    st.markdown(
        "<div class='small-muted'>AI-powered complaint analysis & model comparison</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("### 📊 Model Performance")

    st.markdown("**SVM Accuracy**")
    st.progress(int(svm_accuracy * 100))
    st.write(f"**{svm_accuracy * 100:.2f}%**")
    st.caption("Test Accuracy")

    st.markdown("**Naive Bayes Accuracy**")
    st.progress(int(nb_accuracy * 100))
    st.write(f"**{nb_accuracy * 100:.2f}%**")
    st.caption("Test Accuracy")

    accuracy_df_sidebar = pd.DataFrame({
        "Model": ["SVM", "Naive Bayes"],
        "Accuracy": [svm_accuracy * 100, nb_accuracy * 100]
    })

    st.markdown("---")
    st.markdown("### 📈 Accuracy Comparison")
    chart_sidebar = px.bar(
        accuracy_df_sidebar,
        x="Model",
        y="Accuracy",
        text="Accuracy",
        color="Model",
        color_discrete_map={"SVM": "#3b82f6", "Naive Bayes": "#22c55e"}
    )
    chart_sidebar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    chart_sidebar.update_layout(
        height=250,
        showlegend=False,
        yaxis_title="Accuracy (%)",
        xaxis_title="",
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(chart_sidebar, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Model Insights")
    better_model = "SVM" if svm_accuracy > nb_accuracy else "Naive Bayes"
    st.markdown(f"<div class='insight-item'>✅ <b>{better_model}</b> performs better overall.</div>", unsafe_allow_html=True)
    st.markdown("<div class='insight-item'>✅ Naive Bayes is generally faster and works well for text classification.</div>", unsafe_allow_html=True)
    st.markdown("<div class='insight-item'>✅ Compare both models on the same complaint.</div>", unsafe_allow_html=True)


st.markdown("<div class='app-title'>Complaint Intelligence Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='app-subtitle'>AI-powered complaint analysis & model comparison</div>", unsafe_allow_html=True)


st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='compact-title'>Enter Complaint</div>", unsafe_allow_html=True)

st.text_area(
    "Complaint",
    key="complaint_text",
    placeholder="Type customer complaint here...",
    label_visibility="collapsed"
)
btn1, _, _ = st.columns([1.2, 1.2, 6])

with btn1:
    analyze_clicked = st.button("🔍 Analyze", use_container_width=True)


st.markdown("</div>", unsafe_allow_html=True)


if analyze_clicked:
    complaint_text = st.session_state.complaint_text.strip()

    if not complaint_text:
        st.warning("Please enter a complaint.")
    else:
        svm_result = analyze_with_model(svm_model, complaint_text)
        nb_result = analyze_with_model(nb_model, complaint_text)

        st.markdown("<div class='section-card'>", unsafe_allow_html=True)

        left, center, right = st.columns([5.2, 0.9, 5.2], vertical_alignment="top")

        with left:
            st.markdown("<div class='model-pill-blue'>🧠 SVM Model</div>", unsafe_allow_html=True)
            st.markdown("<div class='result-title'>🔎 Results</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-row'><b>📁 Issue Category:</b> {svm_result['issue']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-row'><b>🙂 Sentiment:</b> {svm_result['sentiment']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-row'><b>📊 CSI Score:</b> {svm_result['csi']}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='{get_priority_class(svm_result['priority'])}'>✅ Priority: {svm_result['priority']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<div class='result-row'><b>Recommended Action:</b> {svm_result['action']}</div>", unsafe_allow_html=True)

        with center:
            st.markdown("""
            <div class='divider-wrap'>
                <div class='divider-line'></div>
                <div class='vs-badge'>VS</div>
                <div class='divider-line'></div>
            </div>
            """, unsafe_allow_html=True)

        with right:
            st.markdown("<div class='model-pill-green'>🧠 Naive Bayes Model</div>", unsafe_allow_html=True)
            st.markdown("<div class='result-title'>🔎 Results</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-row'><b>📁 Issue Category:</b> {nb_result['issue']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-row'><b>🙂 Sentiment:</b> {nb_result['sentiment']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result-row'><b>📊 CSI Score:</b> {nb_result['csi']}</div>", unsafe_allow_html=True)
            st.markdown(
                f"<div class='{get_priority_class(nb_result['priority'])}'>✅ Priority: {nb_result['priority']}</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"<div class='result-row'><b>Recommended Action:</b> {nb_result['action']}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        bottom1, bottom2, bottom3 = st.columns([1.15, 1.15, 1.2], vertical_alignment="top")

        accuracy_df = pd.DataFrame({
            "Model": ["SVM", "Naive Bayes"],
            "Accuracy": [svm_accuracy * 100, nb_accuracy * 100]
        })

        with bottom1:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='compact-title'>📊 Accuracy Comparison</div>", unsafe_allow_html=True)

            main_chart = px.bar(
                accuracy_df,
                x="Model",
                y="Accuracy",
                text="Accuracy",
                color="Model",
                color_discrete_map={"SVM": "#3b82f6", "Naive Bayes": "#22c55e"}
            )
            main_chart.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            main_chart.update_layout(
                height=240,
                showlegend=False,
                yaxis_title="Accuracy (%)",
                xaxis_title="",
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(main_chart, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with bottom2:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='compact-title'>📋 Model Summary</div>", unsafe_allow_html=True)

            summary_df = pd.DataFrame({
                "Metric": ["Accuracy", "Predicted Issue", "Sentiment", "Priority"],
                "SVM": [
                    f"{svm_accuracy * 100:.2f}%",
                    svm_result["issue"],
                    svm_result["sentiment"],
                    svm_result["priority"]
                ],
                "Naive Bayes": [
                    f"{nb_accuracy * 100:.2f}%",
                    nb_result["issue"],
                    nb_result["sentiment"],
                    nb_result["priority"]
                ]
            })
            st.dataframe(summary_df, use_container_width=True, hide_index=True, height=240)
            st.markdown("</div>", unsafe_allow_html=True)

        with bottom3:
            st.markdown("<div class='section-card'>", unsafe_allow_html=True)
            st.markdown("<div class='compact-title'>🏆 Best Model</div>", unsafe_allow_html=True)

            diff = abs((svm_accuracy - nb_accuracy) * 100)

            if svm_accuracy > nb_accuracy:
                st.success(f"SVM is better overall by {diff:.2f}% accuracy.")
            elif nb_accuracy > svm_accuracy:
                st.success(f"Naive Bayes is better overall by {diff:.2f}% accuracy.")
            else:
                st.info("Both models have the same accuracy.")

            st.write(f"**SVM Accuracy:** {svm_accuracy * 100:.2f}%")
            st.write(f"**Naive Bayes Accuracy:** {nb_accuracy * 100:.2f}%")
            st.write("Use SVM for stronger overall performance. Use Naive Bayes for a lighter and faster text model.")
            st.markdown("</div>", unsafe_allow_html=True)

        st.caption("Models are evaluated on the test dataset. Results may vary with new complaints.")
