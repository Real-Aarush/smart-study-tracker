import plotly.express as px
import streamlit as st
import pandas as pd
from utils.model import train_model

st.set_page_config(page_title="Smart Study Analyzer", layout="wide")

st.title("📚 Smart Study Analyzer")
st.markdown("Analyze study patterns and predict student performance.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/student_data.csv")


df = load_data()

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
        color: white;
    }
    
    /* General text */
html, body, [class*="css"]  {
    color: #E0E0E0;
}

/* Slider thumb (the draggable circle) */
.stSlider > div > div > div > div > div {
    background-color: #00D4FF;
}

/* Title and headers */
h1, h2, h3, h4 {
    color: #00D4FF;
}

    </style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.header("📋 Dataset Summary")
    st.metric("Total Students", len(df))
    st.metric("Average Score", f"{df['final_score'].mean():.1f}")
    st.metric("Top Score", f"{df['final_score'].max():.1f}")
    st.metric("Lowest Score", f"{df['final_score'].min():.1f}")
    st.markdown("---")
    st.info("Adjust the sliders in the Predict tab to get your personalized score prediction.")



# Train model
model, mae, r2 = train_model()

st.subheader("🎯 Predict Student Score")

col1, col2 = st.columns(2)

with col1:
    study  = st.slider("Study Hours per Day",  1.0, 10.0, 5.0)
    sleep  = st.slider("Sleep Hours per Day",  4.0,  9.0, 7.0)

with col2:
    attend = st.slider("Attendance (%)",       50.0, 100.0, 75.0)
    prev   = st.slider("Previous Score",       40.0,  95.0, 65.0)

if st.button("Predict Score"):
    prediction = model.predict([[study, sleep, attend, prev]])[0]
    st.metric(label="Predicted Final Score", value=f"{prediction:.1f} / 100")



tab1, tab2, tab3 = st.tabs(["📊 Data & Stats", "📈 Visualizations", "🎯 Predict"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10))
    st.subheader("Basic Statistics")
    st.write(df.describe())

with tab2:
    st.subheader("🔥 Correlation Heatmap")

    corr = df.corr()
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        title="Feature Correlation Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Score Distribution")

    fig2 = px.histogram(
        df,
        x="final_score",
        nbins=20,
        color_discrete_sequence=["#636EFA"],
        title="Distribution of Final Scores"
    )
    fig2.update_layout(bargap=0.1)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📈 Study Hours vs Final Score")

    fig3 = px.scatter(
        df,
        x="study_hours",
        y="final_score",
        color="attendance",
        size="sleep_hours",
        hover_data=["prev_score"],
        title="Study Hours vs Final Score",
        color_continuous_scale="Viridis"
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🏆 Feature Importance")

    importance = pd.DataFrame({
        'Feature': ['Study Hours', 'Sleep Hours', 'Attendance', 'Previous Score'],
        'Coefficient': model.coef_
    })
    importance = importance.sort_values('Coefficient', ascending=True)

    fig4 = px.bar(
        importance,
        x='Coefficient',
        y='Feature',
        orientation='h',
        color='Coefficient',
        color_continuous_scale='Blues',
        title='Feature Importance (Model Coefficients)'
    )
    st.plotly_chart(fig4, use_container_width=True)

with tab3:
    st.subheader("📐 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Absolute Error", f"{mae:.2f} pts")
    col2.metric("R² Score", f"{r2:.3f}")

    st.subheader("💡 Personalized Insights")

    insights = []

    if study < 4:
        insights.append("⚠️ You're studying less than 4 hours. Aim for at least 6 hours daily.")
    elif study >= 7:
        insights.append("✅ Great study hours! Keep it consistent.")

    if sleep < 6:
        insights.append("⚠️ You're sleeping less than 6 hours. Poor sleep reduces retention.")
    elif sleep >= 7:
        insights.append("✅ Good sleep schedule. This boosts memory consolidation.")

    if attend < 70:
        insights.append("⚠️ Attendance below 70% is risky. Try to attend more classes.")
    elif attend >= 85:
        insights.append("✅ Excellent attendance. You're not missing key content.")

    if prev < 50:
        insights.append("⚠️ Previous score is low. Focus on fundamentals first.")
    elif prev >= 75:
        insights.append("✅ Strong previous performance. Build on this momentum.")


    if prediction >= 80:
        insights.append("🏆 You're on track for a distinction. Stay consistent.")
    elif prediction >= 60:
        insights.append("📈 decent score predicted. Small improvements can push you above 80.")
    else:
        insights.append("🚨 High risk of low score. Immediate improvement needed.")

    for insight in insights:
        st.write(insight)

    import plotly.graph_objects as go

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction,
        title={'text': "Predicted Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "red"},
                {'range': [50, 75], 'color': "orange"},
                {'range': [75, 100], 'color': "green"}
            ],
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)
