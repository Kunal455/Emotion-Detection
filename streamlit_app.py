import streamlit as st
import predictor
import time

# Page config
st.set_page_config(
    page_title="Sentiment AI Pro",
    page_icon="🧠",
    layout="centered"
)


st.markdown("""
<style>
    .stTextInput > div > div > input {
        background-color: #2b2b2b;
        color: red;
        border-radius: 10px;
        padding: 10px;
    }
    .result-box {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: blue;
        text-align: center;
        font-size: 24px;
        margin-top: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Application Logic
st.title("🧠 Sentiment AI")
st.write("Discover the hidden emotions in your text with our 28-emotion AI.")

# Input
user_input = st.text_input("How are you feeling today?", placeholder="Type something... e.g., 'I feel hopeless' or 'Hurray!'")

if st.button("Analyze Emotion", type="primary"):
    if not user_input:
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing neurons..."):
            # Simulation of processing time for effect
            time.sleep(0.5) 
            
            # Predict
            try:
                result = predictor.predict_emotion(user_input)
                
                # result is like "😊 Joy"
                # Let's split it if possible
                if " " in result:
                    emoji, emotion = result.split(" ", 1)
                else:
                    emoji = "✨"
                    emotion = result
                
                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size: 60px;">{emoji}</div>
                    <div style="font-weight: bold; font-size: 30px;">{emotion}</div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.markdown("*Powered by Logistic Regression on GoEmotions dataset (211k samples)*")