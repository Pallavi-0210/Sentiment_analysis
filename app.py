import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from preprocess import preprocess_and_stem

# Load model and vectorizer
model = joblib.load("model/best_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")

# Label mapping
label_map = {0: "Negative", 2: "Neutral", 4: "Positive"}
emoji_map = {"Negative": "😞", "Neutral": "😐", "Positive": "😊"}

st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="wide")
st.title("💬 Tweet Sentiment Analyzer")

# === Function to highlight confidence
def highlight_confidence(val):
    try:
        score = float(val.strip('%'))
        if score < 70:
            return 'background-color: red; color: white'
        elif score < 90:
            return 'background-color: orange; color: black'
        else:
            return 'background-color: green; color: white'
    except:
        return ''

# === Function to generate wordclouds


def show_wordclouds(data, sentiment_col, text_col):
    st.subheader("☁️ Word Clouds per Sentiment")
    sentiments = data[sentiment_col].unique()
    cols = st.columns(len(sentiments))

    for idx, sentiment in enumerate(sentiments):
        text = ' '.join(data[data[sentiment_col] == sentiment][text_col])
        if not text.strip():
            continue  # Skip empty groups

        wordcloud = WordCloud(width=500, height=300, background_color='white').generate(text)

        with cols[idx]:
            st.markdown(f"**{sentiment}** {emoji_map.get(sentiment, '')}")
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)  # Prevent matplotlib overlap


# === 1. Single Tweet Analysis ===
st.header("🔍 Analyze Single Tweet")
user_input = st.text_area("Enter a tweet:")
if st.button("Predict Sentiment"):
    if user_input.strip():
        processed = preprocess_and_stem(user_input)
        vector = vectorizer.transform([processed])
        probs = model.predict_proba(vector)[0]
        prediction = model.predict(vector)[0]
        sentiment = label_map.get(prediction, "Unknown")
        confidence = probs[model.classes_.tolist().index(prediction)]
        st.success(f"Predicted Sentiment: **{sentiment}** {emoji_map.get(sentiment, '')} (Confidence: {confidence:.2%})")

        st.subheader("📊 Confidence Scores")
        for label, prob in zip(model.classes_, probs):
            st.write(f"{label_map[label]}: {prob:.2%}")
    else:
        st.warning("Please enter a tweet.")

st.divider()

# === 2. TXT File Upload ===
st.header("📂 Upload .txt File of Tweets")
uploaded_txt = st.file_uploader("Upload a .txt file (one tweet per line):", type="txt")

if uploaded_txt:
    tweets = uploaded_txt.read().decode("utf-8").splitlines()
    if tweets:
        st.write(f"🧾 Total Tweets: {len(tweets)}")
        processed = [preprocess_and_stem(tw) for tw in tweets]
        vectors = vectorizer.transform(processed)
        predictions = model.predict(vectors)
        probs = model.predict_proba(vectors)
        sentiments = [label_map.get(p, "Unknown") for p in predictions]
        confidences = [f"{probs[i][model.classes_.tolist().index(predictions[i])] * 100:.2f}%" for i in range(len(predictions))]

        df = pd.DataFrame({
            "tweet": tweets,
            "Processed": processed,
            "Predicted Sentiment": sentiments,
            "Confidence": confidences
        })

        # Filter and toggle
        df['Confidence (%)'] = df['Confidence'].str.rstrip('%').astype(float)
        threshold = st.slider("Minimum Confidence (%) to Display", 0, 100, 70)
        view_all = st.toggle("👁️ Show All Predictions (Ignore Confidence Filter)", value=False)
        display_df = df if view_all else df[df['Confidence (%)'] >= threshold]

        st.dataframe(display_df[['tweet', 'Predicted Sentiment', 'Confidence']].style.applymap(highlight_confidence, subset=['Confidence']))

        # Pie chart
        st.subheader("📊 Sentiment Distribution")
        fig = px.pie(display_df, names='Predicted Sentiment', title='Sentiment Distribution by Confidence Threshold', color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)

        show_wordclouds(display_df, 'Predicted Sentiment', 'Processed')

        # Download
        csv = display_df[['tweet', 'Predicted Sentiment', 'Confidence']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Results as CSV", data=csv, file_name='txt_sentiments.csv', mime='text/csv')
    else:
        st.warning("No tweets found in the file.")

st.divider()

# === 3. CSV File Upload ===
st.header("📁 Upload .csv File of Tweets")
uploaded_csv = st.file_uploader("Upload a .csv file with a 'tweet' column:", type="csv")

if uploaded_csv:
    df = pd.read_csv(uploaded_csv)
    if 'tweet' in df.columns:
        st.write(f"✅ Loaded {len(df)} tweets.")
        df['Processed'] = df['tweet'].astype(str).apply(preprocess_and_stem)
        vectors = vectorizer.transform(df['Processed'])
        predictions = model.predict(vectors)
        probs = model.predict_proba(vectors)
        sentiments = [label_map.get(p, "Unknown") for p in predictions]
        confidences = [f"{probs[i][model.classes_.tolist().index(predictions[i])] * 100:.2f}%" for i in range(len(predictions))]

        df['Predicted Sentiment'] = sentiments
        df['Confidence'] = confidences
        df['Confidence (%)'] = df['Confidence'].str.rstrip('%').astype(float)

        threshold = st.slider("Minimum Confidence (%) to Display", 0, 100, 70, key="csv_slider")
        view_all = st.toggle("👁️ Show All Predictions (Ignore Confidence Filter)", value=False, key="csv_toggle")
        display_df = df if view_all else df[df['Confidence (%)'] >= threshold]

        st.dataframe(display_df[['tweet', 'Predicted Sentiment', 'Confidence']].style.applymap(highlight_confidence, subset=['Confidence']))

        st.subheader("📊 Sentiment Distribution")
        fig = px.pie(display_df, names='Predicted Sentiment', title='Sentiment Distribution by Confidence Threshold', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)

        show_wordclouds(display_df, 'Predicted Sentiment', 'Processed')

        csv = display_df[['tweet', 'Predicted Sentiment', 'Confidence']].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Filtered Results as CSV", data=csv, file_name='csv_sentiments.csv', mime='text/csv')
    else:
        st.error("CSV must contain a 'tweet' column.")
