import streamlit as st
import pickle
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import string
import nltk
nltk.download('stopwords')

# Load the saved models and vectorizer
with open('models/logistic_regression_model.pkl', 'rb') as file:
    logistic_regression_model = pickle.load(file)
with open('models/lightgbm_model1.pkl', 'rb') as file:
    lightgbm_model = pickle.load(file)
with open('models/random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)
with open('models/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

additional_stopwords = [...]  # Your additional stopwords list

def remove_punctuation(text):
    """Remove punctuation from the text."""
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_additional_stopwords(text):
    """Remove additional stopwords from the text."""
    words = text.split()
    filtered_words = [word for word in words if word not in additional_stopwords]
    return ' '.join(filtered_words)

def generate_wordcloud(text):
    """Generate a word cloud from the text."""
    stop_words = set(stopwords.words('english'))
    wordcloud = WordCloud(width=800, height=800, background_color='white',
                          stopwords=stop_words, min_font_size=10).generate(text)
    return wordcloud

def main():
    st.title("Sentiment Analysis Web App")

    # Activities selection
    activities = ["Sentiment Analysis", "Generate Word Cloud", "Text Analysis of URL"]
    choice = st.sidebar.selectbox("Choose Activity", activities)

    if choice == "Sentiment Analysis":
        st.subheader("Sentiment Analysis")

        # Model selection
        models = ["Logistic Regression", "LightGBM", "Random Forest"]
        model_choice = st.selectbox("Choose Model", models)

        raw_text = st.text_area("Enter Text Here", "Type Here")

        if st.button("Analyze Text"):
            if raw_text != "":
                raw_text_clean = raw_text.lower()
                raw_text_clean = remove_punctuation(raw_text_clean)
                raw_text_clean = remove_additional_stopwords(raw_text_clean)

                # Vectorize the cleaned text
                X_test = vectorizer.transform([raw_text_clean])

                # Model prediction
                if model_choice == "Logistic Regression":
                    model = logistic_regression_model
                elif model_choice == "LightGBM":
                    model = lightgbm_model
                else:
                    model = random_forest_model

                prediction = model.predict(X_test)[0]

                # Display prediction
                if prediction == 1:
                    st.success("Positive Sentiment 😀")
                else:
                    st.error("Negative Sentiment 😞")

                # Display prediction probability if available
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_test)[0]
                    st.write(f"Confidence: Positive: {proba[1]:.2f}, Negative: {proba[0]:.2f}")
            else:
                st.warning("Please enter text for analysis.")

    elif choice == "Generate Word Cloud":
        st.subheader("Generate Word Cloud")
        raw_text = st.text_area("Enter Text Here", "Type Here")

        # Generate Word Cloud button is pressed
        if st.button("Generate Word Cloud"):
            if raw_text != "":
                # Display word cloud for the original text
                st.subheader("Original Text Word Cloud")
                original_wordcloud = generate_wordcloud(raw_text)
                plt.figure(figsize=(8, 8), facecolor=None)
                plt.imshow(original_wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(plt)

                # Clean the text
                cleaned_text = raw_text.lower()
                cleaned_text = remove_punctuation(cleaned_text)
                cleaned_text = remove_additional_stopwords(cleaned_text)

                # Display word cloud for the cleaned text
                st.subheader("Cleaned Text Word Cloud")
                cleaned_wordcloud = generate_wordcloud(cleaned_text)
                plt.figure(figsize=(8, 8), facecolor=None)
                plt.imshow(cleaned_wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(plt)
            else:
                st.warning("Please enter text to generate word cloud.")


if __name__ == '__main__':
    main()
