# Fake News Predictor

## **Inspiration**

When brainstorming a creative AI project to address a common issue, we were struck by the prevalence of misinformation in today’s world. From sensationalized news on platforms like TikTok to misleading articles circulating online, fake news is produced and shared at an alarming rate. Recognizing how easily people can fall for and spread such information, we were inspired to develop a simple yet powerful tool. Our goal was to empower users to evaluate the credibility of articles quickly and effectively, equipping them with the means to combat fake news and promote informed decision-making.

## **What it does**

Fake News Predictor is a simple yet powerful tool designed to help users analyze the credibility of news articles. Users can paste any article into the platform and instantly receive a classification of whether the news is **True** or **False**, alongside a confidence score represented as a percentage. This confidence score allows users to gauge how certain the AI is in its prediction, with higher percentages reflecting greater certainty. 

To provide further insights, Fake News Predictor also analyzes the article for bias using polarity and subjectivity scores:
- **Polarity** measures the sentiment of the text, ranging from -1 (negative) to 1 (positive), helping users understand the tone of the article.
- **Subjectivity** evaluates how opinion-based the article is, ranging from 0 (completely objective) to 1 (highly subjective), enabling users to distinguish between factual reporting and personal opinions.

## ✨ Key Features:

- 📝 Paste any news article into the input box
- 🔍 Instantly classify news as True or False
- 📊 Display a confidence score (%) showing prediction certainty
- 🗳️ Collect user feedback if the prediction seems incorrect
- 🗄️ Store feedback in a database to improve future performance
This combination helps users critically evaluate not just the authenticity, but also the tone and bias of the content they consume.
Together, these metrics empower users to critically assess the tone and bias of the content they’re consuming. If a user disagrees with the AI’s classification, they can submit feedback directly within the platform. This feedback is stored in a database, helping refine and improve the system over time.

## **How we built it**

To create Fake News Predictor, we began by sourcing and preprocessing a large dataset of fake and real news articles. Using **TF-IDF Vectorization**, we analyzed word frequencies and patterns, enabling our AI model to identify common traits in fake news, such as sensational language or repetitive phrases.

## 🧠 Machine Learning & NLP

We trained and evaluated three machine learning models:
- Logistic Regression
- Random Forest
- XGBoost

## 🖥️ Streamlit Web App

- Built an interactive and responsive UI using Streamlit
- Features:
    - News input text area
    - Real-time prediction display
    - Feedback submission system
    - Session state handling

## 📊 Text Analysis

- Implemented TextBlob for:
      - Sentiment (polarity)
      - Subjectivity (bias detection)

## 🗄️ Database

- Used SQLite to store:
    - User-entered news articles
    - Model predictions
    - User feedback
    - Timestamps

We selected the most accurate model, which achieved a **96% success rate**.

The backend was developed using **Flask**, which also processes bias analysis and feedback collection. For additional insights, we implemented **TextBlob** to provide polarity and subjectivity scores to detect tone and bias in articles.

User feedback is stored in a **SQLite** database, creating a system that can learn and improve over time. To ensure a user-friendly experience, we designed the interface in **Figma**, and developed it using **HTML and CSS**.

## **Challenges we ran into**

There were a lot of new frameworks we learned about during this project, allowing us to gain valuable experience. Although it was time-consuming, learning **scikit-learn** and training machine learning models (including logistic regression, random forest, and XGBoost) was exciting.

Initially, our AI model had only **50% accuracy**, and improving it was a tedious process. We had to research and implement different data processing techniques to train the AI more effectively, eventually achieving **96% accuracy**.

Additionally, none of us had experience with **SQLite** or any other database engine, so learning how to use it from scratch to store user feedback was a significant challenge.

## **Accomplishments that we're proud of**

- Successfully training a machine learning model to detect fake news with **96% accuracy**.
- Integrating multiple features—bias detection, sentiment analysis, and user feedback—into one cohesive system.
- Building a working **feedback loop** using SQLite, despite no prior database experience.

## **What we learned**

- Training machine learning models using **scikit-learn**.
- Text processing with **TF-IDF Vectorization**.
- Implementing **sentiment and bias analysis** with **TextBlob**.
- Storing and managing data using **SQLite**.
- Creating a clean, responsive user interface with **HTML/CSS**.

## **What’s Next for Fake News Predictor**

- **Image processing**: Allow users to upload screenshots of articles or posts for analysis.
- **PDF scanning**: Enable full-document uploads for misinformation detection.
- **Related articles**: Suggest credible articles on the same topic when fake news is detected, promoting informed decision-making.
