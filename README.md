# 🚀 AI-Powered Social Media Analytics

> **Turning social media text into meaningful, multi-dimensional and explainable insights.**

An end-to-end **AI-powered social media analytics platform** that analyzes text from social media posts across multiple dimensions — **sentiment, emotion, aspects, topics, and trends**.

The application combines **Deep Learning, NLP, Machine Learning, and Generative AI** into an interactive **Streamlit dashboard** called **SocialSense AI**.

---

## 🌐 Live Demo

🔗 **Live Application:**  
https://ai-powered-social-media-analytics.onrender.com

🔗 **GitHub Repository:**  
https://github.com/2k23cs2313652-rishi/AI-POWERED-SOCIAL-MEDIA-ANALYTICS

---

## ✨ Key Features

### 🧠 1. Sentiment Analysis

Classifies social media posts into:

- 😊 Positive
- 😞 Negative

Multiple deep learning architectures were experimented with:

- RNN
- LSTM
- BiLSTM

The final sentiment classifier uses **BiLSTM**.

**Final BiLSTM Performance:**

| Metric | Score |
|---|---:|
| Accuracy | 79.69% |
| Precision | 79.72% |
| Recall | 79.69% |
| F1-Score | 79.68% |

---

### 💭 2. Emotion Detection

Detects four emotions from social media text:

- 😠 Anger
- 😄 Joy
- 🌱 Optimism
- 😢 Sadness

The system uses a **BiLSTM-based deep learning model** with **pretrained GloVe embeddings**.

---

### 🔍 3. Aspect-Based Sentiment Analysis

Identifies important aspects mentioned in a post and determines the sentiment associated with them.

For example:

> "The camera is amazing but the battery life is terrible."

The system can identify:

| Aspect | Sentiment |
|---|---|
| Camera | Positive |
| Battery life | Negative |

The current implementation uses **spaCy noun-phrase extraction** to identify aspect candidates and the trained sentiment model to estimate their sentiment.

---

### 📊 4. Topic Analysis

The application uses:

- TF-IDF
- K-Means clustering

to group similar social media posts.

The top words from each cluster are used to interpret the resulting groups as topics.

Example topic categories include:

- 🎉 Celebrations & Special Occasions
- 🌙 Evening / Emotional
- 💼 Work Routine
- 🎵 Music & Entertainment
- 👥 #FollowFriday Culture

---

### 📈 5. Trend Analysis

The application analyzes how social media discussions change over time.

It provides visualizations for:

- Sentiment trends
- Emotion trends
- Topic trends
- Posting activity

The dataset contains posts from:

**April 6, 2009 → June 25, 2009**

---

### 🤖 6. Generative AI Insights

The platform uses **Groq-hosted LLM inference** to convert structured ML results into easy-to-understand insights.

Instead of replacing the trained ML models, the LLM acts as an **interpretation layer**.

For example, the system can combine:

```text
Sentiment → Positive
Emotion → Joy
Topic → Work Routine
Aspect → Customer Service
