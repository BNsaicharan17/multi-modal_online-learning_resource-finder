"""
Train an optimized text classification model for learning topic prediction.
Uses TF-IDF + LinearSVC with 120+ augmented training samples across 8 categories.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import joblib
import os

# ---------------------------------------------------------------------------
# 1. Expanded training data – 15+ examples per category, synonym-augmented
# ---------------------------------------------------------------------------
data = {
    "text": [
        # ── Machine Learning ──
        "learn machine learning",
        "ml algorithms supervised unsupervised",
        "machine learning tutorial beginner",
        "regression classification clustering",
        "random forest decision tree",
        "gradient boosting xgboost",
        "feature engineering selection",
        "train test split cross validation",
        "hyperparameter tuning grid search",
        "scikit-learn sklearn tutorial",
        "machine learning model evaluation metrics",
        "overfitting underfitting bias variance",
        "ensemble methods bagging boosting",
        "logistic regression linear regression",
        "support vector machine svm kernel",

        # ── Deep Learning ──
        "deep learning neural networks",
        "cnn rnn models",
        "convolutional neural network image",
        "recurrent neural network sequence",
        "lstm gru time series",
        "transformer attention mechanism",
        "backpropagation gradient descent",
        "tensorflow keras pytorch",
        "deep learning gpu training",
        "generative adversarial network gan",
        "autoencoders variational",
        "batch normalization dropout",
        "neural network architecture layers",
        "transfer learning fine tuning pretrained",
        "deep reinforcement learning",

        # ── Python ──
        "python programming",
        "python basics tutorial",
        "python data types variables",
        "python functions loops control flow",
        "python object oriented programming class",
        "python list dictionary comprehension",
        "python file handling io",
        "python decorators generators",
        "python virtual environment pip install",
        "python exception handling try except",
        "python modules packages import",
        "python string manipulation formatting",
        "python lambda map filter reduce",
        "learn python from scratch beginner",
        "python scripting automation",

        # ── NLP ──
        "natural language processing",
        "text mining nlp",
        "sentiment analysis opinion mining",
        "tokenization stemming lemmatization",
        "named entity recognition ner",
        "text classification spam detection",
        "word embeddings word2vec glove",
        "language model bert gpt",
        "part of speech tagging pos",
        "text preprocessing stopwords removal",
        "topic modeling lda",
        "machine translation seq2seq",
        "chatbot conversational ai",
        "information retrieval search engine",
        "speech recognition text to speech",

        # ── Data Science ──
        "data science analytics",
        "data analysis pandas numpy",
        "data visualization matplotlib seaborn",
        "exploratory data analysis eda",
        "statistics probability distribution",
        "hypothesis testing p value",
        "data cleaning wrangling preprocessing",
        "data science project workflow",
        "jupyter notebook data exploration",
        "big data hadoop spark",
        "data pipeline etl",
        "business intelligence dashboard",
        "a b testing experimentation",
        "time series analysis forecasting",
        "data science career roadmap",

        # ── Web Development ──
        "web development html css javascript",
        "react angular vue frontend framework",
        "node.js express backend server",
        "rest api graphql web service",
        "responsive web design mobile first",
        "full stack web developer",
        "django flask web application python",
        "database sql nosql mongodb",
        "web hosting deployment devops",
        "html css layout flexbox grid",
        "javascript dom manipulation event",
        "authentication authorization jwt oauth",
        "web performance optimization caching",
        "progressive web app pwa",
        "build a website from scratch",

        # ── Computer Vision ──
        "computer vision image processing",
        "object detection yolo ssd",
        "image segmentation semantic instance",
        "opencv image manipulation",
        "face recognition detection",
        "optical character recognition ocr",
        "image classification resnet vgg",
        "video analysis tracking",
        "pose estimation body detection",
        "medical image analysis radiology",
        "image augmentation preprocessing",
        "feature extraction sift surf",
        "panorama stitching 3d reconstruction",
        "autonomous driving perception",
        "real time object detection camera",

        # ── Cloud Computing ──
        "cloud computing aws azure gcp",
        "deploy application cloud server",
        "docker container kubernetes orchestration",
        "serverless lambda functions",
        "cloud storage s3 blob",
        "ci cd pipeline github actions",
        "infrastructure as code terraform",
        "microservices architecture",
        "cloud security iam roles",
        "load balancer auto scaling",
        "aws ec2 instance management",
        "azure devops cloud deployment",
        "google cloud platform gcp services",
        "cloud migration strategy",
        "monitoring logging cloud watch",
    ],
    "label": [
        # Machine Learning (15)
        "Machine Learning", "Machine Learning", "Machine Learning",
        "Machine Learning", "Machine Learning", "Machine Learning",
        "Machine Learning", "Machine Learning", "Machine Learning",
        "Machine Learning", "Machine Learning", "Machine Learning",
        "Machine Learning", "Machine Learning", "Machine Learning",

        # Deep Learning (15)
        "Deep Learning", "Deep Learning", "Deep Learning",
        "Deep Learning", "Deep Learning", "Deep Learning",
        "Deep Learning", "Deep Learning", "Deep Learning",
        "Deep Learning", "Deep Learning", "Deep Learning",
        "Deep Learning", "Deep Learning", "Deep Learning",

        # Python (15)
        "Python", "Python", "Python",
        "Python", "Python", "Python",
        "Python", "Python", "Python",
        "Python", "Python", "Python",
        "Python", "Python", "Python",

        # NLP (15)
        "NLP", "NLP", "NLP",
        "NLP", "NLP", "NLP",
        "NLP", "NLP", "NLP",
        "NLP", "NLP", "NLP",
        "NLP", "NLP", "NLP",

        # Data Science (15)
        "Data Science", "Data Science", "Data Science",
        "Data Science", "Data Science", "Data Science",
        "Data Science", "Data Science", "Data Science",
        "Data Science", "Data Science", "Data Science",
        "Data Science", "Data Science", "Data Science",

        # Web Development (15)
        "Web Development", "Web Development", "Web Development",
        "Web Development", "Web Development", "Web Development",
        "Web Development", "Web Development", "Web Development",
        "Web Development", "Web Development", "Web Development",
        "Web Development", "Web Development", "Web Development",

        # Computer Vision (15)
        "Computer Vision", "Computer Vision", "Computer Vision",
        "Computer Vision", "Computer Vision", "Computer Vision",
        "Computer Vision", "Computer Vision", "Computer Vision",
        "Computer Vision", "Computer Vision", "Computer Vision",
        "Computer Vision", "Computer Vision", "Computer Vision",

        # Cloud Computing (15)
        "Cloud Computing", "Cloud Computing", "Cloud Computing",
        "Cloud Computing", "Cloud Computing", "Cloud Computing",
        "Cloud Computing", "Cloud Computing", "Cloud Computing",
        "Cloud Computing", "Cloud Computing", "Cloud Computing",
        "Cloud Computing", "Cloud Computing", "Cloud Computing",
    ],
}

# ---------------------------------------------------------------------------
# 2. Build TF-IDF + LinearSVC pipeline
# ---------------------------------------------------------------------------
df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),      # unigrams + bigrams
    max_features=5000,
    sublinear_tf=True,       # apply log normalization
    strip_accents="unicode",
    lowercase=True,
)

X = vectorizer.fit_transform(df["text"])
y = df["label"]

classifier = LinearSVC(
    C=1.0,
    max_iter=10000,
    class_weight="balanced",  # handle any class imbalance
)
classifier.fit(X, y)

# ---------------------------------------------------------------------------
# 3. Cross-validation report
# ---------------------------------------------------------------------------
scores = cross_val_score(classifier, X, y, cv=3, scoring="accuracy")
print(f"Cross-validation accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")

# ---------------------------------------------------------------------------
# 4. Save artefacts
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(classifier, "models/text_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
# Also save the training texts + labels for similarity fallback
joblib.dump(df, "models/training_data.pkl")

print("✅ Model, vectorizer, and training data saved to models/")