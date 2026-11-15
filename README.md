# Phishing-Email-Detection-and-Security-Analysis
An ML-based Phishing Email Detection system using Python, TF-IDF, and supervised learning algorithms (Naive Bayes, Logistic Regression, and Random Forest) to classify emails as Legitimate or Spam, with a PhishGuard dashboard for security analysis.
## About the Project

Phishing Email Detection and Security Analysis is a Machine Learning-based cybersecurity project developed as part of the MCA Minor Project (SKE309).

The project aims to identify suspicious and spam emails by analyzing email content using supervised machine learning techniques. It provides an automated approach to classifying emails as *Legitimate or Spam*, helping users recognize potentially harmful email messages.

The system uses Natural Language Processing (NLP) techniques and TF-IDF (Term Frequency–Inverse Document Frequency) for text feature extraction. Three machine learning algorithms are implemented and compared to identify the most effective classification model.

## Objectives

- To detect phishing-related and spam email messages using Machine Learning.
- To preprocess and clean email datasets for accurate classification.
- To extract meaningful text features using TF-IDF.
- To train and compare multiple classification algorithms.
- To evaluate model performance using accuracy, precision, recall and F1-score.
- To provide a simple web-based interface for email analysis.

## Technologies Used

- *Programming Language:* Python
- *Machine Learning:* Scikit-learn
- *Data Processing:* Pandas, NumPy, SciPy
- *Feature Extraction:* TF-IDF
- *Algorithms:* Multinomial Naive Bayes, Logistic Regression, Random Forest
- *Web Interface:* Flask, HTML, CSS
- *Model Persistence:* Joblib

## Machine Learning Algorithms

The project implements three supervised learning algorithms:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Random Forest Classifier

Logistic Regression achieved the best recorded performance on the held-out test dataset.

## Model Performance

| Algorithm | Accuracy |
|---|---:|
| Naive Bayes | 97.17% |
| Logistic Regression | 99.17% |
| Random Forest | 98.50% |

Logistic Regression achieved 99.17% accuracy, with 100% spam precision, 95% spam recall and a 97.44% spam F1-score on the test dataset.

## Project Features

- Email text preprocessing and cleaning.
- TF-IDF-based feature extraction.
- Automated email classification.
- Comparison of three machine learning models.
- Confusion matrix and performance evaluation.
- PhishGuard dashboard for email security analysis.

## Dataset

The project uses the public SpamAssassin email corpus, containing legitimate and spam email examples for supervised machine learning.

## Future Scope

- Integration with real-time email services.
- Improved detection using larger and more recent datasets.
- URL and sender-domain analysis.
- Explainable AI for better prediction transparency.
- Enhanced web dashboard and user experience.

## Disclaimer

This project is developed for academic and educational purposes. It serves as a preliminary email screening tool and does not guarantee the detection of every phishing attempt. Users should independently verify suspicious emails and avoid sharing sensitive information.

## Academic Project

*Course:* MCA Minor Project (SKE309)  
*Project Title:* Phishing Email Detection and Security Analysis  
*Institution:* Amity University Kolkata  
*Academic Session:* 2026–2027
