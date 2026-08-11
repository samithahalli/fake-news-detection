# Fake News Detection Using Machine Learning

A machine learning web application that classifies news articles as **Real News** or **Fake News** using Natural Language Processing (NLP).

## Project Overview

Fake news can spread quickly through online platforms and make it difficult for people to distinguish reliable information from misleading content.

This project uses machine learning and text processing techniques to analyze news content and predict whether an article is likely to be real or fake.

The trained model is integrated into a simple **Streamlit web application** where users can enter news content and receive a prediction with a confidence score.

## Features

- Fake and real news classification
- Text preprocessing and cleaning
- TF-IDF text vectorization
- Logistic Regression classification
- Prediction confidence score
- Interactive Streamlit web interface
- Model evaluation using accuracy, precision, recall and F1-score

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Streamlit
- Joblib
- Regular Expressions

## Machine Learning Workflow

```text
News Dataset
     ↓
Data Loading
     ↓
Data Cleaning
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Train/Test Split
     ↓
Logistic Regression
     ↓
Model Evaluation
     ↓
Save Model
     ↓
Streamlit Web Application
     ↓
Real / Fake Prediction