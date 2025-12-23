## Project Overview

This project is a simple Spam Email Classification system built using Support Vector Machines (SVM).
The goal of this project was to understand how machine learning models work on text data and how a trained model can be used in a real application. This is a learning-focused project and was built step by step while revising SVM concepts.

# What this project does

* Takes a text message as input

* Converts the text into numbers using TF-IDF

* Uses an SVM model to classify the message as:

     * Spam

     * Not Spam (Ham)

# Dataset

* SMS Spam Collection Dataset by UCI

* Contains labeled messages as spam or ham

* The dataset is imbalanced, which helped in understanding real-world data issues

# Machine Learning Workflow

1\. Loaded and cleaned the dataset

2\. Performed basic exploratory analysis (class counts, message length)

3\. Converted text data into numerical form using TF-IDF Vectorizer

4\. Trained a Linear Support Vector Classifier

5\. Evaluated the model using confusion matrix and classification report

6\. Saved the trained model and vectorizer using pickle

# Tech Stack

Python, Pandas, NumPy, scikit-learn, Streamlit




**Note**

This project is made for learning purposes only and should not be used as a real spam filtering system.