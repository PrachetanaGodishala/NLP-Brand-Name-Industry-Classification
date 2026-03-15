# NLP-Based Brand Name Industry Classification System

## Project Overview

This project presents an **Artificial Intelligence and Natural Language Processing (NLP) based system** designed to automatically classify brand names into their respective **industry groups**.

The system analyzes brand names using **DistilRoBERT word embeddings** and applies **machine learning algorithms** to predict the most appropriate **industry category based on the Global Industry Classification Standard (GICS)**.

The project demonstrates how **NLP and machine learning can automate brand classification**, which can assist businesses in market analysis, brand research, and data organization.

---

# Objectives

The main objectives of this project are:

* To apply **Natural Language Processing techniques** to analyze brand names.
* To generate **word embeddings using DistilRoBERT models**.
* To classify brand names into **industry categories using machine learning models**.
* To evaluate model performance using **accuracy metrics, ROC curves, and confusion matrices**.
* To build a **web interface using Flask** that allows users to interact with the classification system.

---

# Technologies Used

## Programming Language

Python

## Machine Learning Libraries

* Scikit-learn
* Transformers
* PyTorch

## Data Processing Libraries

* Pandas
* NumPy

## Visualization

* Matplotlib
* Seaborn

## NLP Tools

* NLTK
* DistilRoBERT embeddings

## Web Framework

Flask

---

# System Architecture

Brand Name Input
↓
Text Preprocessing (Tokenization, Cleaning)
↓
DistilRoBERT Word Embedding Generation
↓
Machine Learning Classification
(Logistic Regression / Random Forest / SVC)
↓
Prediction of Industry Category
↓
Performance Evaluation and Visualization

---

# Dataset

The dataset used in this project contains **brand names mapped to industry groups based on the Global Industry Classification Standard (GICS)**.

Example industry categories include:

* Automobiles & Components
* Banks
* Capital Goods
* Financial Services
* Consumer Services
* Technology Hardware & Equipment
* Pharmaceuticals, Biotechnology & Life Sciences
* Telecommunications Services
* Utilities
* Transportation

---

# Machine Learning Models Implemented

The following machine learning models were trained and evaluated:

1. Logistic Regression
2. Random Forest Classifier
3. Support Vector Classifier (SVC)

These models were trained using **DistilRoBERT word embeddings** extracted from the brand name dataset.

---

# Performance Evaluation

Model performance was evaluated using the following metrics:

* Accuracy Score
* Confusion Matrix
* ROC Curve
* Class-wise Performance Graphs

The results are visualized using graphs stored in the **results folder**.

---

# Web Application

A **Flask-based web application** was developed to allow users to interact with the system.

Users can:

* Login or register
* Upload brand name datasets
* Run the classification model
* View prediction results
* Analyze performance graphs

The web interface is built using HTML templates and Flask backend integration.

---

# Project Structure

BrandNameClassification

Dataset/
Contains industry classification datasets

model/
Stores trained machine learning models and embeddings

results/
Contains performance graphs and evaluation results

templates/
HTML files used for the web interface

static/
Static resources used by the web application

uploads/
Files uploaded by users

app.py
Main Flask application file

graphs.py
Generates performance graphs

metrics_calculator.py
Calculates evaluation metrics

Main_Jupyter.ipynb
Model training and experimentation notebook

requirements.txt
Project dependencies

---

# Installation and Setup

Install required Python libraries:

pip install -r requirements.txt

Run the Flask application:

python app.py

Open the application in the browser to interact with the system.

---

# Results

The project successfully classifies brand names into their corresponding industry categories with strong model performance. Visualizations such as **ROC curves, confusion matrices, and class-wise performance graphs** help analyze the effectiveness of the classification models.

---

# Conclusion

This project demonstrates the practical application of **Natural Language Processing and Machine Learning in brand analysis and classification**. By combining **DistilRoBERT embeddings with machine learning models**, the system effectively predicts industry categories for brand names.

The solution can be further expanded to support **real-time brand analytics, marketing intelligence, and automated business classification systems**.

