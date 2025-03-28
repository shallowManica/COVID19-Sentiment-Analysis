# COVID-19 Tweet Sentiment Analysis Using CNN, RNN and LSTM

## Overview
This repository contains a Jupyter Notebook that demonstrates sentiment analysis on COVID-19-related tweets using deep learning techniques. The goal is to classify tweets into sentiment categories (e.g., Negative, Positive, Other) by exploring and comparing three deep learning architectures:
- **Convolutional Neural Network (CNN)**
- **Artificial Neural Network (ANN)**
- **Long Short-Term Memory (LSTM) Network**

These models help reveal patterns in public sentiment during the COVID-19 pandemic.

## Dataset
The dataset is sourced from [Kaggle: Coronavirus tweets NLP - Text Classification](https://www.kaggle.com/datasets/datatattle/covid-19-nlp-text-classification). It comprises tweets related to the COVID-19 pandemic that have been manually tagged for sentiment. Key columns in the dataset include:
- **Location:** The origin of the tweet.
- **Tweet At:** Timestamp of when the tweet was posted.
- **Original Tweet:** The text content of the tweet.
- **Label:** The manually assigned sentiment (e.g., Negative, Positive, Other).

## Features
- **Data Preprocessing:** Clean and tokenize tweet texts to prepare for model input.
- **Modeling:** Implementation and comparative analysis of three models:
  - **CNN:** To capture local features from text sequences.
  - **ANN:** A baseline deep learning model for text classification.
  - **LSTM:** To capture sequential dependencies and context in tweets.
- **Evaluation & Visualization:** Analyze model performance using accuracy metrics and visualize sentiment distribution and trends.

## Requirements
- Python 3.x
- Jupyter Notebook or JupyterLab

Required Python packages:
- pandas
- numpy
- nltk
- tensorflow (or keras)
- matplotlib
- seaborn

## Installation & Setup
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/COVID19-Sentiment-Analysis.git
   ```
2. **Navigate to the Project Directory:**
   ```bash
   cd COVID19-Sentiment-Analysis
   ```

## Usage
- **Launch Jupyter Notebook:**
  ```bash
  jupyter notebook
  ```
- Open the `sentiment_analy.ipynb` notebook and run the cells sequentially to preprocess the data, train the models, and visualize the results.

## File Structure
- `sentiment_analy.ipynb`: The main notebook with the complete sentiment analysis workflow.
- `data/`: Directory to store the COVID-19 tweets dataset (or instructions to download it from Kaggle).
