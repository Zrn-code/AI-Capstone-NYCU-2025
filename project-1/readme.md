# AI Capstone Project - NYCU 2025

## Project Overview

This repository contains a stock market prediction project that uses natural language processing (NLP) techniques to analyze news headlines and predict stock market movements. The project is structured into two main components:

1. **Data Collection System** - A crawler that gathers stock market data and relevant news headlines from Google News
2. **Machine Learning System** - An analysis pipeline that processes news data and trains multiple models to predict market directions

## Features

### Data Collection (`crawler.py`)

- Fetches historical stock market data from Yahoo Finance
- Collects Google News headlines across five categories:
  - Technology
  - Business
  - Market
  - Economy
  - World events
- Combines stock data with news headlines for analysis
- Handles data preprocessing and date alignment

### Machine Learning Analysis (`ai.py`)

- Text preprocessing (removing stopwords, punctuation, etc.)
- Feature extraction using TF-IDF vectorization
- Implements multiple classification models:
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
- Provides cross-validation and performance evaluation
- Includes unsupervised learning techniques:
  - K-Means clustering
  - DBSCAN clustering
- Handles class imbalance through resampling
