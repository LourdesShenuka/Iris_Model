# Iris Flower Classification API

## Problem Description
This project predicts the species of Iris flowers (setosa, versicolor, virginica) based on four features: sepal length, sepal width, petal length, and petal width.

## Dataset
- The dataset is built-in from `scikit-learn` (`load_iris()`)
- Contains 150 samples with 4 features
- Target classes: `setosa`, `versicolor`, `virginica`

## Model
- **Model Type:** RandomForestClassifier
- **Reason for choice:** Works well on small datasets, provides high accuracy, and gives probability estimates.
- **Accuracy:** 100% on the test set

## Front-End
- Simple, responsive interface built with **Bootstrap**
- Users can input flower features and get predictions in real-time

## API Endpoints

### 1. Health Check
- **Endpoint:** `GET /health`
- **Response Example:**
```json
{
  "status": "healthy",
  "message": "ML Model API is running"
}