# ANA 680 – Wine Quality ML Deployment

This project trains a linear regression model on the UCI Wine Quality dataset and deploys it as a REST API using Docker and Heroku.

## Project Repository
GitHub: https://github.com/Jkrans/ana680-wine-quality-ml-deployment

## Live Application (Heroku)
Health Check Endpoint:
https://jk-wine-quality-api-0f73a9de9776.herokuapp.com/health

## Endpoints

- `GET /health` → Returns application status
- `POST /predict` → Returns predicted wine quality score

## Example Prediction Request

```bash
curl -X POST https://jk-wine-quality-api-0f73a9de9776.herokuapp.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11,
    "total sulfur dioxide": 34,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'