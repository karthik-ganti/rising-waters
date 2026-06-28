from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
from datetime import datetime

db = SQLAlchemy()


class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(50), nullable=False, default='Meteorologist')

    weather_data = db.relationship('WeatherData', backref='user', lazy=True)


class WeatherData(db.Model):
    __tablename__ = 'weather_data'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    annual_rainfall = db.Column(db.Float, nullable=False)
    cloud_visibility = db.Column(db.Float, nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    seasonal_rainfall = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    prediction = db.relationship('PredictionResult', backref='weather_data', uselist=False, lazy=True)


class MLModel(db.Model):
    __tablename__ = 'ml_model'

    id = db.Column(db.Integer, primary_key=True)
    model_name = db.Column(db.String(100), nullable=False)
    algorithm_type = db.Column(db.String(100), nullable=False)
    accuracy = db.Column(db.Float, nullable=False)
    model_file = db.Column(db.String(200), nullable=False)

    predictions = db.relationship('PredictionResult', backref='ml_model', lazy=True)


class PredictionResult(db.Model):
    __tablename__ = 'prediction_result'

    id = db.Column(db.Integer, primary_key=True)
    data_id = db.Column(db.Integer, db.ForeignKey('weather_data.id'), nullable=False)
    model_id = db.Column(db.Integer, db.ForeignKey('ml_model.id'), nullable=False)
    flood_result = db.Column(db.String(20), nullable=False)
    flood_probability = db.Column(db.Float, nullable=False)
    prediction_date = db.Column(db.DateTime, default=datetime.utcnow)
