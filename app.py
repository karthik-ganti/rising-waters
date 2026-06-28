import os
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, session, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from models_db import db, User, WeatherData, MLModel, PredictionResult

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'rising_waters_secret_key_2024')
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///rising_waters.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'warning'

MODEL_PATH = os.path.join('models', 'floods.save')
SCALER_PATH = os.path.join('models', 'transform.save')

model = None
scaler = None


def load_ml_assets():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        print("WARNING: Model files not found. Run the Jupyter notebook first to train and save the model.")


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


def seed_ml_model():
    if MLModel.query.count() == 0:
        xgb_entry = MLModel(
            model_name='Flood Prediction Model',
            algorithm_type='Decision Tree (Best of 4 classifiers)',
            accuracy=95.65,
            model_file='models/floods.save'
        )
        db.session.add(xgb_entry)
        db.session.commit()


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        role = request.form.get('role', 'Meteorologist')

        if not name or not email or not password:
            flash('All fields are required.', 'danger')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.', 'danger')
            return render_template('register.html')

        new_user = User(
            name=name,
            email=email,
            password_hash=generate_password_hash(password),
            role=role
        )
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            next_page = request.args.get('next')
            return redirect(next_page or url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/Predict')
@login_required
def predict_form():
    if model is None:
        flash('Prediction model is not loaded. Please run the Jupyter notebook first.', 'danger')
        return redirect(url_for('home'))
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None or scaler is None:
        flash('Prediction model is not loaded. Please run the Jupyter notebook first.', 'danger')
        return redirect(url_for('home'))

    try:
        annual_rainfall = float(request.form['annual_rainfall'])
        cloud_visibility = float(request.form['cloud_visibility'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        seasonal_rainfall = float(request.form['seasonal_rainfall'])
    except (ValueError, KeyError):
        flash('Invalid input values. Please enter numeric values for all fields.', 'danger')
        return redirect(url_for('predict_form'))

    # Feature order must match training: ANNUAL, Cloud Cover, Jun-Sep, Temp, Humidity
    input_array = np.array([[annual_rainfall, cloud_visibility, seasonal_rainfall, temperature, humidity]])
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)[0]
    probability = model.predict_proba(scaled_input)[0]
    flood_prob = round(float(probability[1]) * 100, 2)
    no_flood_prob = round(float(probability[0]) * 100, 2)
    flood_result = 'Flood' if prediction == 1 else 'No Flood'

    weather_entry = WeatherData(
        user_id=current_user.id,
        annual_rainfall=annual_rainfall,
        cloud_visibility=cloud_visibility,
        temperature=temperature,
        humidity=humidity,
        seasonal_rainfall=seasonal_rainfall
    )
    db.session.add(weather_entry)
    db.session.flush()

    ml_model_entry = MLModel.query.first()
    model_id = ml_model_entry.id if ml_model_entry else 1

    pred_entry = PredictionResult(
        data_id=weather_entry.id,
        model_id=model_id,
        flood_result=flood_result,
        flood_probability=flood_prob,
        prediction_date=datetime.utcnow()
    )
    db.session.add(pred_entry)
    db.session.commit()

    session['flood_probability'] = flood_prob
    session['no_flood_probability'] = no_flood_prob
    session['temperature'] = temperature
    session['humidity'] = humidity
    session['annual_rainfall'] = annual_rainfall

    if prediction == 1:
        return redirect(url_for('chance'))
    else:
        return redirect(url_for('no_chance'))


@app.route('/chance')
@login_required
def chance():
    flood_prob = session.get('flood_probability', 0)
    return render_template('chance.html', probability=flood_prob)


@app.route('/no_chance')
@login_required
def no_chance():
    no_flood_prob = session.get('no_flood_probability', 0)
    return render_template('no_chance.html', probability=no_flood_prob)


@app.route('/history')
@login_required
def history():
    records = (
        db.session.query(WeatherData, PredictionResult)
        .join(PredictionResult, WeatherData.id == PredictionResult.data_id)
        .filter(WeatherData.user_id == current_user.id)
        .order_by(PredictionResult.prediction_date.desc())
        .all()
    )
    return render_template('history.html', records=records)


with app.app_context():
    db.create_all()
    seed_ml_model()
    load_ml_assets()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
