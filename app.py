from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('iris_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([np.array(features)])
    flower = ['Setosa', 'Versicolor', 'Virginica'][prediction[0]]
    return render_template('index.html', prediction_text=f'Predicted Flower: {flower}')

if __name__ == '__main__':
    app.run(debug=True)
