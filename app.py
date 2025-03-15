from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))
le1 = pickle.load(open('le1.pkl', 'rb'))
le2 = pickle.load(open('le2.pkl', 'rb'))
ct = pickle.load(open('ct.pkl', 'rb'))
sc = pickle.load(open('sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form['model'].strip() 
        year = int(request.form['year'])
        transmission = request.form['transmission']
        mileage = float(request.form['mileage'])
        fuel_type = request.form['fuel_type'].strip() 
        tax = float(request.form['tax'])
        mpg = float(request.form['mpg'])
        engine_size = float(request.form['engine_size'])

        print("\n Received Inputs:")
        print(f"Model: {model_name}, Year: {year}, Transmission: {transmission}, Mileage: {mileage}, Fuel Type: {fuel_type}, Tax: {tax}, MPG: {mpg}, Engine Size: {engine_size}")

        available_models = [m.strip() for m in le1.classes_]
        print(" Available Models in Label Encoder:", available_models)

        if model_name not in available_models:
            return render_template('index.html', prediction_text=f"Error: Model name '{model_name}' not recognized! Available models: {available_models}")

        if fuel_type not in le2.classes_:
            return render_template('index.html', prediction_text=f"Error: Fuel type '{fuel_type}' not recognized!")

        model_encoded = le1.transform([model_name])[0]
        fuel_encoded = le2.transform([fuel_type])[0]

        print(f" Encoded Model: {model_encoded}, Encoded Fuel Type: {fuel_encoded}")

        input_data = np.array([[model_encoded, year, transmission, mileage, fuel_encoded, tax, mpg, engine_size]])

        print(f" Before One-Hot Encoding: Input shape = {input_data.shape}")

        input_data = ct.transform(input_data)

        print(f" After One-Hot Encoding: Input shape = {input_data.shape}")

        input_data = sc.transform(input_data)

        print(" Transformed Input Data:", input_data)

        predicted_price = model.predict(input_data)[0]

        print(f" Predicted Price: £{predicted_price:.2f}")

        return render_template('index.html', prediction_text=f'Predicted Car Price: £{predicted_price:.2f}')
    
    except Exception as e:
        print(" Error:", str(e))
        return render_template('index.html', prediction_text=f"Error in prediction: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)
