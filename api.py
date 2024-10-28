import warnings
from flask_cors import CORS
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
import calendar
warnings.filterwarnings("ignore")
app = Flask(__name__)
CORS(app)

# Load the models
efficiency_model = joblib.load('multi_output_model_weekly.joblib')
le = joblib.load('label_encoder.joblib')
daily_model = joblib.load('MultiOutputRegressor.pkl')


def predict_efficiency(month, week, res_elev, rain, storage, total_sluice_issue, inflow):
    if isinstance(month, str):
        if month.isdigit():
            month_numeric = int(month)
        else:
            try:
                month_numeric = list(calendar.month_name).index(month.capitalize())
            except ValueError:
                month_numeric = list(calendar.month_abbr).index(month.capitalize()[:3])
    else:
        month_numeric = month
    if month_numeric < 1 or month_numeric > 12:
        raise ValueError("Invalid month value")

    month_encoded = le.transform([month_numeric])[0]
    month_names = list(calendar.month_name)[1:]  

    predictions = []
    for w in range(week, 5):
        input_data = pd.DataFrame({
            'Month': [month_encoded],
            'Week': [w],
            'RES_ELEV': [res_elev + np.random.normal(0, 5)],
            'RAIN': [max(0, rain + np.random.normal(0, 2))],
            'Storage': [storage + np.random.normal(0, 50)],
            'Total_Sluice_Issue': [total_sluice_issue],
            'Inflow': [max(0, inflow + np.random.normal(0, 5))],
        })

        input_data = input_data[['Month', 'Week', 'RES_ELEV', 'RAIN', 'Storage', 'Total_Sluice_Issue', 'Inflow']]
        week_prediction = efficiency_model.predict(input_data)

        # Extract the predicted POWER_FLOW and ENERGY
        power_flow_pred = week_prediction[0][0]
        energy_pred = week_prediction[0][1]
        efficiency_pred = week_prediction[0][2]

        predictions.append([f'{power_flow_pred:.2f}', f'{energy_pred:.2f}', f'{efficiency_pred:.2f}'])

    return predictions, month_names[month_numeric - 1]

# Define the route for daily predictions
@app.route('/daily', methods=['POST'])
def daily():
    try:
        data = request.json
        required_fields = ['RES_ELEV', 'RAIN', 'Storage', 'Total_Sluice_Issue', 'Inflow']
        for field in required_fields:
            if data.get(field) is None:
                raise ValueError(f"Missing value for {field}")

        features = [float(data.get(feature)) for feature in required_fields]
        features_array = np.array([features])

        predictions = daily_model.predict(features_array)

        power_flow_pred = predictions[0][0]
        energy_pred = predictions[0][1]
        efficiency_pred = predictions[0][2]
        response = {
            "power_flow_pred": power_flow_pred,
            "energy_pred": energy_pred,
            "efficiency_pred": efficiency_pred
        }
        return jsonify({'predicted_efficiency': response})
    except ValueError as ve:
        return jsonify({'error': str(ve)}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Define the route for weekly predictions
@app.route('/weekly', methods=['POST'])
def weekly():

    try:
        data = request.get_json()

        month = data.get('month')
        week = int(data.get('week'))
        res_elev = float(data.get('res_elev'))
        rain = float(data.get('rain'))
        storage = float(data.get('storage'))
        total_sluice_issue = float(data.get('total_sluice_issue'))
        inflow = float(data.get('inflow'))

        predictions, month_name = predict_efficiency(month, week, res_elev, rain, storage, total_sluice_issue, inflow)
        print(predictions, month_name)
        weekly_data = {}
        for i, prediction in enumerate(predictions):
            power_flow_pred = float(prediction[0])
            energy_pred = float(prediction[1])
            efficiency_pred = float(prediction[2])

            week_key = f'option{i+1}' 
            weekly_data[week_key] = {
                'efficiency': f'{efficiency_pred:.2f}',
                'energy': f'{energy_pred:.2f}',
                'power_flow': f'{power_flow_pred:.2f}'
            }
        response = {
            'month': month_name,
            'weeks': weekly_data
        }

        return jsonify([response])

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': 'Bad Request', 'message': str(e)}), 400



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001)
