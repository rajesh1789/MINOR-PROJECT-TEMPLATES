from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained RandomForestClassifier model
model = pickle.load(open("model1.pkl", 'rb'))

# Define the feature columns based on the model's requirements
feature_cols = ['gender', 'age', 'admission_type_id', 'discharge_disposition_id',
                'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
                'metformin', 'repaglinide', 'glipizide', 'insulin', 'change',
                'diabetesMed', 'age_derived', 'count_Steady', 'count_Down', 'count_Up']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        return render_template("index.html")
    return render_template("index.html")

@app.route('/out', methods=['POST'])
def output():
    # Collect data from the form
    gender = request.form['gender']
    age = request.form['age']
    admission_type_id = request.form['admission_type_id']
    discharge_disposition_id = request.form['discharge_disposition_id']
    admission_source_id = request.form['admission_source_id']
    time_in_hospital = request.form['time_in_hospital']
    num_lab_procedures = request.form['num_lab_procedures']
    num_medications = request.form['num_medications']
    number_outpatient = request.form['number_outpatient']
    number_emergency = request.form['number_emergency']
    number_inpatient = request.form['number_inpatient']
    diag_1 = request.form['diag_1']
    diag_2 = request.form['diag_2']
    diag_3 = request.form['diag_3']
    number_diagnoses = request.form['number_diagnoses']
    metformin = request.form['metformin']
    repaglinide = request.form['repaglinide']
    glipizide = request.form['glipizide']
    insulin = request.form['insulin']
    change = request.form['change']
    diabetesMed = request.form['diabetesMed']
    age_derived = request.form['age_derived']
    count_Steady = request.form['count_Steady']
    count_Down = request.form['count_Down']
    count_Up = request.form['count_Up']

    # Create a DataFrame with the collected data
    data = [[gender, age, admission_type_id, discharge_disposition_id,
             admission_source_id, time_in_hospital, num_lab_procedures,
             num_medications, number_outpatient, number_emergency,
             number_inpatient, diag_1, diag_2, diag_3, number_diagnoses,
             metformin, repaglinide, glipizide, insulin, change,
             diabetesMed, age_derived, count_Steady, count_Down, count_Up]]

    # Convert data into a DataFrame using feature_cols
    input_df = pd.DataFrame(data, columns=feature_cols)

    # Perform integer conversion for relevant fields
    try:
        # Attempt to convert relevant fields to integers
        input_df['age'] = input_df['age'].astype(int)
        input_df['admission_type_id'] = input_df['admission_type_id'].astype(float)
        input_df['discharge_disposition_id'] = input_df['discharge_disposition_id'].astype(float)
        input_df['admission_source_id'] = input_df['admission_source_id'].astype(float)
        input_df['time_in_hospital'] = input_df['time_in_hospital'].astype(float)
        input_df['num_lab_procedures'] = input_df['num_lab_procedures'].astype(float)
        input_df['num_medications'] = input_df['num_medications'].astype(float)
        input_df['number_outpatient'] = input_df['number_outpatient'].astype(float)
        input_df['number_emergency'] = input_df['number_emergency'].astype(float)
        input_df['number_inpatient'] = input_df['number_inpatient'].astype(float)
        input_df['number_diagnoses'] = input_df['number_diagnoses'].astype(float)
        input_df['age_derived'] = input_df['age_derived'].astype(float)
        input_df['count_Steady'] = input_df['count_Steady'].astype(float)
        input_df['count_Down'] = input_df['count_Down'].astype(float)
        input_df['count_Up'] = input_df['count_Up'].astype(float)
    except ValueError as e:
        return render_template("error.html", message=str(e))  # Handle the error gracefully

    # Perform prediction using the model
    pred = model.predict(input_df)
    pred = pred[0]

    # Prepare result message based on prediction
    if pred:
        return render_template("output.html", y="This patient will be readmitted")
    else:
        return render_template("output.html", y="This patient will not be readmitted")

if __name__ == '__main__':
    app.run(debug=True)
