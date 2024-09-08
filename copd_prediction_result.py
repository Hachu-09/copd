import pickle
import numpy as np
import pandas as pd
import google.generativeai as gemini

# Configure the API
gemini.configure(api_key="AIzaSyASUFBrNl_EsBuo8QD2_1HDGZXlcVAiG_o")

# Load the model and scaler from the pickle file
with open('RESPIRATORY_MODEL/COPD/copd_prediction.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)
    ensemble_model = saved_data['model']
    scaler = saved_data['scaler']

# Define the column names based on the original dataset
column_names = ['AGE', 'PackHistory', 'COPDSEVERITY', 'MWT1', 'MWT2', 'MWT1Best', 'FEV1', 'FEV1PRED', 'FVC', 'FVCPRED', 
                'CAT', 'HAD', 'SGRQ', 'AGEquartiles', 'copd', 'gender', 'smoking', 'Diabetes', 'muscular', 'hypertension', 'AtrialFib']

# Function to predict risk based on input data
def predict_copd(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_df = pd.DataFrame([input_data_as_numpy_array], columns=column_names)  # Create DataFrame with feature names
    input_data_scaled = scaler.transform(input_data_df)  # Scale input data
    
    # Predict using the ensemble model
    prediction = ensemble_model.predict(input_data_scaled)[0]
    
    # Get the probability of the positive class (COPD)
    probability = ensemble_model.predict_proba(input_data_scaled)[0][1]
    risk_percentage = probability * 100

    # Determine risk
    risk = "The person is at risk of COPD" if prediction == 1 else "The person is not at risk of COPD"
    
    # Determine the type of disease
    disease_type = "Chronic Obstructive Pulmonary Disease (COPD)" if prediction == 1 else "No COPD"

    return prediction, risk, risk_percentage, disease_type

AGE = float(input("Enter AGE: "))
PackHistory = float(input("Enter PackHistory: "))
COPDSEVERITY = float(input("Enter COPDSEVERITY: "))
MWT1 = float(input("Enter MWT1: "))
MWT2 = float(input("Enter MWT2: "))
MWT1Best = float(input("Enter MWT1Best: "))
FEV1 = float(input("Enter FEV1: "))
FEV1PRED = float(input("Enter FEV1PRED: "))
FVC = float(input("Enter FVC: "))
FVCPRED = float(input("Enter FVCPRED: "))
CAT = float(input("Enter CAT: "))
HAD = float(input("Enter HAD: "))
SGRQ = float(input("Enter SGRQ: "))
AGEquartiles = float(input("Enter AGEquartiles: "))
copd = float(input("Enter copd: "))
gender = float(input("Enter gender (1 = Male, 0 = Female): "))
smoking = float(input("Enter smoking (1 = Yes, 0 = No): "))
Diabetes = float(input("Enter Diabetes (1 = Yes, 0 = No): "))
muscular = float(input("Enter muscular (1 = Yes, 0 = No): "))
hypertension = float(input("Enter hypertension (1 = Yes, 0 = No): "))
AtrialFib = float(input("Enter AtrialFib (1 = Yes, 0 = No): "))

# Combine user input into a single tuple
input_data = (AGE, PackHistory, COPDSEVERITY, MWT1, MWT2, MWT1Best, FEV1, FEV1PRED, FVC, FVCPRED, CAT, HAD, SGRQ, AGEquartiles, copd, gender, smoking, Diabetes, muscular, hypertension, AtrialFib)

# Predict and display results
prediction, risk, risk_percentage, disease_type = predict_copd(input_data)

# Display results
print(f"Risk: {risk}")
print(f"Risk Percentage: {risk_percentage:.2f}%")
print(f"Problem: {disease_type}")

# Function to generate prevention report
def generate_prevention_report(risk, disease, age):
    prompt = f"""
    Provide a general wellness report with the following sections:

    1. **Introduction**
        -Purpose of the Report: Clearly state why this report is being generated, including its relevance to the individual’s health.
        -Overview of Health & Wellness: Briefly describe the importance of understanding and managing health risks, with a focus on proactive wellness and disease prevention.
        -Personalized Context: Include the user's specific details such as age, gender, and any relevant medical history that can be linked to the risk factor and disease.

    2. **Risk Description**
        -Detailed Explanation of Risk: Describe the identified risk factor in detail, including how it impacts the body and its potential consequences if left unaddressed.
        -Associated Conditions: Mention any other health conditions commonly associated with this risk factor.
        -Prevalence and Statistics: Provide some general statistics or prevalence rates to contextualize the risk (e.g., how common it is in the general population or specific age groups).

    3. **Stage of Risk**
        -Risk Level Analysis: Provide a more granular breakdown of the risk stages (e.g., low, medium, high), explaining what each stage means in terms of potential health outcomes.
        -Progression: Discuss how the risk may progress over time if not managed, and what signs to watch for that indicate worsening or improvement.

    4. **Recommendations**
        -Personalized Action Plan: Suggest specific, actionable steps the individual can take to mitigate the risk or manage the disease (e.g., dietary changes, exercise plans, medical treatments).
        -Lifestyle Modifications: Tailor suggestions to the individual’s lifestyle, providing practical tips for integrating these changes.
        -Monitoring and Follow-up: Recommend how the user should monitor their health and when to seek follow-up care.

    5. **Conclusion**
        -Summary of Key Points: Recap the most important points from the report, focusing on what the individual should remember and prioritize.
        -Encouragement: Offer positive reinforcement and encouragement for taking proactive steps toward better health.

    **Details:**
    Risk: {risk}
    Disease: {disease}
    Age: {age}

    Note: This information is for general wellness purposes. For specific health concerns, consult a healthcare professional.
    """

    try:
        response = gemini.generate_text(
            prompt=prompt,
            temperature=0.5,
            max_output_tokens=1000
        )
        
        report = response.result if hasattr(response, 'result') else None
        
        if not report:
            print("The response from the API did not contain a result.")
        
        return report
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Check if the person has COPD (prediction == 1) before generating the report
if prediction == 1:
    # Collect additional inputs for report generation
    name = input("\nEnter your Name: ")
    age = int(input("Enter your Age: "))

    # Generate the report
    report = generate_prevention_report(
        risk=risk_percentage,  # Use the percentage of risk
        disease=disease_type,  # Disease type (COPD)
        age=age
    )

    if report:
        print("\nGenerated Wellness Report:")
        print(report)
    else:
        print("Failed to generate a report. Please check the API response and try again.")
else:
    print("The person is not at risk of COPD, so no report will be generated.")
