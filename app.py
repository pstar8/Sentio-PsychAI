from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F
import joblib




# Load model and tokenizer
model = joblib.load("psychai_model.pkl")
vectorizer = joblib.load("psychai_vectorizer.pkl")

# Load the tokenizer (make sure the path matches your model)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



# model_path = "./psychai_model"
# tokenizer = BertTokenizer.from_pretrained(model_path)
# model = BertForSequenceClassification.from_pretrained(model_path)
# model.eval()

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    print("User Input:", user_input)

    # Tokenize input
    X_input = vectorizer.transform([user_input])
    # inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model.predict(X_input)[0]


    # Map to your label list (must match training order)
    labels = ["Anxiety", " Depression", " Sleeping Disorder", " Chronic Stress", 
              " Excessive Worry", " Bipolar Disorder", " OCD  —  Obessive Complusive Disorder ", "  Borderline Personality Disorder", " PTSD  —  Post Traumatic Stress Disorder", " Social Anxiety", " Autism Disorder", " Dissociative Disorder",
              " ADHD", " Grief", " Addiction", " Low Self Esteem", " Anger Issues", " Relationship Issues",
              " Caregiver Fatigue", " Academic Stress"]
    prediction = [label for label, value in zip(labels, outputs) if value == 1]
    print("Prediction:", prediction)

    return render_template('index.html', prediction=", ".join(prediction) or "No condition detected", user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)


    







    # To keep userText after analysis, modify the predict route to pass user_input back to the template.
    # In your index.html, ensure the textarea/input for userText uses value="{{ user_input|default('') }}".

    # Example for predict route (already above):
    # return render_template('index.html', prediction=", ".join(prediction_result) or "No condition detected", user_input=user_input)