from django.shortcuts import render
import pickle

import os
from django.conf import settings

# Load models once
BASE_DIR = settings.BASE_DIR
vectorizer = pickle.load(open(os.path.join(BASE_DIR, 'vectorizer.pkl'),'rb'))
nb = pickle.load(open(os.path.join(BASE_DIR, 'nb.pkl'),'rb'))
svm = pickle.load(open(os.path.join(BASE_DIR, 'svm.pkl'),'rb'))
lr = pickle.load(open(os.path.join(BASE_DIR, 'lr.pkl'),'rb'))

# Model info
model_info = {
    "Naive Bayes": {"accuracy": "96%", "acc_val": 96.00, "f1_val": 94.00},
    "SVM": {"accuracy": "97%", "acc_val": 97.00, "f1_val": 95.00},
    "Logistic Regression": {"accuracy": "95%", "acc_val": 95.00, "f1_val": 92.00}
}

model_objects = {
    "Naive Bayes": nb,
    "SVM": svm,
    "Logistic Regression": lr,
}

def get_spam_reasons(text):
    text_lower = text.lower()
    spam_keywords = [
        "free", "win", "prize", "money", "urgent", "click", "guarantee", 
        "offer", "discount", "winner", "cash", "lottery", "claim", "limited", 
        "congratulations", "act now", "100%", "urgent", "important", "alert"
    ]
    found = [kw for kw in spam_keywords if kw in text_lower]
    return found

def home(request):
    results = {}
    message = ""
    selected_model = "Naive Bayes"
    selected_result = None

    if request.method == "POST":
        message = (request.POST.get("message") or "").strip()
        posted_model = (request.POST.get("selected_model") or "").strip()
        if posted_model in model_objects:
            selected_model = posted_model

        if message:
            data = vectorizer.transform([message])
            for model_name, model_obj in model_objects.items():
                prediction = model_obj.predict(data)[0]
                results[model_name] = {
                    "prediction": "Spam" if prediction else "Not Spam",
                    "accuracy": model_info[model_name]["accuracy"],
                    "acc_val": model_info[model_name]["acc_val"],
                    "f1_val": model_info[model_name]["f1_val"],
                }

            selected_result = results.get(selected_model)

    context = {
        "results": results,
        "message": message,
        "selected_model": selected_model,
        "selected_result": selected_result,
        "spam_reasons": get_spam_reasons(message) if selected_result and selected_result['prediction'] == 'Spam' else [],
        "model_choices": list(model_objects.keys()),
        "best_model": max(model_info, key=lambda name: model_info[name]["f1_val"]),
        "best_model_prediction": results.get(max(model_info, key=lambda name: model_info[name]["f1_val"]), {}).get("prediction") if results else None,
    }
    return render(request, "index.html", context)