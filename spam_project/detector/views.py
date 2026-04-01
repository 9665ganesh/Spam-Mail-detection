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
    "Naive Bayes": {"accuracy": "97%", "recall": "0.83"},
    "SVM": {"accuracy": "98%", "recall": "0.90"},
    "Logistic Regression": {"accuracy": "98%", "recall": "0.94"}
}

model_objects = {
    "Naive Bayes": nb,
    "SVM": svm,
    "Logistic Regression": lr,
}

def recall_value(recall_text):
    return float(str(recall_text).strip())

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
                    "recall": model_info[model_name]["recall"],
                }

            selected_result = results.get(selected_model)

    context = {
        "results": results,
        "message": message,
        "selected_model": selected_model,
        "selected_result": selected_result,
        "model_choices": list(model_objects.keys()),
        "best_model": max(model_info, key=lambda name: recall_value(model_info[name]["recall"])),
    }
    return render(request, "index.html", context)