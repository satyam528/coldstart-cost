import joblib
import pandas as pd
from django.conf import settings
from django.shortcuts import render

# Load the trained model ONCE when the server starts.
MODEL_PATH = settings.PROJECT_ROOT / "models" / "model.pkl"
model = joblib.load(MODEL_PATH)

MEMORY_OPTIONS = [128, 256, 512, 1024, 2048, 3008]
RUNTIME_OPTIONS = ["python", "node", "java"]


def index(request):
    context = {
        "memory_options": MEMORY_OPTIONS,
        "runtime_options": RUNTIME_OPTIONS,
        "values": {"memory_mb": 512, "runtime": "python", "package_mb": 20, "idle_min": 10},
    }

    if request.method == "POST":
        try:
            values = {
                "memory_mb": int(request.POST["memory_mb"]),
                "runtime": request.POST["runtime"],
                "package_mb": float(request.POST["package_mb"]),
                "idle_min": float(request.POST["idle_min"]),
            }
            if values["runtime"] not in RUNTIME_OPTIONS:
                raise ValueError("bad runtime")
            if not (1 <= values["package_mb"] <= 250 and 0 <= values["idle_min"] <= 600):
                raise ValueError("out of range")

            row = pd.DataFrame([values])
            micro_usd = float(model.predict(row)[0])           # model outputs micro-USD
            context["values"] = values
            context["result_usd"] = f"{micro_usd / 1e6:.8f}"
            context["per_million"] = f"{micro_usd:.2f}"        # micro-USD per call = USD per 1M calls
        except (KeyError, ValueError):
            context["error"] = "Please enter valid values (package 1-250 MB, idle 0-600 min)."

    return render(request, "predictor/index.html", context)
