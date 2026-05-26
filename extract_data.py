import os
import json
import pandas as pd
import re

BASE_DIR = "Open_Set"

SELECTED_MODELS = [
   "vit_base_patch16_224","DeiTModel","PVTModel","TNTModel","CrossViTModel","swinv2_base_window12_192_22k","MAEModel","BEiTModel","SimMIMModel","MobileViTModel","MaxViTModel","swinv2_small_window8_256_22k"
]

# 🔥 Custom attack order
ATTACK_ORDER = [
    "CL",
    "E-display",
    "Fake with Add On",
    "Generated",
    "PostMortem",
    "Print and E-display",
    "Printed"
]

rows = []

for model_name in SELECTED_MODELS:
    model_path = os.path.join(BASE_DIR, model_name)

    if not os.path.exists(model_path):
        continue

    for attack_name in os.listdir(model_path):
        attack_path = os.path.join(model_path, attack_name)

        if not os.path.isdir(attack_path):
            continue

        logs_path = os.path.join(attack_path, "Logs")
        results_path = os.path.join(attack_path, "Results")

        if not os.path.exists(logs_path) or not os.path.exists(results_path):
            continue

        # -------- Accuracy --------
        report_files = [f for f in os.listdir(logs_path) if f.endswith("_report.json")]
        if not report_files:
            continue

        with open(os.path.join(logs_path, report_files[0]), "r") as f:
            report_data = json.load(f)

        accuracy = report_data.get("accuracy", None)

        # -------- ACER parsing --------
        tdr_files = [f for f in os.listdir(results_path) if f.endswith("_TDR-ACER.csv")]
        if not tdr_files:
            continue

        with open(os.path.join(results_path, tdr_files[0]), "r") as f:
            content = f.read()

        pattern = r"([\d\.]+)\s+and\s+([\d\.]+)\s+and\s+([\d\.]+)\s+@\s+([\d\.]+)\s+and\s+([\d\.]+)\s+and\s+([\d\.]+)"
        match = re.search(pattern, content)

        if match:
            apcer = float(match.group(1))
            bpcer = float(match.group(3))
            acer = float(match.group(4))
            threshold = float(match.group(6))
        else:
            apcer = bpcer = acer = threshold = None

        rows.append({
            "Model": model_name,
            "Attack_Type": attack_name,
            "APCER": apcer,
            "BPCER": bpcer,
            "ACER": acer,
            "Threshold": threshold,
            "Accuracy": accuracy
        })

# ==============================
# Create DataFrame
# ==============================
final_df = pd.DataFrame(rows)

# 🔥 Apply custom attack order
final_df["Attack_Type"] = pd.Categorical(
    final_df["Attack_Type"],
    categories=ATTACK_ORDER,
    ordered=True
)

# 🔥 Sort by Model then Attack order
final_df = final_df.sort_values(["Model", "Attack_Type"])

# Keep column order
final_df = final_df[
    ["Model", "Attack_Type", "APCER", "BPCER", "ACER", "Threshold", "Accuracy"]
]

final_df.to_csv("Vision_models.csv", index=False)

print("Saved Vision_models.csv")
print(final_df)