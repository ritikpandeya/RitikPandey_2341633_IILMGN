# predict.py
# Live predictor that ALSO accepts a measured power reading and compares it to the ML prediction.
# Save this file in the same folder as model.pkl

from joblib import load
import pandas as pd
import numpy as np

# ---- CONFIG ----
MODEL_FILENAME = 'model.pkl'   # change if your model filename is different
WASTE_DIFF_KW = 1.0           # if measured > predicted + this -> flag "Possible Waste/Anomaly"
WASTE_PERCENT = 0.30          # or if measured > predicted * (1 + this) -> flag

# ---- load model ----
try:
    model = load(MODEL_FILENAME)
except Exception as e:
    print("ERROR: Could not load model file:", MODEL_FILENAME)
    print("Exception:", e)
    raise SystemExit(1)

print("\n--- REAL-TIME ROOM ENERGY PREDICTOR (Measured vs Predicted) ---")
print("Enter values your teacher gives. Press Ctrl+C to quit.\n")

while True:
    try:
        room = int(input("Room number (e.g., 208): ").strip())
        area = float(input("Room area in sq.ft (e.g., 350): ").strip())
        appliances = int(input("Number of appliances (e.g., 4): ").strip())
        temperature = float(input("Temperature now (°C) (e.g., 27): ").strip())
        time_of_day = int(input("Hour (0-23) (e.g., 14): ").strip())
        measured_power = float(input("Measured power right now (kWh) (e.g., 11.5): ").strip())
    except KeyboardInterrupt:
        print("\nUser requested exit. Goodbye!")
        break
    except Exception as e:
        print("Invalid input — please enter numeric values. Error:", e)
        continue

    # Prepare input exactly as training features
    new_df = pd.DataFrame([{
        'room': room,
        'area': area,
        'appliances': appliances,
        'temperature': temperature,
        'time_of_day': time_of_day
    }])

    # Make prediction
    pred = model.predict(new_df)[0]

    # Comparison logic
    diff_kw = measured_power - pred
    diff_pct = (diff_kw / pred) if pred != 0 else float('inf')

    # Determine status
    status = "Normal"
    reason = ""
    if (diff_kw > WASTE_DIFF_KW) or (diff_pct > WASTE_PERCENT):
        status = "POSSIBLE WASTE / ANOMALY"
        reason = f"Measured > Predicted by {diff_kw:.2f} kWh ({diff_pct*100:.1f}%)"
    elif diff_kw < -WASTE_DIFF_KW and diff_pct < -WASTE_PERCENT:
        status = "Measured MUCH LOWER than expected (check meter or inputs)"
        reason = f"Measured lower by {-diff_kw:.2f} kWh ({-diff_pct*100:.1f}%)"

    # Friendly interpretation
    if status == "Normal":
        if pred < 5:
            interp = "Low expected usage"
        elif pred < 10:
            interp = "Moderate expected usage"
        else:
            interp = "High expected usage (but within expected range)"
    else:
        interp = status

    # Print results
    print("\n-----------------------------")
    print(f"Room: {room}")
    print(f"Predicted electricity usage (kWh): {pred:.2f}")
    print(f"Measured electricity (kWh): {measured_power:.2f}")
    print(f"Difference (measured - predicted): {diff_kw:.2f} kWh ({diff_pct*100:.1f}%)")
    print(f"Result: {interp}")
    if reason:
        print("Reason:", reason)
    print("-----------------------------\n")

    cont = input("Predict/Check another room? (y/n): ").strip().lower()
    if cont != 'y':
        print("Exiting. Good luck with your demo!")
        break
