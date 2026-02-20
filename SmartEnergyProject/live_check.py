# live_check.py
import joblib

# Load model correctly using joblib
model = joblib.load("model.pkl")

def check_room():
    print("\n------ Live Room Electricity Checker ------\n")

    room = int(input("Enter room number (e.g., 118): "))
    area = int(input("Room area in sq.ft (e.g., 200): "))
    apps = int(input("Number of appliances (e.g., 3): "))
    temp = float(input("Temperature now (°C, e.g., 25): "))
    hour = int(input("Hour of day (0–23): "))

    # Prepare features for prediction
    features = [[room, area, apps, temp, hour]]

    pred = model.predict(features)[0]

    print("\n-------------------------------")
    print(f"Room: {room}")
    print(f"Predicted electricity usage (kWh): {pred:.2f}")

    # Interpretation / Meaning
    if pred < 6:
        meaning = "Low usage → Efficient."
    elif pred < 10:
        meaning = "Moderate usage → Normal."
    else:
        meaning = "High usage → Possible wastage ⚠️"

    print(f"Interpretation: {meaning}")
    print("-------------------------------\n")

# Loop so you can check multiple rooms
while True:
    check_room()
    again = input("Check another room? (y/n): ").strip().lower()
    if again != "y":
        print("Exiting. Good luck with your demo!")
        break
