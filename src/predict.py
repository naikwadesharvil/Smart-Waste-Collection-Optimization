
import joblib
import pandas as pd

model = joblib.load('../outputs/model.pkl')

sample = pd.DataFrame({
    'fill_level': [80],
    'weight': [25],
    'days_since_last': [6],
    'temperature': [32]
})

prediction = model.predict(sample)

print("Prediction:", prediction[0])

if prediction[0] == 1:
    print("Send truck")
else:
    print("No collection needed")
