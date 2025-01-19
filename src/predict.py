import numpy as np

def predict_next_20_days(scaled_data, model, scaler):
    last_60_days = scaled_data[-60:]
    next_20_days = []
    for _ in range(20):
        pred_input = last_60_days.reshape(1, -1, 1)
        next_pred = model.predict(pred_input)[0, 0]
        next_20_days.append(next_pred)
        last_60_days = np.append(last_60_days[1:], [[next_pred]], axis=0)

    return scaler.inverse_transform(np.array(next_20_days).reshape(-1, 1))
