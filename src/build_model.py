from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam

def build_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=50, return_sequences=True, input_shape=(60, 1))))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(units=50, return_sequences=False)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_absolute_percentage_error')
    return model
