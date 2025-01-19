def get_recommendation(current_price, predicted_price):
    if predicted_price > current_price * 1.05:  # If predicted price is 5% higher than current
        return "Buy"
    elif predicted_price < current_price * 0.95:  # If predicted price is 5% lower than current
        return "Sell"
    else:
        return "Hold"
