from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# ── Load model & scaler once at startup ─────────────
model  = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')
print("✅ Model and scaler loaded successfully")

# ── Feature order must match train.py ───────────────
FEATURE_ORDER = [
    'tenure',
    'num_of_orders',
    'days_since_last_purchase',
    'total_spent',
    'avg_order_value',
    'num_complaints',
    'discount_used',
    'login_frequency',
    'preferred_category'
]

# ── Routes ───────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Build feature array in correct order
        features = [float(data[f]) for f in FEATURE_ORDER]

        # Scale and predict
        features_scaled = scaler.transform([features])
        probability      = float(model.predict_proba(features_scaled)[0][1])
        prediction       = int(model.predict(features_scaled)[0])
        percent          = round(probability * 100, 1)

        # Risk classification
        if probability >= 0.70:
            risk    = 'High Risk'
            color   = '#EF4444'
            emoji   = '🔴'
            message = 'This customer is highly likely to churn. Take immediate action!'
            action  = 'Send a special discount coupon or exclusive loyalty reward right now.'
        elif probability >= 0.40:
            risk    = 'Medium Risk'
            color   = '#F59E0B'
            emoji   = '🟡'
            message = 'This customer shows moderate churn risk. Monitor closely.'
            action  = 'Send a personalized email with product recommendations this week.'
        else:
            risk    = 'Low Risk'
            color   = '#10B981'
            emoji   = '🟢'
            message = 'This customer is happy and unlikely to churn.'
            action  = 'Continue current engagement. No urgent action needed.'

        return jsonify({
            'probability': percent,
            'prediction':  prediction,
            'risk':        risk,
            'color':       color,
            'emoji':       emoji,
            'message':     message,
            'action':      action
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 45)
    print("  ChurnGuard is running!")
    print(f"  Open  →  http://localhost:{port}")
    print("=" * 45 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)