"""Quick API test to verify predictions"""
import requests
import json

BASE_URL = 'http://localhost:5000'

# Test 1: Ham email
print("=" * 60)
print("TEST 1: Legitimate Ham Email")
print("=" * 60)
ham_payload = {
    'sender_email': 'alice@enron.com',
    'email_content': 'Quick question about the Q3 report attached. Please review.',
    'use_enhanced': False
}
resp = requests.post(f'{BASE_URL}/api/analyze', json=ham_payload)
data = resp.json()
print(json.dumps(data, indent=2))
print()

# Test 2: Spam email with baseline
print("=" * 60)
print("TEST 2: Spam Email (Baseline Model)")
print("=" * 60)
spam_payload = {
    'sender_email': 'promo123@gmail.com',
    'email_content': 'Congratulations! You won a free prize. Click http://spam.example to claim viagra now.',
    'use_enhanced': False
}
resp = requests.post(f'{BASE_URL}/api/analyze', json=spam_payload)
data = resp.json()
print(json.dumps(data, indent=2))
print()

# Test 3: Spam email with enhanced
print("=" * 60)
print("TEST 3: Spam Email (Enhanced Model with Domain Flag)")
print("=" * 60)
spam_enhanced = {
    'sender_email': 'promo123@gmail.com',
    'email_content': 'Congratulations! You won a free prize. Click http://spam.example to claim viagra now.',
    'use_enhanced': True
}
resp = requests.post(f'{BASE_URL}/api/analyze', json=spam_enhanced)
data = resp.json()
print(json.dumps(data, indent=2))
print()

# Test 4: Metrics endpoint
print("=" * 60)
print("TEST 4: Model Metrics")
print("=" * 60)
resp = requests.get(f'{BASE_URL}/api/metrics')
data = resp.json()
if data.get('success'):
    metrics = data.get('metrics', {})
    enhanced = metrics.get('enhanced', {})
    print(f"Model Accuracy: {enhanced.get('accuracy', 'N/A')*100:.2f}%")
    print(f"Model Precision: {enhanced.get('precision', 'N/A')*100:.2f}%")
    print(f"Model Recall: {enhanced.get('recall', 'N/A')*100:.2f}%")
    print(f"Model F1-Score: {enhanced.get('f1_score', 'N/A')*100:.2f}%")
else:
    print("Error:", data)
