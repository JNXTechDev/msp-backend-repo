import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import detector

examples = [
    {
        'sender': 'alice@enron.com',
        'content': 'Quick question about the Q3 report attached. Please review.'
    },
    {
        'sender': 'promo123@gmail.com',
        'content': 'Congratulations! You won a free prize. Click http://spam.example to claim viagra now.'
    }
]

for ex in examples:
    out = detector.predict(ex['sender'], ex['content'], use_enhanced=True)
    print('---')
    print('Sender:', ex['sender'])
    print('Content (truncated):', ex['content'][:120])
    print('Prediction:', out['prediction'])
    print('Confidence:', out['confidence'])
    print('Domain flag:', out['domain_flag'])
    print('Sender domain:', out['sender_domain'])
