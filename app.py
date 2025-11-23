from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import re
import email
from email import policy
from email.parser import BytesParser
import joblib
import glob
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

# List of free/public email domains
FREE_DOMAINS = [
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'aol.com',
    'icloud.com', 'mail.com', 'protonmail.com', 'zoho.com', 'yandex.com',
    'live.com', 'msn.com', 'inbox.com', 'gmx.com', 'fastmail.com'
]

class SpamDetector:
    def __init__(self, models_dir="models"):
        """Try to load pre-trained artifacts from `models_dir`. If they are
        missing fall back to the dummy models currently used in development.
        """
        self.models_dir = os.path.abspath(models_dir)
        self.is_trained = False
        self.meta_columns = []

        # Placeholders; will be set either by loading artifacts or by training dummy
        self.vectorizer = None
        self.model = None

        # Attempt to load saved artifacts (prefer sklearn Pipelines)
        try:
            model_path = os.path.join(self.models_dir, 'model.pkl')
            meta_path = os.path.join(self.models_dir, 'meta_config.pkl')

            def _find_latest(prefix):
                pattern = os.path.join(self.models_dir, f"{prefix}*.pkl")
                candidates = glob.glob(pattern)
                if not candidates:
                    return None
                calibrated = [c for c in candidates if 'calibrat' in os.path.basename(c).lower()]
                if calibrated:
                    return max(calibrated, key=os.path.getmtime)
                return max(candidates, key=os.path.getmtime)

            loaded_any = False
            try:
                enhanced_candidate = _find_latest('enhanced_pipeline')
                baseline_candidate = _find_latest('baseline_pipeline')

                if enhanced_candidate:
                    try:
                        self.pipeline_enhanced = joblib.load(enhanced_candidate)
                        print(f"Loaded enhanced pipeline from {enhanced_candidate} (type={type(self.pipeline_enhanced)})")
                        loaded_any = True
                    except Exception as e:
                        print('Failed loading enhanced candidate', enhanced_candidate, e)

                if baseline_candidate:
                    try:
                        self.pipeline_baseline = joblib.load(baseline_candidate)
                        print(f"Loaded baseline pipeline from {baseline_candidate} (type={type(self.pipeline_baseline)})")
                        loaded_any = True
                    except Exception as e:
                        print('Failed loading baseline candidate', baseline_candidate, e)

                # fallback: model.pkl might contain a pipeline (older experiments)
                if not loaded_any and os.path.exists(model_path):
                    try:
                        maybe = joblib.load(model_path)
                        if hasattr(maybe, 'predict'):
                            self.pipeline_enhanced = maybe
                            print(f"Loaded legacy model from {model_path} (type={type(maybe)})")
                            loaded_any = True
                    except Exception as e:
                        print('Failed loading model.pkl', e)

                # load meta config if present (best-effort)
                if os.path.exists(meta_path):
                    try:
                        meta = joblib.load(meta_path)
                        if isinstance(meta, dict):
                            self.meta_columns = meta.get('meta_columns', [meta.get('domain_feature')] if meta.get('domain_feature') else [])
                    except Exception as e:
                        print('Failed loading meta_config.pkl', e)

                if loaded_any:
                    self.is_trained = True
                    print(f"Loaded pipeline artifacts from {self.models_dir}")
                else:
                    print("Model artifacts not found; training dummy models as fallback")
                    self._train_dummy_model()
            except Exception as e:
                print('Error while loading artifacts:', e)
                self._train_dummy_model()
        except Exception as e:
            # if loading fails, use dummy models but log the error
            print("Failed to load artifacts:", e)
            self._train_dummy_model()
    
    def extract_sender_domain(self, email_address):
        """Extract domain from email address"""
        if not email_address:
            return None
        
        # Clean email address
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', email_address)
        if email_match:
            email_clean = email_match.group(0)
            domain = email_clean.split('@')[-1].lower()
            return domain
        return None
    
    def is_free_domain(self, domain):
        """Check if domain is a free/public domain"""
        if not domain:
            return False
        return domain.lower() in FREE_DOMAINS
    
    def preprocess_text(self, text):
        """Preprocess email text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def predict(self, sender_email, email_content, use_enhanced=False):
        """
        Predict if email is spam or ham
        
        Returns:
            dict: {
                'prediction': 'spam' or 'ham',
                'confidence': float,
                'domain_flag': bool,
                'sender_domain': str
            }
        """
        if not self.is_trained:
            raise Exception("Model not trained")
        
        # Extract domain and flag
        domain = self.extract_sender_domain(sender_email)
        domain_flag = 1 if self.is_free_domain(domain) else 0

        # Preprocess content
        processed_content = self.preprocess_text(email_content)

        # If pipelines are loaded, use them (preferred)
        if self.is_trained and (getattr(self, 'pipeline_enhanced', None) is not None or getattr(self, 'pipeline_baseline', None) is not None):
            # Baseline prediction (text-only)
            if getattr(self, 'pipeline_baseline', None) is not None:
                try:
                    pred_b = int(self.pipeline_baseline.predict([processed_content])[0])
                    # prefer calibrated probability if available
                    if hasattr(self.pipeline_baseline, 'predict_proba'):
                        try:
                            conf_b = float(self.pipeline_baseline.predict_proba([processed_content])[0][1])
                        except Exception:
                            try:
                                dec_b = float(self.pipeline_baseline.decision_function([processed_content])[0])
                            except Exception:
                                dec_b = 0.0
                            conf_b = 1 / (1 + np.exp(-dec_b))
                    else:
                        try:
                            dec_b = float(self.pipeline_baseline.decision_function([processed_content])[0])
                        except Exception:
                            dec_b = 0.0
                        conf_b = 1 / (1 + np.exp(-dec_b))
                except Exception:
                    pred_b = 0
                    conf_b = 0.0
            else:
                pred_b = 0
                conf_b = 0.0

            # Enhanced prediction (text + meta)
            if getattr(self, 'pipeline_enhanced', None) is not None:
                try:
                    df_row = pd.DataFrame({'email_content':[processed_content], 'domain_flag':[domain_flag]})
                    pred_e = int(self.pipeline_enhanced.predict(df_row)[0])
                    # prefer predict_proba for calibrated pipeline
                    if hasattr(self.pipeline_enhanced, 'predict_proba'):
                        try:
                            conf_e = float(self.pipeline_enhanced.predict_proba(df_row)[0][1])
                        except Exception:
                            try:
                                dec_e = float(self.pipeline_enhanced.decision_function(df_row)[0])
                            except Exception:
                                dec_e = 0.0
                            conf_e = 1 / (1 + np.exp(-dec_e))
                    else:
                        try:
                            dec_e = float(self.pipeline_enhanced.decision_function(df_row)[0])
                        except Exception:
                            dec_e = 0.0
                        conf_e = 1 / (1 + np.exp(-dec_e))
                except Exception:
                    pred_e = pred_b
                    conf_e = conf_b
            else:
                pred_e = pred_b
                conf_e = conf_b

            if use_enhanced:
                prediction = pred_e
                confidence = conf_e
            else:
                prediction = pred_b
                confidence = conf_b
        else:
            # fallback to dummy models (text-only)
            X_vec = self.baseline_vectorizer.transform([processed_content])
            prediction = int(self.baseline_model.predict(X_vec)[0])
            decision = float(self.baseline_model.decision_function(X_vec)[0])
            confidence = 1 / (1 + np.exp(-decision))
        
        # Convert returned confidence to be the probability of the predicted class
        # (previously we returned the spam probability regardless of prediction,
        # which made ham predictions look like very low-confidence when spam
        # probability was small). Present the class-matched probability here.
        if prediction == 1:
            conf_for_label = float(confidence)
        else:
            conf_for_label = float(1.0 - confidence)

        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': conf_for_label,
            'domain_flag': bool(domain_flag),
            'sender_domain': domain or 'unknown'
        }

# Initialize detector
detector = SpamDetector()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyze email for spam detection"""
    try:
        data = request.get_json()
        
        sender_email = data.get('sender_email', '')
        email_content = data.get('email_content', '')
        raw_headers = data.get('raw_headers', '')
        use_enhanced = data.get('use_enhanced', False)
        
        # If raw headers provided, try to parse
        if raw_headers:
            try:
                # Parse email headers
                msg = email.message_from_string(raw_headers)
                sender_email = msg.get('From', sender_email)
                subject = msg.get('Subject', '')
                email_content = f"{subject} {email_content}"
            except:
                pass
        
        # Validate inputs
        if not sender_email or not email_content:
            return jsonify({
                'error': 'Both sender email and email content are required'
            }), 400
        
        # Get predictions from both models
        baseline_result = detector.predict(sender_email, email_content, use_enhanced=False)
        enhanced_result = detector.predict(sender_email, email_content, use_enhanced=True)
        
        return jsonify({
            'success': True,
            'baseline': {
                'prediction': baseline_result['prediction'],
                'confidence': round(baseline_result['confidence'] * 100, 2),
                'model_name': 'Baseline (TF-IDF + Linear SVM)'
            },
            'enhanced': {
                'prediction': enhanced_result['prediction'],
                'confidence': round(enhanced_result['confidence'] * 100, 2),
                'model_name': 'Enhanced (+Domain Flag)'
            },
            'metadata': {
                'sender_domain': baseline_result['sender_domain'],
                'domain_flag': baseline_result['domain_flag'],
                'domain_type': 'Free/Public Domain' if baseline_result['domain_flag'] else 'Custom/Corporate Domain'
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze-file', methods=['POST'])
def analyze_file():
    """Analyze .eml file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not file.filename.lower().endswith('.eml'):
            return jsonify({'error': 'Only .eml files are supported'}), 400
        
        # Parse .eml file
        msg = BytesParser(policy=policy.default).parsebytes(file.read())
        
        sender_email = msg.get('From', '')
        subject = msg.get('Subject', '')
        
        # Get email body
        body = ''
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_content()
                    break
        else:
            body = msg.get_content()
        
        email_content = f"{subject} {body}"
        use_enhanced = request.form.get('use_enhanced', 'false').lower() == 'true'
        
        # Get predictions
        baseline_result = detector.predict(sender_email, email_content, use_enhanced=False)
        enhanced_result = detector.predict(sender_email, email_content, use_enhanced=True)
        
        return jsonify({
            'success': True,
            'baseline': {
                'prediction': baseline_result['prediction'],
                'confidence': round(baseline_result['confidence'] * 100, 2),
                'model_name': 'Baseline (TF-IDF + Linear SVM)'
            },
            'enhanced': {
                'prediction': enhanced_result['prediction'],
                'confidence': round(enhanced_result['confidence'] * 100, 2),
                'model_name': 'Enhanced (+Domain Flag)'
            },
            'metadata': {
                'sender_domain': baseline_result['sender_domain'],
                'domain_flag': baseline_result['domain_flag'],
                'domain_type': 'Free/Public Domain' if baseline_result['domain_flag'] else 'Custom/Corporate Domain'
            },
            'parsed_data': {
                'sender': sender_email,
                'subject': subject
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def metrics():
    """Return model performance metrics from verified evaluation"""
    try:
        metrics_path = os.path.join(os.path.dirname(__file__), 'experiments_out', 'verified_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as fh:
                verified_metrics = json.load(fh)
            return jsonify({
                'success': True,
                'metrics': verified_metrics
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Verified metrics file not found. Please run train_verified.py first.'
            }), 404
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_trained': detector.is_trained
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
