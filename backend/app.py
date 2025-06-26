from flask import Flask, request, jsonify, session
from flask import Flask, request, jsonify, session, send_from_directory, send_file
from flask_cors import CORS
import os
import pandas as pd
import traceback
import re
from pipeline import run_pipeline, LLMConfig, PipelineConfig, DataAnalysisPipeline
from datasets.data import DataReader
import logging
import json
import uuid
import math
import numpy as np

# Import the new data structures from the agent script
from data_aggregation_agent import StrategicRecommendation, Section


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24).hex())

# Global store for active sessions
active_sessions = {}


def get_or_create_session(session_id=None):
    """Get existing session or create a new one"""
    if session_id and session_id in active_sessions:
        return session_id, active_sessions[session_id]
    
    new_session_id = str(uuid.uuid4())
    active_sessions[new_session_id] = {
        'dataset': None,
        'pipeline': None,
        'recommendations': [],
        'chat_history': {}
    }
    return new_session_id, active_sessions[new_session_id]



# Add this after creating your Flask app
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build')
    if path != "" and os.path.exists(os.path.join(static_folder, path)):
        return send_from_directory(static_folder, path)
    else:
        return send_from_directory(static_folder, 'index.html')

# Update CORS to allow your domain
CORS(app, supports_credentials=True, origins=["*"])  # Update this later with your actual domain

# Configure upload folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATASET_FOLDER = os.path.join(BASE_DIR, 'datasets')
AGGREGATION_FOLDER = os.path.join(BASE_DIR, 'strategic_recommendations')

# Global store for active sessions
active_sessions = {}

class CustomDataReader(DataReader):
    """Extended DataReader class for custom datasets"""
    def __init__(self, description, features_description, data=None):
        self.description = description
        self.features_description = features_description
        self.train_base = data
        self.train_input = data
        if data is not None:
            self.chosen_features = data.columns.tolist()

def ensure_directories_exist():
    """Ensure all required directories exist"""
    directories = [UPLOAD_FOLDER, DATASET_FOLDER, AGGREGATION_FOLDER]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def parse_column_descriptions(data: str) -> dict:
    """Parse column descriptions from string format to dictionary."""
    descriptions = {}
    if not data:
        return descriptions
    for line in data.strip().split('\n'):
        if ':' in line:
            col, desc = line.split(':', 1)
            descriptions[col.strip()] = desc.strip()
    return descriptions

# --- FINAL FIX: Add handling for numpy.ndarray ---
def clean_for_json(data):
    """
    Recursively cleans a data structure to make it JSON serializable.
    Converts special float values, numpy types, and other non-standard types.
    """
    if isinstance(data, dict):
        return {str(k): clean_for_json(v) for k, v in data.items()}
    
    if isinstance(data, (list, tuple)):
        return [clean_for_json(v) for v in data]
    
    # Handle numpy arrays by converting them to lists
    if isinstance(data, np.ndarray):
        return clean_for_json(data.tolist())
    
    # Handle pandas Series
    if isinstance(data, pd.Series):
        return clean_for_json(data.tolist())
    
    # Handle pandas DataFrames
    if isinstance(data, pd.DataFrame):
        return clean_for_json(data.to_dict(orient='records'))
    
    # Handle special float values
    if isinstance(data, (float, np.float64, np.float32)):
        if math.isnan(data) or math.isinf(data):
            return None
        return float(data)
    
    # Handle numpy integers
    if isinstance(data, (np.integer, np.int64, np.int32)):
        return int(data)
    
    # Handle numpy booleans
    if isinstance(data, np.bool_):
        return bool(data)
    
    # Handle pandas Timestamps and datetime objects
    if isinstance(data, (pd.Timestamp, pd.DatetimeIndex)):
        return data.isoformat() if hasattr(data, 'isoformat') else str(data)
    
    # Handle datetime objects
    if hasattr(data, 'isoformat'):
        return data.isoformat()
    
    # Handle any other numpy types
    if hasattr(data, 'item'):
        return clean_for_json(data.item())
    
    # Handle bytes
    if isinstance(data, bytes):
        return data.decode('utf-8', errors='ignore')
    
    # Convert any remaining complex objects to string
    if hasattr(data, '__dict__') and not isinstance(data, (str, int, float, bool)):
        return str(data)
    
    return data

def format_recommendation_for_json(rec: StrategicRecommendation) -> dict:
    """Converts a StrategicRecommendation object into a standard dictionary."""
    def format_section(section: Section) -> dict:
        table_data = None
        if section and section.table is not None and not section.table.empty:
            # The conversion to dict handles most types, final cleaning will catch the rest
            table_data = section.table.to_dict(orient='records')
        
        return {
            "text": section.text if section else "",
            "table_data": table_data
        }

    return {
        'id': rec.id,
        'title': rec.title,
        'finding': format_section(rec.finding),
        'action_logic': format_section(rec.action_logic),
        'feasibility': format_section(rec.feasibility),
        'effect': format_section(rec.effect),
        'generated_from_feedback': rec.generated_from_feedback,
        'chat': []
    }

@app.route('/api/analyze', methods=['GET', 'HEAD', 'POST'])
def analyze_data():
    """Handle data analysis requests."""
    if request.method in ['GET', 'HEAD']:
        return jsonify({"status": "ok", "message": "Backend is running."})
        
    try:
        ensure_directories_exist()
        
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '' or not file.filename.endswith('.csv'):
            return jsonify({'error': 'A valid CSV file is required'}), 400

        dataset_description = request.form.get('description')
        columns_description = request.form.get('columns')
        
        num_recommendations_str = request.form.get('iterations', '2')
        try:
            num_recs = int(num_recommendations_str)
        except ValueError:
            num_recs = 2
        
        if not dataset_description or not columns_description:
            return jsonify({'error': 'Dataset and columns descriptions are required'}), 400

        upload_path = os.path.join(UPLOAD_FOLDER, 'data.csv')
        file.save(upload_path)
        
        data = pd.read_csv(upload_path)
        dataset = CustomDataReader(
            description=dataset_description,
            features_description=parse_column_descriptions(columns_description),
            data=data
        )
        
        llm_model = LLMConfig(platform='anthropic', model_name="claude-3-haiku-20240307")

        config_params = {
            'dc_missing_threshold': 1,
            'dc_outlier_method': 'iqr',
            'dc_outlier_threshold': 1.5,
            'dc_handle_categorical': True,
            'da_num_recommendations': num_recs,
            'llm_config': llm_model
        }
        config = PipelineConfig(**config_params)
        pipeline = DataAnalysisPipeline(dataset, config)
        recommendations = pipeline.run()
        
        full_results = []
        for rec in recommendations:
            if isinstance(rec, StrategicRecommendation):
                full_results.append(format_recommendation_for_json(rec))

        session_id, session_data = get_or_create_session()
        session_data['dataset'] = dataset
        session_data['pipeline'] = pipeline
        session_data['recommendations'] = recommendations
        
        final_payload = {
            'session_id': session_id,
            'recommendations': full_results
        }
        cleaned_payload = clean_for_json(final_payload)
        
        return jsonify(cleaned_payload)
            
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            'error': f'Analysis pipeline failed: {str(e)}',
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/generate-from-feedback', methods=['POST'])
def generate_from_feedback():
    """Generate a new recommendation based on user feedback"""
    try:
        data = request.json
        session_id = data.get('session_id')
        feedback = data.get('feedback')
        source_recommendation_id = data.get('recommendation_id')
        
        if not session_id or not feedback or not source_recommendation_id:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        if session_id not in active_sessions:
            return jsonify({'error': 'Session not found'}), 404
        
        session_data = active_sessions[session_id]
        pipeline = session_data.get('pipeline')
        
        if not pipeline:
            return jsonify({'error': 'No active pipeline found for this session'}), 404
        
        new_recommendation = pipeline.da_agent.generate_recommendation_from_feedback(
            feedback_text=feedback,
            source_recommendation_id=source_recommendation_id
        )
            
        if not new_recommendation:
            return jsonify({'error': 'Failed to generate a new recommendation from feedback'}), 500
        
        session_data['recommendations'].append(new_recommendation)
        
        if 'chat_history' not in session_data:
            session_data['chat_history'] = {}
        if str(source_recommendation_id) not in session_data['chat_history']:
            session_data['chat_history'][str(source_recommendation_id)] = []
        
        chat_entry = {
            'user': feedback,
            'assistant': f"I've generated a new recommendation based on your feedback. You can find it titled: '{new_recommendation.title}'.",
            'timestamp': pd.Timestamp.now().isoformat()
        }
        session_data['chat_history'][str(source_recommendation_id)].append(chat_entry)
        
        formatted_result = format_recommendation_for_json(new_recommendation)
        
        final_payload = {
            'new_recommendation': formatted_result,
            'chat_history_update': {
                'recommendation_id': source_recommendation_id,
                'history': session_data['chat_history'][str(source_recommendation_id)]
            }
        }
        cleaned_payload = clean_for_json(final_payload)
            
        return jsonify(cleaned_payload)
            
    except Exception as e:
        logger.error(f"Error generating from feedback: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# Other routes can remain the same
@app.route('/api/chat-history/<session_id>/<recommendation_id>', methods=['GET'])
def get_chat_history(session_id, recommendation_id):
    if session_id not in active_sessions:
        return jsonify({'error': 'Session not found'}), 404
    session_data = active_sessions[session_id]
    chat_history = session_data.get('chat_history', {}).get(recommendation_id, [])
    return jsonify({'chat_history': clean_for_json(chat_history)})

@app.route('/api/session', methods=['GET'])
def list_sessions():
    session_list = []
    for session_id, data in active_sessions.items():
        if data.get('dataset'):
            session_list.append({
                'id': session_id,
                'num_recommendations': len(data.get('recommendations', [])),
                'description': data.get('dataset').description if data.get('dataset') else 'No description'
            })
    return jsonify({'sessions': clean_for_json(session_list)})

@app.route('/api/session/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    if session_id in active_sessions:
        del active_sessions[session_id]
        return jsonify({'status': 'success', 'message': 'Session deleted'})
    else:
        return jsonify({'error': 'Session not found'}), 404

if __name__ == '__main__':
    ensure_directories_exist()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8000)), debug=True)