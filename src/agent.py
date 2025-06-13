import os
import json
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini (Google Generative AI)
genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

# Choose model: user can set GEMINI_MODEL; default to a widely available Gemini model
GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'models/gemini-1.5-flash')

# Try to instantiate the requested model; if it fails, fall back to the first available model that supports generate_content.
try:
    model = genai.GenerativeModel(GEMINI_MODEL)
    print(f"Using Gemini model: {GEMINI_MODEL}")
except Exception as e:
    # Fallback: find a model that supports generate_content
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            model = genai.GenerativeModel(m.name)
            print(f"Falling back to Gemini model: {m.name}")
            break
    else:
        raise RuntimeError(f"No Gemini model with generateContent capability available. Original error: {e}")

def clean_json_response(response_text):
    """Clean the response text to ensure it's valid JSON."""
    # Remove markdown code block markers
    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # Remove invalid control characters
    cleaned = ''.join(char for char in cleaned if ord(char) >= 32 or char in '\n\r\t')
    
    return cleaned

class RootCauseAnalysisAgent:
    def __init__(self, analysis_window_hours: int = 24):
        """
        Initialize the root cause analysis agent.
        
        Args:
            analysis_window_hours: Hours of historical context to consider
        """
        self.analysis_window_hours = analysis_window_hours
        
    def _get_historical_context(self, 
                              df: pd.DataFrame, 
                              timestamp: datetime,
                              source_ip: str = None,
                              destination_ip: str = None) -> Dict[str, Any]:
        """Get historical context for the analysis."""
        window_start = timestamp - timedelta(hours=self.analysis_window_hours)
        
        # Filter data within time window
        historical_data = df[
            (df['timestamp'] >= window_start) & 
            (df['timestamp'] <= timestamp)
        ]
        
        # Filter by IPs if provided
        if source_ip:
            historical_data = historical_data[historical_data['source_ip'] == source_ip]
        if destination_ip:
            historical_data = historical_data[historical_data['destination_ip'] == destination_ip]
        
        # Calculate statistics
        stats = {
            'total_events': len(historical_data),
            'success_rate': (historical_data['status'] == 'SUCCESS').mean(),
            'avg_latency': historical_data['latency'].mean(),
            'avg_bytes_sent': historical_data['bytes_sent'].mean(),
            'avg_bytes_received': historical_data['bytes_received'].mean(),
            'common_errors': historical_data['error_code'].value_counts().to_dict(),
            'protocol_distribution': historical_data['protocol'].value_counts().to_dict()
        }
        
        return stats
    
    def analyze_anomaly(self, anomaly: Dict[str, Any], historical_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an anomaly using Gemini's API to determine the root cause.
        
        Args:
            anomaly: Dictionary containing anomaly details
            historical_context: Dictionary containing historical statistics
            
        Returns:
            Dictionary containing the analysis results
        """
        # Construct prompt for Gemini
        prompt = f"""As a network security expert, analyze this network anomaly and determine its root cause.

Anomaly Details:
- Timestamp: {anomaly['timestamp']}
- Anomaly Score: {anomaly['anomaly_score']:.4f}
- Protocol: {anomaly['context']['protocol']}
- Status: {anomaly['context']['status']}
- Error Code: {anomaly['context']['error_code']}
- Source IP: {anomaly['context']['source_ip']}
- Destination IP: {anomaly['context']['destination_ip']}
- Bytes Sent: {anomaly['metrics']['bytes_sent']}
- Bytes Received: {anomaly['metrics']['bytes_received']}
- Latency: {anomaly['metrics']['latency']}ms

Historical Context (Last {self.analysis_window_hours} hours):
- Total Events: {historical_context['total_events']}
- Success Rate: {historical_context['success_rate']:.2%}
- Average Latency: {historical_context['avg_latency']:.2f}ms
- Average Bytes Sent: {historical_context['avg_bytes_sent']:.2f}
- Average Bytes Received: {historical_context['avg_bytes_received']:.2f}

Common Errors: {json.dumps(historical_context['common_errors'], indent=2)}
Protocol Distribution: {json.dumps(historical_context['protocol_distribution'], indent=2)}

Based on this information, please:
1. Identify the most likely root cause of this anomaly
2. Provide a confidence level (0-100%)
3. List supporting evidence
4. Suggest potential remediation steps

Format your response as a JSON object with the following structure:
{{
    "root_cause": "string",
    "confidence": number,
    "evidence": ["string"],
    "remediation_steps": ["string"]
}}"""

        # Call Gemini model without a timeout parameter
        try:
            response = model.generate_content(prompt)
        except Exception as e:
            # Return a structured error so the pipeline continues gracefully
            return {
                "root_cause": f"Error calling Gemini API: {e}",
                "confidence": 0,
                "evidence": [],
                "remediation_steps": []
            }

        # Parse and validate response
        try:
            cleaned_response = clean_json_response(response.text)
            analysis = json.loads(cleaned_response)
            return analysis
        except json.JSONDecodeError as e:
            print(f"Raw response from Gemini API: {response.text}")
            print(f"Cleaned response: {cleaned_response}")
            print(f"JSON parsing error: {e}")
            return {
                "root_cause": "Error parsing model response",
                "confidence": 0,
                "evidence": [],
                "remediation_steps": []
            }

    def analyze_anomalies(self, df: pd.DataFrame, anomaly_reports: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Analyze a list of anomalies and generate root cause analysis for each.
        
        Args:
            df: DataFrame containing all network logs
            anomaly_reports: List of anomaly reports to analyze
            
        Returns:
            List of dictionaries containing analysis results
        """
        analyses = []
        
        for anomaly in anomaly_reports:
            # Get historical context
            historical_context = self._get_historical_context(
                df,
                pd.to_datetime(anomaly['timestamp']),
                anomaly['context']['source_ip'],
                anomaly['context']['destination_ip']
            )
            
            # Analyze anomaly
            analysis = self.analyze_anomaly(anomaly, historical_context)
            
            # Combine anomaly details with analysis
            full_report = {
                'anomaly': anomaly,
                'historical_context': historical_context,
                'analysis': analysis
            }
            
            analyses.append(full_report)
        
        return analyses

def main():
    # Load data with predictions
    print("Loading network logs with predictions...")
    df = pd.read_csv('data/network_logs_with_predictions.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Load anomaly reports
    anomaly_reports = []
    for _, row in df[df['is_anomaly']].iterrows():
        report = {
            'timestamp': row['timestamp'],
            'anomaly_score': row['anomaly_score'],
            'metrics': {
                'bytes_sent': row['bytes_sent'],
                'bytes_received': row['bytes_received'],
                'latency': row['latency']
            },
            'context': {
                'source_ip': row['source_ip'],
                'destination_ip': row['destination_ip'],
                'protocol': row['protocol'],
                'status': row['status'],
                'error_code': row['error_code']
            }
        }
        anomaly_reports.append(report)
    
    # Initialize and run analysis
    print("Analyzing anomalies...")
    agent = RootCauseAnalysisAgent()
    analyses = agent.analyze_anomalies(df, anomaly_reports)
    
    # Save results
    with open('data/root_cause_analyses.json', 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    
    print(f"Analyzed {len(analyses)} anomalies")
    print("Results saved to data/root_cause_analyses.json")

if __name__ == "__main__":
    main() 
