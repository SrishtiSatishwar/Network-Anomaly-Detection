import os
import json
from datetime import datetime
from data_generator import NetworkLogGenerator
from anomaly_detector import NetworkAnomalyDetector
from agent import RootCauseAnalysisAgent
from dotenv import load_dotenv
from synthetic_gan_data_generator import SyntheticGANDataGenerator
import numpy as np

def setup_environment():
    """Create necessary directories and load environment variables."""
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load environment variables
    load_dotenv()
    
    # Verify OpenAI API key is set
    if not os.getenv('OPENAI_API_KEY'):
        raise EnvironmentError(
            "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file."
        )

def generate_data():
    """Generate synthetic network log data using GAN."""
    print("\n1. Generating synthetic network data...")
    generator = SyntheticGANDataGenerator()
    real_data = np.random.rand(1000, 10)  # Example real data for training
    generator.train(real_data, epochs=100)
    df = generator.generate_dataset(num_entries=10000)
    df.to_csv('data/network_logs.csv', index=False)
    print(f"Generated {len(df)} log entries")
    return df

def detect_anomalies(df):
    """Detect anomalies in the network data."""
    print("\n2. Detecting anomalies...")
    detector = NetworkAnomalyDetector(contamination=0.05)
    
    # Train the model
    detector.train(df)
    
    # Detect anomalies
    df_with_predictions, anomaly_reports = detector.predict(df)
    
    # Save results
    df_with_predictions.to_csv('data/network_logs_with_predictions.csv', index=False)
    detector.save_model('data/anomaly_detector.joblib')
    
    print(f"Found {len(anomaly_reports)} anomalies")
    return df_with_predictions, anomaly_reports

def analyze_root_causes(df, anomaly_reports):
    """Analyze root causes of detected anomalies."""
    print("\n3. Analyzing root causes...")
    agent = RootCauseAnalysisAgent()
    # Process anomalies in batches of 20
    batch_size = 20
    analyses = []
    for i in range(0, len(anomaly_reports), batch_size):
        batch = anomaly_reports[i:i + batch_size]
        batch_analyses = agent.analyze_anomalies(df, batch)
        analyses.extend(batch_analyses)
        print(f"Processed {i + len(batch)} of {len(anomaly_reports)} anomalies")
    
    # Save detailed analyses
    with open('data/root_cause_analyses.json', 'w') as f:
        json.dump(analyses, f, indent=2, default=str)
    
    # Create summary report
    summary = create_summary_report(analyses)
    with open('data/analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Completed analysis of {len(analyses)} anomalies")
    return analyses

def create_summary_report(analyses):
    """Create a summary report of the analyses."""
    summary = {
        'total_anomalies': len(analyses),
        'timestamp': datetime.now().isoformat(),
        'root_causes': {},
        'avg_confidence': 0,
        'common_remediation_steps': set()
    }
    
    # Collect statistics
    total_confidence = 0
    for analysis in analyses:
        root_cause = analysis['analysis']['root_cause']
        confidence = analysis['analysis']['confidence']
        
        # Count root causes
        if root_cause in summary['root_causes']:
            summary['root_causes'][root_cause]['count'] += 1
            summary['root_causes'][root_cause]['avg_confidence'] += confidence
        else:
            summary['root_causes'][root_cause] = {
                'count': 1,
                'avg_confidence': confidence
            }
        
        # Track total confidence for overall average
        total_confidence += confidence
        
        # Collect unique remediation steps
        summary['common_remediation_steps'].update(
            analysis['analysis']['remediation_steps']
        )
    
    # Calculate averages
    summary['avg_confidence'] = total_confidence / len(analyses)
    
    # Calculate average confidence per root cause
    for cause in summary['root_causes'].values():
        cause['avg_confidence'] = cause['avg_confidence'] / cause['count']
    
    # Convert set to list for JSON serialization
    summary['common_remediation_steps'] = list(summary['common_remediation_steps'])
    
    return summary

def main():
    """Run the complete analysis pipeline."""
    print("Starting network anomaly analysis pipeline...")
    
    try:
        # Setup environment
        setup_environment()
        
        # Generate synthetic data
        df = generate_data()
        
        # Detect anomalies
        df_with_predictions, anomaly_reports = detect_anomalies(df)
        
        # Analyze root causes
        analyses = analyze_root_causes(df_with_predictions, anomaly_reports)
        
        print("\nPipeline completed successfully!")
        print("Results are saved in the 'data' directory:")
        print("- network_logs.csv: Raw network logs")
        print("- network_logs_with_predictions.csv: Logs with anomaly predictions")
        print("- anomaly_detector.joblib: Trained anomaly detection model")
        print("- root_cause_analyses.json: Detailed analysis of each anomaly")
        print("- analysis_summary.json: Summary statistics of findings")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
