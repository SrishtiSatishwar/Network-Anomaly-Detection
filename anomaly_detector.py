import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

class NetworkAnomalyDetector:
    def __init__(self, contamination: float = 0.05):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: The expected proportion of outliers in the dataset.
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.feature_columns = [
            'bytes_sent', 'bytes_received', 'latency'
        ]
        
    def _preprocess_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the data for anomaly detection.
        
        Args:
            df: Input DataFrame with network log data.
            
        Returns:
            Preprocessed numpy array ready for anomaly detection.
        """
        # Extract numerical features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled
        
    def train(self, df: pd.DataFrame) -> None:
        """
        Train the anomaly detection model.
        
        Args:
            df: Training data DataFrame.
        """
        X = self._preprocess_data(df)
        self.model.fit(X)
        
    def predict(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Detect anomalies in the input data.
        
        Args:
            df: Input DataFrame to analyze.
            
        Returns:
            Tuple containing:
            - DataFrame with anomaly predictions
            - List of dictionaries containing anomaly details
        """
        X = self._preprocess_data(df)
        
        # Get anomaly predictions (-1 for anomalies, 1 for normal)
        predictions = self.model.predict(X)
        
        # Add predictions to DataFrame
        df_with_predictions = df.copy()
        df_with_predictions['is_anomaly'] = predictions == -1
        
        # Get anomaly scores
        anomaly_scores = self.model.score_samples(X)
        df_with_predictions['anomaly_score'] = anomaly_scores
        
        # Create detailed anomaly reports
        anomaly_reports = []
        
        for idx, row in df_with_predictions[df_with_predictions['is_anomaly']].iterrows():
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
        
        return df_with_predictions, anomaly_reports
    
    def save_model(self, path: str) -> None:
        """Save the trained model and scaler."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load a trained model and scaler."""
        saved_objects = joblib.load(path)
        self.model = saved_objects['model']
        self.scaler = saved_objects['scaler']

def main():
    # Load sample data
    print("Loading network logs...")
    df = pd.read_csv('data/network_logs.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Initialize and train detector
    print("Training anomaly detector...")
    detector = NetworkAnomalyDetector(contamination=0.05)
    detector.train(df)
    
    # Detect anomalies
    print("Detecting anomalies...")
    df_with_predictions, anomaly_reports = detector.predict(df)
    
    # Save results
    df_with_predictions.to_csv('data/network_logs_with_predictions.csv', index=False)
    print(f"Found {len(anomaly_reports)} anomalies")
    print("Results saved to data/network_logs_with_predictions.csv")
    
    # Save model
    detector.save_model('data/anomaly_detector.joblib')
    print("Model saved to data/anomaly_detector.joblib")

if __name__ == "__main__":
    main() 