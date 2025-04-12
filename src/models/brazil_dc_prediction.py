import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import shap
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_data():
    """Load preprocessed São Paulo census tract data."""
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load São Paulo census tract data
    logging.info("Loading São Paulo census tract data...")
    model_input_path = project_root / 'data' / 'model_input' / 'brasil'
    
    # Load census tracts with features
    census_tracts_path = model_input_path / 'sao_paulo_census_tracts_full.csv'
    logging.info(f"Loading census tracts from: {census_tracts_path}")
    
    if not census_tracts_path.exists():
        raise FileNotFoundError(f"Census tract data not found at: {census_tracts_path}")
    
    census_tracts_df = pd.read_csv(census_tracts_path)
    logging.info(f"Loaded {len(census_tracts_df)} census tracts")
    
    return census_tracts_df

def prepare_features(census_tracts_df):
    """Prepare features for the model from census tract data."""
    logging.info("Preparing features for modeling...")
    
    # Check if required columns exist
    required_columns = [
        'has_data_center', 
        'cell_tower_count', 
        'cell_tower_density',
        'min_dist_to_substation_km', 
        'min_dist_to_transmission_line_km',
        'viirs_mean'
    ]
    
    missing_columns = [col for col in required_columns if col not in census_tracts_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in census tract data: {missing_columns}")
    
    # Select features and target
    features = [
        'cell_tower_count',
        'cell_tower_density',
        'min_dist_to_substation_km',
        'min_dist_to_transmission_line_km',
        'viirs_mean'
    ]
    
    # Create feature matrix and target vector
    X = census_tracts_df[features]
    y = census_tracts_df['has_data_center']
    
    # Handle any NaN values
    X = X.fillna(0)
    
    # Print class distribution
    logging.info("\nClass distribution in the dataset:")
    logging.info(y.value_counts())
    
    return X, y

def train_and_evaluate_model(X, y):
    """Train and evaluate the model using cross-validation with SMOTE."""
    logging.info("Training and evaluating model...")
    
    # Convert DataFrames to numpy arrays
    X_array = X.values
    y_array = y.values
    
    # Scale features
    scaler = StandardScaler()
    
    # Define model architecture
    def create_model():
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
    
    # Cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    cv_auc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_array)):
        logging.info(f"Training fold {fold+1}/5...")
        
        X_train, X_val = X_array[train_idx], X_array[val_idx]
        y_train, y_val = y_array[train_idx], y_array[val_idx]
        
        # Print class distribution before SMOTE
        logging.info(f"Class distribution before SMOTE: {Counter(y_train)}")
        
        # Apply SMOTE for the training data
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        
        # Print class distribution after SMOTE
        logging.info(f"Class distribution after SMOTE: {Counter(y_train_resampled)}")
        
        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train_resampled)
        X_val_scaled = scaler.transform(X_val)
        
        # Create and train model
        model = create_model()
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        history = model.fit(
            X_train_scaled, y_train_resampled,
            validation_data=(X_val_scaled, y_val),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy, val_auc = model.evaluate(X_val_scaled, y_val, verbose=0)
        logging.info(f"Fold {fold+1} - Validation Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        
        cv_scores.append(val_accuracy)
        cv_auc_scores.append(val_auc)
        
        # Generate predictions and calculate confusion matrix
        y_pred_proba = model.predict(X_val_scaled)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_val, y_pred))
        
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_val, y_pred))
    
    logging.info(f"Cross-validation accuracy: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
    logging.info(f"Cross-validation AUC: {np.mean(cv_auc_scores):.4f} ± {np.std(cv_auc_scores):.4f}")
    
    # Train final model on all data with SMOTE
    logging.info("\nTraining final model on all data...")
    X_resampled, y_resampled = SMOTE(random_state=42).fit_resample(X_array, y_array)
    X_scaled = scaler.fit_transform(X_resampled)
    
    final_model = create_model()
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True
    )
    
    final_model.fit(
        X_scaled, y_resampled,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    return final_model, scaler

def save_model_artifacts(model, scaler, X, feature_names):
    """Save model and related artifacts."""
    project_root = Path(__file__).resolve().parent.parent.parent
    models_dir = project_root / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = models_dir / 'sao_paulo_dc_model.h5'
    model.save(model_path)
    logging.info(f"Model saved to {model_path}")
    
    # Save scaler
    scaler_path = models_dir / 'sao_paulo_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")
    
    # Generate and save SHAP values/explanations
    logging.info("Generating SHAP explanations...")
    
    try:
        # Scale a sample of data for SHAP
        X_sample = X.sample(min(1000, len(X)), random_state=42)
        X_sample_scaled = scaler.transform(X_sample)
        
        # Create explainer
        explainer = shap.DeepExplainer(model, X_sample_scaled)
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # Plot summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values[0], X_sample, feature_names=feature_names, show=False)
        plt.tight_layout()
        
        # Save plot
        shap_plot_path = project_root / 'outputs' / 'figures' / 'sao_paulo_shap_summary.png'
        plt.savefig(shap_plot_path, dpi=300, bbox_inches='tight')
        logging.info(f"SHAP summary plot saved to {shap_plot_path}")
        plt.close()
        
    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {e}")

def main():
    """Main function to run the São Paulo data center prediction pipeline."""
    logging.info("Starting São Paulo data center prediction pipeline...")
    
    # Load data
    census_tracts_df = load_data()
    
    # Prepare features
    X, y = prepare_features(census_tracts_df)
    
    # Train and evaluate model
    model, scaler = train_and_evaluate_model(X, y)
    
    # Save model artifacts
    save_model_artifacts(model, scaler, X, X.columns.tolist())
    
    logging.info("São Paulo data center prediction pipeline completed successfully!")

if __name__ == "__main__":
    main() 