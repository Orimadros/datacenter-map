import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import shap
import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from collections import Counter
import logging
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def load_data():
    """Load preprocessed São Paulo census tract data and merge with urban/rural classification."""
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Load preprocessed São Paulo census tract data with infrastructure features
    logging.info("Loading São Paulo census tract data with infrastructure features...")
    model_input_path = project_root / 'data' / 'model_input' / 'brasil'
    raw_data_path = project_root / 'data' / 'raw' / 'brasil'
    
    # Load census tracts with infrastructure features
    census_tracts_path = model_input_path / 'sao_paulo_census_tracts_full.csv'
    logging.info(f"Loading census tracts from: {census_tracts_path}")
    
    if not census_tracts_path.exists():
        raise FileNotFoundError(f"Census tract data not found at: {census_tracts_path}")
    
    # Load infrastructure features
    infra_df = pd.read_csv(census_tracts_path)
    # Convert CD_SETOR to string
    infra_df['CD_SETOR'] = infra_df['CD_SETOR'].astype(str)
    logging.info(f"Loaded {len(infra_df)} census tracts with infrastructure features")
    
    # Get list of São Paulo census tract IDs
    sp_tract_ids = infra_df['CD_SETOR'].unique()
    logging.info(f"Found {len(sp_tract_ids)} unique São Paulo census tract IDs")
    
    # Load urban/rural classification from shapefile
    logging.info("Loading urban/rural classification from census tract shapefile...")
    shapefile_path = raw_data_path / 'BR_setores_CD2022' / 'BR_setores_CD2022.shp'
    
    if not shapefile_path.exists():
        raise FileNotFoundError(f"Census tract shapefile not found at: {shapefile_path}")
    
    # Load shapefile and filter for São Paulo census tracts
    classification_df = gpd.read_file(shapefile_path)
    classification_df['CD_SETOR'] = classification_df['CD_SETOR'].astype(str)
    classification_df = classification_df[classification_df['CD_SETOR'].isin(sp_tract_ids)]
    
    # Keep only necessary columns
    classification_cols = [
        'CD_SETOR',
        'CD_SIT',      # Urban/Rural situation code
        'CD_TIPO',     # Special area types
        'AREA_KM2'     # Area in square kilometers
    ]
    classification_df = classification_df[classification_cols]
    
    # Rename columns to match our naming convention
    classification_df = classification_df.rename(columns={
        'CD_SIT': 'CD_SITUACAO'
    })
    
    logging.info(f"Loaded urban/rural classification for {len(classification_df)} São Paulo census tracts")
    
    # Merge infrastructure and classification data
    logging.info("Merging infrastructure and urban/rural classification data...")
    merged_df = pd.merge(
        infra_df,
        classification_df,
        on='CD_SETOR',
        how='left',
        validate='1:1'
    )
    
    # Log merge results
    logging.info(f"Final dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns")
    logging.info("Available features:")
    logging.info(merged_df.columns.tolist())
    
    # Check for any missing values after merge
    check_columns = ['CD_SITUACAO', 'CD_TIPO', 'AREA_KM2']
    missing_counts = merged_df[check_columns].isnull().sum()
    if missing_counts.any():
        logging.warning("Missing values in classification features:")
        logging.warning(missing_counts[missing_counts > 0])
    
    return merged_df

def prepare_features(census_tracts_df):
    """Prepare features for the model from census tract data."""
    logging.info("Preparing features for modeling...")
    
    # Check if required columns exist
    required_columns = [
        'has_data_center',
        # Infrastructure features
        'cell_tower_count',
        'cell_tower_density',
        'min_dist_to_substation_km',
        'min_dist_to_transmission_line_km',
        'viirs_mean',
        # Physical features
        'AREA_KM2',           # Area in square kilometers
        # Urban/Rural characteristics
        'CD_SITUACAO',        # Urban/Rural situation code
        'CD_TIPO'             # Special area types
    ]
    
    missing_columns = [col for col in required_columns if col not in census_tracts_df.columns]
    if missing_columns:
        logging.warning(f"Missing some desired columns: {missing_columns}")
        # Remove missing columns from required_columns
        required_columns = [col for col in required_columns if col not in missing_columns]
        logging.info(f"Proceeding with available columns: {required_columns}")
    
    # Select features (excluding target variable 'has_data_center')
    features = [col for col in required_columns if col != 'has_data_center']
    
    # Create feature matrix and target vector
    X = census_tracts_df[features]
    y = census_tracts_df['has_data_center']
    
    # Handle categorical variables
    categorical_features = ['CD_SITUACAO', 'CD_TIPO']
    available_categorical = [col for col in categorical_features if col in X.columns]
    
    if available_categorical:
        logging.info("Converting categorical variables to one-hot encoding...")
        X = pd.get_dummies(X, columns=available_categorical, prefix=available_categorical)
    
    # Handle any NaN values
    X = X.fillna(0)
    
    # Print class distribution
    logging.info("\nClass distribution in the dataset:")
    logging.info(y.value_counts())
    
    # Print selected features
    logging.info("\nSelected features for modeling:")
    logging.info(X.columns.tolist())
    logging.info(f"\nTotal number of features after one-hot encoding: {X.shape[1]}")
    
    # Print summary statistics for numeric features
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_features) > 0:
        logging.info("\nSummary statistics for numeric features:")
        logging.info(X[numeric_features].describe())
    
    return X, y

def create_model(input_shape):
    """Create the model using Keras functional API."""
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.AUC(name='auc')
        ]
    )
    return model

def train_and_evaluate_model(X, y):
    """Train and evaluate the model using cross-validation with SMOTE."""
    logging.info("Training and evaluating model...")
    
    # Convert to numpy arrays
    X_array = X.values
    y_array = y.values
    
    # Scale features
    scaler = StandardScaler()
    
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
        model = create_model(input_shape=(X.shape[1],))
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
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
        val_metrics = model.evaluate(X_val_scaled, y_val, verbose=0)
        val_loss, val_accuracy, val_auc = val_metrics
        logging.info(f"Fold {fold+1} - Validation Accuracy: {val_accuracy:.4f}, AUC: {val_auc:.4f}")
        
        cv_scores.append(val_accuracy)
        cv_auc_scores.append(val_auc)
        
        # Generate predictions and calculate confusion matrix
        y_pred_proba = model.predict(X_val_scaled, verbose=0)
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
    
    final_model = create_model(input_shape=(X.shape[1],))
    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,
        restore_best_weights=True,
        mode='min'
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
    
    # Save model using the SavedModel format
    model_path = models_dir / 'sao_paulo_dc_model'
    tf.saved_model.save(model, str(model_path))
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
        background = X_sample_scaled[:100]  # Use first 100 samples as background
        explainer = shap.DeepExplainer(model, background)
        shap_values = explainer.shap_values(X_sample_scaled)
        
        # Plot summary
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[0] if isinstance(shap_values, list) else shap_values,
            X_sample,
            feature_names=feature_names,
            show=False
        )
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