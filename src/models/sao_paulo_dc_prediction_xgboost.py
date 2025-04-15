import logging
import pandas as pd
import geopandas as gpd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
# from sklearn.utils.class_weight import compute_sample_weight # Not strictly needed if using scale_pos_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import sys

# Add project root to Python path to allow importing from 'src'
root_dir = str(Path(__file__).resolve().parent.parent.parent)
if root_dir not in sys.path:
    sys.path.append(root_dir)

from src.visualization.plot_heatmap import plot_probability_heatmap

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_project_root() -> Path:
    """Find the project root based on a marker file or directory."""
    # Using the current file's location to find the root
    current_file_path = Path(__file__).resolve()
    # Iterate upwards to find the root marker
    for parent in current_file_path.parents:
        if (parent / 'README.md').exists() or (parent / '.git').exists() or (parent / 'setup.py').exists():
            logging.info(f"Project root identified as: {parent}")
            return parent
    # Fallback if no marker found
    logging.warning("Could not automatically determine project root via markers. Defaulting to 2 levels up.")
    # Default assumption if specific markers aren't found
    project_root = current_file_path.parent.parent.parent
    if (project_root / 'src').is_dir() and (project_root / 'data').is_dir():
         logging.info(f"Using default project root: {project_root}")
         return project_root
    else:
         logging.error("Project root structure not as expected. Defaulting to CWD, paths might be incorrect.")
         return Path.cwd() # Or raise an exception

def load_data(file_path: Path) -> gpd.GeoDataFrame:
    """Load the GeoJSON data."""
    logging.info(f"Loading data from {file_path}...")
    if not file_path.exists():
        logging.error(f"Data file not found at: {file_path}")
        sys.exit(1)
    try:
        gdf = gpd.read_file(file_path)
        logging.info(f"Data loaded successfully. Shape: {gdf.shape}")
        # Basic validation
        required_cols = ['has_data_center', 'geometry', 'CD_SETOR'] # Add other essential feature cols
        missing = [col for col in required_cols if col not in gdf.columns]
        if missing:
            # Allow missing 'geometry' if it's not strictly needed downstream for features,
            # but log a warning. Let features check handle specific needs.
            if 'geometry' in missing and not any(f in missing for f in ['has_data_center', 'CD_SETOR']):
                 logging.warning("Missing 'geometry' column, proceeding without it for feature selection.")
            else:
                 raise ValueError(f"Missing required columns for basic structure or target: {missing}")
        # Check if it's a GeoDataFrame only if geometry is present
        if 'geometry' in gdf.columns and not isinstance(gdf, gpd.GeoDataFrame):
             raise TypeError("Input data should be a GeoDataFrame if 'geometry' column is present.")
        return gdf
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1) # Exit if data loading fails

def save_artifacts(model: xgb.XGBClassifier, scaler: StandardScaler, output_dir: Path):
    """Save the trained model and scaler."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'xgboost_model.joblib'
    scaler_path = output_dir / 'scaler.joblib'
    try:
        joblib.dump(model, model_path)
        logging.info(f"Model saved successfully to {model_path}")
        joblib.dump(scaler, scaler_path)
        logging.info(f"Scaler saved successfully to {scaler_path}")
    except Exception as e:
        logging.error(f"Error saving artifacts: {e}")

def train_and_evaluate_xgb(df: pd.DataFrame, output_dir: Path):
    """
    Train an XGBoost model on the provided DataFrame (df) using specified features and target.
    Implements stratified train-test split, scaling, and imbalance adjustment via scale_pos_weight.
    Evaluates the model and saves plots and artifacts.

    Returns:
        model: Trained XGBoost classifier.
        scaler: Fitted StandardScaler object.
        X_test_scaled: Scaled DataFrame of test features.
        y_test: Series of test target.
    """
    # Define features and target.
    # Make sure these columns exist in your dataframe.
    features = [
        'area_km2',
        'cell_tower_count',
        'cell_tower_density',
        'min_dist_to_substation_km',
        'min_dist_to_transmission_line_km',
        'min_dist_to_hyperscaler_km', # Added back
        'viirs_mean',
        'CD_SIT'
    ]
    target = 'has_data_center'

    # Basic check for features and target
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        logging.error(f"Missing feature columns in input data: {missing_features}")
        sys.exit(1)
    if target not in df.columns:
        logging.error(f"Target column '{target}' not found in input data.")
        sys.exit(1)

    # One-hot encode categorical feature 'CD_SIT'
    if 'CD_SIT' in features:
        logging.info("Applying one-hot encoding on 'CD_SIT'...")
        df = pd.get_dummies(df, columns=['CD_SIT'], prefix='SIT', drop_first=True)
        # Update features: remove original and add the new one-hot columns
        features.remove('CD_SIT')
        new_cat_cols = [col for col in df.columns if col.startswith('SIT_')]
        features.extend(new_cat_cols)
        logging.info(f"Features after one-hot encoding: {features}")

    # Separate features and target.
    X = df[features]
    y = df[target]

    logging.info(f"Final features selected: {X.columns.tolist()}")
    logging.info(f"Target variable: {target}")
    logging.info("Target distribution:")
    logging.info(y.value_counts(normalize=True))

    # Check for missing values. If any are found, log and exit.
    if X.isnull().sum().any():
        logging.warning("NaN values found in the following features:")
        logging.warning(X.isnull().sum()[X.isnull().sum() > 0])
        logging.error("Execution stopped: NaN values detected in features. Please clean the data first.")
        sys.exit(1) # Exit script if NaNs are present
    # else: # If no NaNs, proceed (this else is implicit)
    #    pass 

    logging.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y
    )

    logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    logging.info(f"Training target distribution:\n{y_train.value_counts(normalize=True)}")
    logging.info(f"Test target distribution:\n{y_test.value_counts(normalize=True)}")


    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    if pos_count == 0:
        logging.warning("No positive samples in training set; setting scale_pos_weight to 1.")
        scale_pos_weight = 1
    else:
        scale_pos_weight = neg_count / pos_count
    logging.info(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Scale features
    logging.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Convert back to DataFrame for feature names in importance plot
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)


    # Initialize and train XGBoost classifier.
    logging.info("Initializing and training XGBoost classifier with adjusted hyperparameters...")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss', # Used for potential internal evaluation/logging if verbose=True
        use_label_encoder=False, # Deprecated, recommended to set False
        scale_pos_weight=scale_pos_weight,
        # --- Adjusted Hyperparameters ---
        n_estimators=300,      # Reduced from 400
        learning_rate=0.05,    # Kept
        max_depth=4,           # Reduced from 6
        subsample=0.8,         # Kept
        colsample_bytree=0.8,  # Kept
        gamma=0.1,             # Kept
        reg_alpha=0.1,         # Added L1 regularization
        reg_lambda=1,          # Added L2 regularization
        early_stopping_rounds=50 # Moved here from fit
    )

    # Fit model with early stopping
    # Pass the original (non-DataFrame) scaled arrays here
    model.fit(X_train_scaled, y_train,
              eval_set=[(X_test_scaled, y_test)], # Use scaled test set for evaluation
              verbose=False) # Set verbose=True or a number to see training progress

    logging.info("Model training completed.")

    # Evaluate model
    logging.info("Evaluating model performance...")
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    logging.info(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
    logging.info(f"Confusion Matrix:\n{str(confusion_matrix(y_test, y_pred))}")
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        logging.info(f"ROC AUC Score: {roc_auc:.4f}")
    except ValueError as e:
         logging.warning(f"Could not calculate ROC AUC score: {e}")
         roc_auc = None # Handle case with only one class in y_test or y_pred_proba


    # --- Plotting and Saving ---
    output_dir.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

    # Plot ROC Curve
    if roc_auc is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        roc_curve_path = output_dir / 'xgboost_roc_curve.png'
        try:
            plt.savefig(roc_curve_path)
            logging.info(f"ROC curve saved to {roc_curve_path}")
        except Exception as e:
            logging.error(f"Failed to save ROC curve plot: {e}")
        plt.close()
    else:
         logging.info("Skipping ROC curve plotting due to calculation error.")


    # Feature Importance
    logging.info("Feature Importances:")
    try:
        # Use the scaled DataFrame for correct feature names
        feature_importances = pd.Series(model.feature_importances_, index=X_test_scaled_df.columns).sort_values(ascending=False)
        logging.info(f"\n{feature_importances.to_string()}")
        # Plot Feature Importance
        plt.figure(figsize=(10, max(8, len(features) * 0.5))) # Adjust height based on number of features
        feature_importances.nlargest(min(20, len(features))).plot(kind='barh') # Plot top 20 or all if fewer
        plt.title('XGBoost Feature Importance (Top 20)')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis() # Display most important at top
        feature_importance_path = output_dir / 'xgboost_feature_importance.png'
        plt.tight_layout()
        plt.savefig(feature_importance_path)
        logging.info(f"Feature importance plot saved to {feature_importance_path}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to generate or save feature importance plot: {e}")


    return model, scaler, X_test_scaled_df, y_test # Return scaled DF for consistency

if __name__ == "__main__":
    project_root = get_project_root()
    # Define paths relative to the root
    data_path = project_root / "data/model_input/brasil/sao_paulo_census_tracts_full.geojson"
    output_dir = project_root / "models/sao_paulo_xgboost" # Specific output dir for this model

    # Load data
    gdf = load_data(data_path)

    # Train, evaluate, and get artifacts
    # This function now handles evaluation plots internally
    model, scaler, X_test, y_test = train_and_evaluate_xgb(gdf, output_dir)

    # Save the final model and scaler
    save_artifacts(model, scaler, output_dir)

    # --- Predict on Full Dataset and Save --- M
    try:
        logging.info("Generating XGBoost predictions for the full dataset...")
        df_pred = gdf.copy() # Use the originally loaded gdf

        # --- Prepare Features for Full Prediction (must match training) ---
        # One-hot encode (if applicable)
        if 'CD_SIT' in df_pred.columns:
            df_pred = pd.get_dummies(df_pred, columns=['CD_SIT'], prefix='SIT', drop_first=True)
        
        # Identify final feature columns (excluding original CD_SIT)
        final_features = [
            'area_km2',
            'cell_tower_count',
            'cell_tower_density',
            'min_dist_to_substation_km',
            'min_dist_to_transmission_line_km',
            'min_dist_to_hyperscaler_km',
            'viirs_mean'
        ] + [col for col in df_pred.columns if col.startswith('SIT_')]

        # Ensure all expected features are present, fill missing with 0 (align with pd.get_dummies)
        # This accounts for potential CD_SIT values present in full data but not training
        train_feature_names = X_test.columns.tolist() # Get feature names from training
        missing_in_pred = list(set(train_feature_names) - set(df_pred.columns))
        for col in missing_in_pred:
            df_pred[col] = 0
        df_pred = df_pred[train_feature_names] # Ensure same order and columns as training
        logging.info(f"Columns aligned for prediction. Shape: {df_pred.shape}")
        
        # Check for NaNs after preparation (just in case)
        if df_pred.isnull().sum().any():
            logging.error("NaN values detected in features before scaling for prediction. Exiting.")
            sys.exit(1)
            
        # Scale features using the *fitted* scaler
        X_full_scaled = scaler.transform(df_pred)
            
        # Predict probabilities
        full_predictions = model.predict_proba(X_full_scaled)[:, 1]
        gdf['predicted_prob'] = full_predictions # Standardize column name
        logging.info(f"XGBoost predictions added. Mean probability: {full_predictions.mean():.4f}")
            
        # --- Save Predictions GeoJSON ---
        prediction_output_dir = project_root / 'data' / 'model_output' / 'brasil'
        prediction_output_dir.mkdir(parents=True, exist_ok=True)
        # Use a distinct file name for XGBoost predictions
        predictions_path = prediction_output_dir / 'sao_paulo_census_tracts_predictions_xgb.geojson' 
            
        # Select relevant columns (using standardized 'predicted_prob')
        cols_to_save = ['CD_SETOR', 'geometry', 'has_data_center', 'predicted_prob']
        # Add other columns if needed, e.g., original features
        existing_cols_to_save = [c for c in cols_to_save if c in gdf.columns]
        if len(existing_cols_to_save) != len(cols_to_save):
             logging.warning(f"Could not find all requested columns for saving. Found: {existing_cols_to_save}")

        logging.info(f"Saving XGBoost predictions to {predictions_path}...")
        gdf[existing_cols_to_save].to_file(predictions_path, driver='GeoJSON')
        logging.info("XGBoost predictions GeoJSON saved successfully.")

        # --- Generate Heatmap ---
        logging.info("Generating heatmap visualization...")
        # Load data centers for overlay
        data_centers_path = project_root / 'data' / 'processed' / 'brasil' / 'data_centers.geojson'
        data_centers_gdf = gpd.read_file(data_centers_path)
        
        # Create plots directory (now using outputs/figures)
        plots_dir = project_root / 'outputs' / 'figures'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate heatmap for São Paulo City
        plot_probability_heatmap(
            tracts_gdf=gdf[existing_cols_to_save],
            probability_col='predicted_prob',
            points_gdf=data_centers_gdf,
            output_path=plots_dir / 'sao_paulo_city_heatmap_xgb.png',
            title="XGBoost Predicted Probabilities",
            cmap='Reds',
            filter_city_code='3550308',
            city_name="São Paulo City",
            point_size=40,
            point_color='blue',
            point_edgecolor='white',
            point_linewidth=0.5,
            show_plot=True
        )
        logging.info("Heatmap generated successfully.")

        # --- Top N Tract Analysis ---
        logging.info("Analyzing data centers in top predicted tracts...")
        gdf_sorted = gdf.sort_values(by='predicted_prob', ascending=False)
        top_n_values = [10, 50, 200]
        total_dcs = gdf['has_data_center'].sum()
        logging.info(f"Total actual data centers in dataset: {total_dcs}")

        for n in top_n_values:
            if len(gdf_sorted) >= n:
                top_n_tracts = gdf_sorted.head(n)
                dcs_in_top_n = top_n_tracts['has_data_center'].sum()
                logging.info(f"Top {n} tracts: {dcs_in_top_n} have data centers.")
            else:
                logging.warning(f"Dataset has fewer than {n} tracts ({len(gdf_sorted)}). Cannot perform Top {n} analysis.")
        
        # --- False Positive Analysis ---
        logging.info("\n=== High-Probability False Positive Analysis ===")
        # Get top 1000 predicted tracts to extract sufficient false positives
        top_1000 = gdf_sorted.head(1000)
        
        # Split into true positives and false positives
        true_positives_large = top_1000[top_1000['has_data_center'] == 1]
        false_positives_all = top_1000[top_1000['has_data_center'] == 0]
        
        # Take the top 300 false positives by probability
        false_positives = false_positives_all.head(300)
        
        logging.info(f"In top 1000 highest-probability tracts: {len(true_positives_large)} true positives, {len(false_positives_all)} false positives")
        logging.info(f"Analyzing top 300 false positives with highest predicted probability")
        
        # All actual data centers for comparison
        all_dcs = gdf[gdf['has_data_center'] == 1]
        
        # Get the feature importances
        feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
        # Get top features (importance > 5%)
        top_feature_names = feature_importances[feature_importances > 0.05].index.tolist()
        
        # Describe the feature values for different groups
        logging.info("\nFeature statistics for high-probability false positives vs. actual data centers:")
        for feature in top_feature_names:
            if feature in false_positives.columns:
                fp_mean = false_positives[feature].mean()
                tp_mean = true_positives_large[feature].mean() if not true_positives_large.empty else "N/A"
                all_dc_mean = all_dcs[feature].mean() if not all_dcs.empty else "N/A"
                dataset_mean = gdf[feature].mean()
                
                logging.info(f"\nFeature: {feature}")
                logging.info(f"  False Positives (top 300) mean: {fp_mean:.4f}")
                logging.info(f"  True Positives mean: {tp_mean}")
                logging.info(f"  All Data Centers mean: {all_dc_mean}")
                logging.info(f"  Dataset mean: {dataset_mean:.4f}")
                
                if isinstance(tp_mean, float) and isinstance(all_dc_mean, float):
                    # If FP values are closer to TP than to dataset average, this feature is driving the high predictions
                    if abs(fp_mean - tp_mean) < abs(fp_mean - dataset_mean):
                        logging.info(f"  Analysis: False positives resemble true data centers in this feature")
                    else:
                        logging.info(f"  Analysis: False positives differ from typical data centers in this feature")
        
        # Show a few examples of highest-probability false positives
        logging.info("\nTop 10 highest-probability false positives:")
        for idx, row in false_positives.head(10).iterrows():
            tract_id = row.get('CD_SETOR', idx)
            prob = row['predicted_prob']
            logging.info(f"Tract {tract_id} - Predicted probability: {prob:.4f}")
            for feature in top_feature_names:
                if feature in row:
                    logging.info(f"  {feature}: {row[feature]:.4f}")
                    
        # Add distribution analysis for key driving features
        logging.info("\nFeature distribution in top 300 false positives:")
        for feature in top_feature_names:
            if feature in false_positives.columns:
                # Calculate percentiles
                p10 = false_positives[feature].quantile(0.1)
                p25 = false_positives[feature].quantile(0.25)
                p50 = false_positives[feature].quantile(0.5)
                p75 = false_positives[feature].quantile(0.75)
                p90 = false_positives[feature].quantile(0.9)
                
                logging.info(f"\nFeature: {feature} distribution:")
                logging.info(f"  10th percentile: {p10:.4f}")
                logging.info(f"  25th percentile: {p25:.4f}")
                logging.info(f"  Median (50th): {p50:.4f}")
                logging.info(f"  75th percentile: {p75:.4f}")
                logging.info(f"  90th percentile: {p90:.4f}")
        # ------------------------------
        
    except Exception as e:
        logging.error(f"Error during full dataset prediction or visualization: {e}", exc_info=True)
    # -------------------------------------------

    logging.info("XGBoost prediction script finished successfully!")
    logging.info(f"Model, scaler, and plots saved in: {output_dir}") 
    