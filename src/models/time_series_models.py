import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
from typing import Dict, List, Tuple, Optional
from config import MODEL_CONFIG
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set TensorFlow to use less verbose logging
tf.get_logger().setLevel('ERROR')

def prepare_lstm_data(df: pd.DataFrame, target_col: str, sequence_length: int = 30,
                     segment_col: str = 'segment') -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Prepare data for LSTM model training"""
    try:
        logger.info(f"Preparing LSTM data with sequence length: {sequence_length}")
        
        # Feature columns (exclude target, date, and identifiers)
        feature_cols = [col for col in df.columns if col not in [target_col, 'date', 'customer_id', segment_col]]
        
        # Initialize scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        X_sequences = []
        y_sequences = []
        
        # Process each segment separately
        segments = df[segment_col].unique() if segment_col in df.columns else ['ALL']
        
        for segment in segments:
            if segment_col in df.columns:
                segment_data = df[df[segment_col] == segment].copy()
            else:
                segment_data = df.copy()
            
            segment_data = segment_data.sort_values('date')
            
            if len(segment_data) < sequence_length + 1:
                logger.warning(f"Insufficient data for segment {segment}: {len(segment_data)} < {sequence_length + 1}")
                continue
            
            # Scale features and target
            features = segment_data[feature_cols].values
            target = segment_data[target_col].values.reshape(-1, 1)
            
            # Fit scaler on first segment, transform on all
            if len(X_sequences) == 0:
                scaler.fit(features)
                target_scaler = MinMaxScaler(feature_range=(0, 1))
                target_scaler.fit(target)
            
            features_scaled = scaler.transform(features)
            target_scaled = target_scaler.transform(target)
            
            # Create sequences
            for i in range(sequence_length, len(features_scaled)):
                X_sequences.append(features_scaled[i-sequence_length:i])
                y_sequences.append(target_scaled[i, 0])
        
        if len(X_sequences) == 0:
            logger.error("No sequences created")
            return np.array([]), np.array([]), np.array([]), np.array([]), scaler
        
        X = np.array(X_sequences)
        y = np.array(y_sequences)
        
        # Train/test split (temporal)
        split_point = int(len(X) * 0.8)
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        
        logger.info(f"LSTM data prepared - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test, scaler
        
    except Exception as e:
        logger.error(f"Error preparing LSTM data: {e}")
        return np.array([]), np.array([]), np.array([]), np.array([]), MinMaxScaler()

def build_lstm_model(input_shape: Tuple[int, int], lstm_units: int = 50,
                    dropout_rate: float = 0.2, num_layers: int = 2) -> Sequential:
    """Build LSTM model architecture"""
    try:
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=lstm_units,
            return_sequences=True if num_layers > 1 else False,
            input_shape=input_shape,
            name='lstm_1'
        ))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i in range(1, num_layers):
            model.add(LSTM(
                units=lstm_units // (i + 1),
                return_sequences=True if i < num_layers - 1 else False,
                name=f'lstm_{i+1}'
            ))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(units=lstm_units // 2, activation='relu', name='dense_1'))
        model.add(Dropout(dropout_rate))
        
        model.add(Dense(units=lstm_units // 4, activation='relu', name='dense_2'))
        model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(units=1, activation='linear', name='output'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info(f"Built LSTM model with {num_layers} LSTM layers and {lstm_units} units")
        return model
        
    except Exception as e:
        logger.error(f"Error building LSTM model: {e}")
        return None

def train_lstm_model(X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray = None, y_val: np.ndarray = None,
                    epochs: int = 50, batch_size: int = 32,
                    patience: int = 10) -> Tuple[Sequential, Dict]:
    """Train LSTM model with early stopping"""
    try:
        logger.info("Training LSTM model...")
        
        # Build model
        input_shape = (X_train.shape[1], X_train.shape[2])
        model = build_lstm_model(
            input_shape=input_shape,
            lstm_units=MODEL_CONFIG['LSTM']['units'],
            dropout_rate=MODEL_CONFIG['LSTM']['dropout']
        )
        
        if model is None:
            return None, {}
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001,
                verbose=1
            )
        ]
        
        # Validation data
        validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        # Model information
        model_info = {
            'model_type': 'LSTM',
            'input_shape': input_shape,
            'total_params': model.count_params(),
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_train_mae': history.history['mae'][-1],
            'training_history': history.history
        }
        
        if validation_data is not None:
            model_info['final_val_loss'] = history.history['val_loss'][-1]
            model_info['final_val_mae'] = history.history['val_mae'][-1]
        
        logger.info(f"LSTM training completed - Final loss: {model_info['final_train_loss']:.4f}")
        return model, model_info
        
    except Exception as e:
        logger.error(f"Error training LSTM model: {e}")
        return None, {}

def evaluate_lstm_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray,
                       scaler: MinMaxScaler = None) -> Dict:
    """Evaluate LSTM model performance"""
    try:
        # Make predictions
        y_pred = model.predict(X_test, verbose=0)
        
        # Inverse transform if scaler provided
        if scaler is not None:
            # Note: This assumes target was also scaled with MinMaxScaler
            y_test_original = y_test  # Assuming already in original scale
            y_pred_original = y_pred.flatten()
        else:
            y_test_original = y_test
            y_pred_original = y_pred.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test_original, y_pred_original)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_original, y_pred_original)
        
        # R-squared
        ss_res = np.sum((y_test_original - y_pred_original) ** 2)
        ss_tot = np.sum((y_test_original - np.mean(y_test_original)) ** 2)
        r2 = 1 - (ss_res / ss_tot)
        
        # MAPE
        mape = np.mean(np.abs((y_test_original - y_pred_original) / (y_test_original + 1e-8))) * 100
        
        # Directional accuracy
        if len(y_test_original) > 1:
            y_test_diff = np.diff(y_test_original)
            y_pred_diff = np.diff(y_pred_original)
            directional_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff)) * 100
        else:
            directional_accuracy = 0.0
        
        evaluation_results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'test_samples': len(y_test_original),
            'predictions': y_pred_original,
            'actual': y_test_original
        }
        
        logger.info(f"LSTM Model Performance:")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Error evaluating LSTM model: {e}")
        return {}

def create_lstm_forecast(model: Sequential, last_sequence: np.ndarray,
                        forecast_steps: int = 30) -> np.ndarray:
    """Create multi-step forecast using trained LSTM model"""
    try:
        logger.info(f"Creating {forecast_steps}-step forecast...")
        
        # Initialize forecast array
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for step in range(forecast_steps):
            # Predict next value
            next_pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
            forecasts.append(next_pred[0, 0])
            
            # Update sequence for next prediction
            # For simplicity, we'll use the prediction for all features
            # In practice, you'd want to handle this more carefully
            new_row = current_sequence[-1].copy()
            new_row[0] = next_pred[0, 0]  # Update first feature with prediction
            
            # Shift sequence and add new row
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(forecasts)
        
    except Exception as e:
        logger.error(f"Error creating LSTM forecast: {e}")
        return np.array([])

def analyze_lstm_predictions(evaluation_results: Dict) -> Dict:
    """Analyze LSTM prediction patterns"""
    try:
        predictions = evaluation_results.get('predictions', np.array([]))
        actual = evaluation_results.get('actual', np.array([]))
        
        if len(predictions) == 0 or len(actual) == 0:
            return {}
        
        # Residual analysis
        residuals = actual - predictions
        
        # Statistical tests on residuals
        from scipy import stats
        
        # Normality test (Shapiro-Wilk)
        if len(residuals) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = 0, 1
        
        # Autocorrelation test (Ljung-Box)
        # This requires statsmodels, so we'll calculate a simple autocorrelation
        autocorr_1 = np.corrcoef(residuals[:-1], residuals[1:])[0, 1] if len(residuals) > 1 else 0
        
        # Heteroscedasticity check (variance in different periods)
        mid_point = len(residuals) // 2
        first_half_var = np.var(residuals[:mid_point]) if mid_point > 0 else 0
        second_half_var = np.var(residuals[mid_point:]) if len(residuals) - mid_point > 0 else 0
        
        analysis_results = {
            'residuals': {
                'mean': np.mean(residuals),
                'std': np.std(residuals),
                'min': np.min(residuals),
                'max': np.max(residuals),
                'skewness': stats.skew(residuals),
                'kurtosis': stats.kurtosis(residuals)
            },
            'normality_test': {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'autocorrelation': {
                'lag_1': autocorr_1,
                'significant': abs(autocorr_1) > 0.1
            },
            'heteroscedasticity': {
                'first_half_variance': first_half_var,
                'second_half_variance': second_half_var,
                'variance_ratio': second_half_var / (first_half_var + 1e-8)
            }
        }
        
        logger.info("LSTM Prediction Analysis:")
        logger.info(f"  Residual mean: {analysis_results['residuals']['mean']:.4f}")
        logger.info(f"  Residual std: {analysis_results['residuals']['std']:.4f}")
        logger.info(f"  Normality (p-value): {shapiro_p:.4f}")
        logger.info(f"  Lag-1 autocorrelation: {autocorr_1:.4f}")
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error analyzing LSTM predictions: {e}")
        return {}

def create_lstm_ensemble(models: List[Sequential], X_test: np.ndarray) -> np.ndarray:
    """Create ensemble predictions from multiple LSTM models"""
    try:
        if not models or len(models) == 0:
            return np.array([])
        
        logger.info(f"Creating ensemble from {len(models)} LSTM models...")
        
        # Collect predictions from all models
        all_predictions = []
        for i, model in enumerate(models):
            pred = model.predict(X_test, verbose=0)
            all_predictions.append(pred.flatten())
        
        # Stack predictions
        predictions_array = np.stack(all_predictions, axis=0)
        
        # Calculate ensemble prediction (mean)
        ensemble_mean = np.mean(predictions_array, axis=0)
        
        # Calculate prediction uncertainty (std)
        ensemble_std = np.std(predictions_array, axis=0)
        
        logger.info(f"Ensemble predictions created with mean uncertainty: {np.mean(ensemble_std):.4f}")
        
        return ensemble_mean, ensemble_std
        
    except Exception as e:
        logger.error(f"Error creating LSTM ensemble: {e}")
        return np.array([]), np.array([])

def train_segment_specific_lstm(df: pd.DataFrame, target_col: str, segment_col: str = 'segment',
                               sequence_length: int = 30) -> Dict[str, Tuple[Sequential, Dict]]:
    """Train separate LSTM models for each customer segment"""
    try:
        logger.info("Training segment-specific LSTM models...")
        
        segment_models = {}
        segments = df[segment_col].unique()
        
        for segment in segments:
            logger.info(f"Training LSTM for segment: {segment}")
            
            # Filter data for this segment
            segment_data = df[df[segment_col] == segment].copy()
            
            if len(segment_data) < sequence_length * 2:  # Need minimum data
                logger.warning(f"Insufficient data for segment {segment}: {len(segment_data)}")
                continue
            
            # Prepare LSTM data for this segment
            X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
                segment_data, target_col, sequence_length, segment_col
            )
            
            if X_train.size == 0:
                logger.warning(f"No training data created for segment {segment}")
                continue
            
            # Train LSTM model
            model, model_info = train_lstm_model(
                X_train, y_train, X_test, y_test,
                epochs=MODEL_CONFIG['LSTM']['epochs'],
                batch_size=MODEL_CONFIG['LSTM']['batch_size']
            )
            
            if model is not None:
                # Evaluate model
                evaluation = evaluate_lstm_model(model, X_test, y_test, scaler)
                model_info.update(evaluation)
                
                segment_models[segment] = (model, model_info)
                logger.info(f"Segment {segment} LSTM trained - RMSE: {evaluation.get('rmse', 'N/A'):.4f}")
            else:
                logger.error(f"Failed to train LSTM for segment {segment}")
        
        return segment_models
        
    except Exception as e:
        logger.error(f"Error training segment-specific LSTM models: {e}")
        return {}

def save_lstm_model(model: Sequential, model_info: Dict, filename: str) -> bool:
    """Save LSTM model and metadata"""
    try:
        # Save model architecture and weights
        model.save(f'{filename}.h5')
        
        # Save model info
        import joblib
        joblib.dump(model_info, f'{filename}_info.pkl')
        
        logger.info(f"LSTM model saved to {filename}.h5")
        return True
        
    except Exception as e:
        logger.error(f"Error saving LSTM model: {e}")
        return False

def load_lstm_model(filename: str) -> Tuple[Sequential, Dict]:
    """Load LSTM model and metadata"""
    try:
        from tensorflow.keras.models import load_model
        import joblib
        
        # Load model
        model = load_model(f'{filename}.h5')
        
        # Load model info
        model_info = joblib.load(f'{filename}_info.pkl')
        
        logger.info(f"LSTM model loaded from {filename}.h5")
        return model, model_info
        
    except Exception as e:
        logger.error(f"Error loading LSTM model: {e}")
        return None, {}

def create_lstm_attention_model(input_shape: Tuple[int, int]) -> Sequential:
    """Create LSTM model with attention mechanism"""
    try:
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input
        
        # Input layer
        inputs = Input(shape=input_shape)
        
        # LSTM layer
        lstm_out = LSTM(
            units=MODEL_CONFIG['LSTM']['units'],
            return_sequences=True,
            dropout=MODEL_CONFIG['LSTM']['dropout']
        )(inputs)
        
        # Attention layer
        attention_out = MultiHeadAttention(
            num_heads=4,
            key_dim=MODEL_CONFIG['LSTM']['units'] // 4
        )(lstm_out, lstm_out)
        
        # Add & Norm
        attention_out = LayerNormalization()(lstm_out + attention_out)
        
        # Global pooling
        pooled_out = tf.keras.layers.GlobalAveragePooling1D()(attention_out)
        
        # Dense layers
        dense_out = Dense(MODEL_CONFIG['LSTM']['units'] // 2, activation='relu')(pooled_out)
        dense_out = Dropout(MODEL_CONFIG['LSTM']['dropout'])(dense_out)
        
        # Output
        outputs = Dense(1, activation='linear')(dense_out)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        logger.info("Created LSTM model with attention mechanism")
        return model
        
    except Exception as e:
        logger.error(f"Error creating attention LSTM model: {e}")
        return None

def main_lstm_pipeline(df: pd.DataFrame, target_col: str, segment_col: str = 'segment',
                      sequence_length: int = 30, train_ensemble: bool = False) -> Dict:
    """Main pipeline for LSTM model training and evaluation"""
    logger.info("=== Starting LSTM Pipeline ===")
    
    try:
        results = {}
        
        # 1. Prepare data
        logger.info("1. Preparing LSTM data...")
        X_train, X_test, y_train, y_test, scaler = prepare_lstm_data(
            df, target_col, sequence_length, segment_col
        )
        
        if X_train.size == 0:
            logger.error("Failed to prepare LSTM data")
            return {}
        
        results['data_info'] = {
            'sequence_length': sequence_length,
            'n_features': X_train.shape[2],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        # 2. Train main LSTM model
        logger.info("2. Training main LSTM model...")
        main_model, main_model_info = train_lstm_model(
            X_train, y_train, X_test, y_test,
            epochs=MODEL_CONFIG['LSTM']['epochs'],
            batch_size=MODEL_CONFIG['LSTM']['batch_size']
        )
        
        if main_model is not None:
            # Evaluate main model
            main_evaluation = evaluate_lstm_model(main_model, X_test, y_test, scaler)
            main_model_info.update(main_evaluation)
            results['main_model'] = (main_model, main_model_info)
            
            # Analysis
            analysis = analyze_lstm_predictions(main_evaluation)
            results['prediction_analysis'] = analysis
        
        # 3. Train segment-specific models
        logger.info("3. Training segment-specific LSTM models...")
        segment_models = train_segment_specific_lstm(df, target_col, segment_col, sequence_length)
        results['segment_models'] = segment_models
        
        # 4. Create ensemble if requested
        if train_ensemble and main_model is not None and segment_models:
            logger.info("4. Creating LSTM ensemble...")
            all_models = [main_model] + [model for model, _ in segment_models.values()]
            
            if len(all_models) > 1:
                ensemble_pred, ensemble_std = create_lstm_ensemble(all_models, X_test)
                results['ensemble_predictions'] = {
                    'predictions': ensemble_pred,
                    'uncertainty': ensemble_std,
                    'n_models': len(all_models)
                }
        
        # 5. Save models
        logger.info("5. Saving LSTM models...")
        if main_model is not None:
            save_lstm_model(main_model, main_model_info, 'data/models/lstm_main')
        
        for segment, (model, info) in segment_models.items():
            save_lstm_model(model, info, f'data/models/lstm_segment_{segment}')
        
        # 6. Create forecasts
        if main_model is not None and len(X_test) > 0:
            logger.info("6. Creating forecasts...")
            forecast = create_lstm_forecast(main_model, X_test[-1], forecast_steps=21)
            results['forecast'] = forecast
        
        logger.info("=== LSTM Pipeline Complete ===")
        return results
        
    except Exception as e:
        logger.error(f"Error in LSTM pipeline: {e}")
        return {}

if __name__ == "__main__":
    # Example usage
    logger.info("Time series models module loaded successfully")
    print("Available functions:")
    print("- prepare_lstm_data")
    print("- build_lstm_model")
    print("- train_lstm_model")
    print("- evaluate_lstm_model")
    print("- create_lstm_forecast")
    print("- analyze_lstm_predictions")
    print("- train_segment_specific_lstm")
    print("- create_lstm_ensemble")
    print("- main_lstm_pipeline")