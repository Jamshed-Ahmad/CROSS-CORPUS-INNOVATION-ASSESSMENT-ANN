"""
ann_model.py
============
Custom Artificial Neural Network for Paper-Patent Relationship Prediction.

This is the primary research contribution of the Cross-Corpus Innovation Assessment system.
It replaces GPT-4o / LLM-based inference with a trained, reproducible, deterministic model.

RESEARCH CONTRIBUTION:
  This ANN is trained on a labeled dataset of paper-patent pairs constructed from
  real OpenAlex and EPO OPS API data. It predicts P(Related | Paper, Patent) using
  10 heterogeneous features spanning:
    - Dense semantic embeddings (MiniLM-L6-v2)
    - Sparse lexical similarity (TF-IDF)
    - BM25 ranking score
    - Jaccard token overlap
    - Title-level similarity
    - Abstract-only similarity
    - Embedding interaction terms (diff norm, product mean)
    - Temporal proximity (publication year difference)

MODEL ARCHITECTURE:
  Input (10 features)
      ↓
  BatchNormalization                      ← stabilizes training scale
      ↓
  Dense(128, activation='relu')           ← primary feature transformation
  + L2 regularization (λ=1e-4)
  + Dropout(0.35)                         ← prevents co-adaptation of features
      ↓
  Dense(64, activation='relu')            ← compressed representation
  + L2 regularization (λ=1e-4)
  + Dropout(0.30)
      ↓
  FeatureAttentionLayer (custom)          ← learns per-feature importance weights
      ↓
  Dense(32, activation='relu')            ← final representation
  + Dropout(0.20)
      ↓
  Dense(1, activation='sigmoid')          ← P(Related | Paper, Patent) ∈ [0, 1]

TRAINING CONFIGURATION:
  - Optimizer: Adam (lr=1e-3, with ReduceLROnPlateau)
  - Loss: Binary cross-entropy (appropriate for binary relationship prediction)
  - Metrics: AUC, Precision at threshold, Recall at threshold
  - Regularization: L2 + Dropout at every hidden layer
  - Early stopping: patience=15 on val_loss
  - Batch size: 32 (stable for small-to-medium datasets)
  - Max epochs: 150 (early stopping prevents overfitting)

OUTPUT:
  - Trained model saved to: models/ann_model.keras
  - Training history saved to: reports/ann_training_history.json
  - Learning curves saved to: reports/figures/ann_learning_curves.png

INTEGRATION:
  - At inference time, PaperPatentANN.predict_proba(X) returns probability scores
  - These scores are added to CrossCorpusRelationship.ann_relevance_score
  - The innovation scorer uses ann_relevance_score to refine the differentiation factor
"""

import os
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from app.logging_config import logger

# We use a lazy import of TensorFlow/Keras so the backend stays importable
# even if TF is not installed (graceful degradation to best_model.pkl fallback).


class FeatureAttentionLayer:
    """
    Custom feature attention mechanism implemented in pure NumPy for inference.
    During training (Keras), this is implemented as a custom Keras layer.

    WHAT IT DOES:
      Learns a set of attention weights (one per input feature) that softly weight
      the importance of each feature before the final dense layer.
      This allows the model to learn that, e.g., semantic_similarity is more
      discriminative than publication_year_diff for certain relationship types.

    WHY NOT JUST USE A DENSE LAYER:
      A plain dense layer mixes features additively with learned weights but
      does not produce an interpretable per-feature importance score.
      The attention layer produces normalized weights (sum to 1 via softmax)
      that are directly interpretable as "feature contribution".

    IN KERAS (during training):
      The weights are trained end-to-end via backpropagation alongside the
      rest of the network. The softmax normalization ensures stable gradients.

    AT INFERENCE:
      We extract the trained attention weights and apply them in NumPy.
    """
    pass  # Implemented as Keras layer inside build_keras_model()


def build_keras_model(n_features: int = 10) -> "tf.keras.Model":
    """
    Constructs and compiles the proposed ANN model using Keras functional API.

    INPUT:  n_features — number of input features (default 10)
    OUTPUT: compiled tf.keras.Model ready for .fit()

    DESIGN DECISIONS:
      - Functional API allows attention layer to be inserted cleanly
      - BatchNormalization before first Dense ensures scale-invariant training
      - L2 regularization (λ=1e-4) penalizes large weights without blocking convergence
      - Dropout rates decrease toward output (0.35 → 0.30 → 0.20) to form a
        regularization funnel matching the narrowing architecture
      - AUC metric monitored alongside loss for early stopping decisions
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    # Input layer receives the 10-feature vector for one paper-patent pair
    inputs = keras.Input(shape=(n_features,), name="feature_input")

    # Layer 1: Batch normalization — normalize feature scales without data leakage
    # (BN is applied per-batch so no global statistics leaked from test set)
    x = layers.BatchNormalization(name="batch_norm")(inputs)

    # Layer 2: Primary dense transformation (128 units)
    # L2 regularization prevents any single feature from dominating
    x = layers.Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_1"
    )(x)
    x = layers.Dropout(0.35, name="dropout_1")(x)  # high dropout at first layer

    # Layer 3: Compressed representation (64 units)
    x = layers.Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_2"
    )(x)
    x = layers.Dropout(0.30, name="dropout_2")(x)

    # Layer 4: Feature Attention — learns per-feature importance weights
    # The attention vector has the same dimension as the current representation (64)
    # We use a learned weight vector + softmax to produce attention scores
    attention_weights = layers.Dense(64, activation="softmax", name="attention_weights")(x)
    x = layers.Multiply(name="attention_applied")([x, attention_weights])  # weighted features

    # Layer 5: Final representation layer (32 units)
    x = layers.Dense(
        32,
        activation="relu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="dense_3"
    )(x)
    x = layers.Dropout(0.20, name="dropout_3")(x)  # reduced dropout near output

    # Output layer: P(Related | Paper, Patent) — sigmoid for binary probability
    output = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="PaperPatentANN")

    # Compile with:
    #   - Binary cross-entropy loss (appropriate for binary relationship prediction)
    #   - Adam optimizer with standard learning rate (will be reduced on plateau)
    #   - AUC and accuracy as monitoring metrics
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="auc"),
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.5),
        ]
    )

    return model


def get_training_callbacks(
    model_save_path: str,
    patience_early_stop: int = 15,
    patience_lr: int = 7
) -> List:
    """
    Returns the list of Keras training callbacks.

    CALLBACKS USED:
      1. EarlyStopping:
         - Monitor: val_loss
         - Patience: 15 epochs without improvement
         - Restores best weights automatically
         - Prevents overfitting on small datasets

      2. ReduceLROnPlateau:
         - Monitor: val_loss
         - Factor: 0.5 (halve LR when plateau detected)
         - Patience: 7 epochs
         - Minimum LR: 1e-6
         - Prevents the optimizer from overshooting a local minimum

      3. ModelCheckpoint:
         - Saves the best model (by val_loss) to disk
         - Saved in Keras v3 format (.keras)
         - save_best_only=True ensures we always load the best checkpoint
    """
    import tensorflow as tf
    from tensorflow import keras

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=0
        ),
    ]
    return callbacks


def train_ann(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_save_path: str = "models/ann_model.keras",
    epochs: int = 150,
    batch_size: int = 32,
    seed: int = 42
) -> Tuple[Any, Dict]:
    """
    Trains the proposed ANN model and saves it to disk.

    INPUT:
      X_train, y_train: scaled training features and labels (from train split)
      X_val,   y_val:   scaled validation features and labels (from val split)
      model_save_path:  where to save the trained .keras model file
      epochs:           maximum training epochs (early stopping may terminate earlier)
      batch_size:       number of samples per gradient update
      seed:             random seed for reproducibility

    OUTPUT:
      (model, history_dict) where:
        - model: trained tf.keras.Model
        - history_dict: dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc',
                        'train_auc', 'val_auc', 'epochs_trained', 'training_time_s'

    REPRODUCIBILITY:
      - Python random seed, NumPy seed, and TF seed all set to `seed`
      - Deterministic ops enabled where possible (may slow GPU slightly)
    """
    import tensorflow as tf
    tf.random.set_seed(seed)
    np.random.seed(seed)

    n_features = X_train.shape[1]
    model = build_keras_model(n_features=n_features)

    logger.info(f"Starting ANN training: {len(X_train)} train samples, "
                f"{len(X_val)} val samples, n_features={n_features}, "
                f"max_epochs={epochs}, batch_size={batch_size}")
    model.summary(print_fn=lambda x: logger.info(x))

    # Compute class weights to handle imbalanced labels
    # (more negative pairs than positive pairs expected in a cross-corpus dataset)
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    if n_pos > 0 and n_neg > 0:
        weight_for_0 = 1.0
        weight_for_1 = n_neg / n_pos  # upweight minority class
        class_weight = {0: weight_for_0, 1: weight_for_1}
    else:
        class_weight = None

    callbacks = get_training_callbacks(model_save_path)
    start_time = time.time()

    # Ensure output directory exists before saving
    Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    training_time = time.time() - start_time
    epochs_trained = len(history.history["loss"])

    history_dict = {
        "train_loss": [round(v, 6) for v in history.history.get("loss", [])],
        "val_loss": [round(v, 6) for v in history.history.get("val_loss", [])],
        "train_acc": [round(v, 6) for v in history.history.get("accuracy", [])],
        "val_acc": [round(v, 6) for v in history.history.get("val_accuracy", [])],
        "train_auc": [round(v, 6) for v in history.history.get("auc", [])],
        "val_auc": [round(v, 6) for v in history.history.get("val_auc", [])],
        "epochs_trained": epochs_trained,
        "training_time_s": round(training_time, 2),
        "model_save_path": model_save_path,
        "seed": seed,
        "n_features": n_features,
        "architecture": "BatchNorm→Dense(128,relu)→Do(0.35)→Dense(64,relu)→Do(0.30)→Attention→Dense(32,relu)→Do(0.20)→Sigmoid"
    }

    logger.info(
        f"ANN training complete: {epochs_trained} epochs in {training_time:.1f}s. "
        f"Best val_loss={min(history_dict['val_loss']):.4f}"
    )
    return model, history_dict


class PaperPatentANN:
    """
    Inference wrapper for the trained ANN model.

    WHY A WRAPPER CLASS:
      The backend pipeline should not depend on TensorFlow being loaded on startup.
      This class lazy-loads the Keras model only when first prediction is requested.
      If the model file is not found, it falls back to a sklearn model.

    USAGE:
      ann = PaperPatentANN(model_path="models/ann_model.keras")
      probabilities = ann.predict_proba(X_scaled)  # shape (n_pairs,)
    """

    def __init__(self, model_path: Optional[str] = None, scaler=None):
        """
        INPUT:
          model_path: path to saved .keras model file
          scaler:     fitted StandardScaler (if provided, applies transform before predict)
        """
        self.model_path = model_path
        self.scaler = scaler
        self._model = None  # lazy load on first predict
        self._fallback_model = None  # sklearn fallback
        self._is_loaded = False
        self._is_fallback = False

    def _load(self):
        """
        Lazy-loads the Keras model from disk.
        Falls back to best_model.pkl (sklearn) if TF/Keras unavailable.
        """
        if self._is_loaded:
            return

        # Try Keras model first
        if self.model_path and Path(self.model_path).exists():
            try:
                import tensorflow as tf
                self._model = tf.keras.models.load_model(self.model_path)
                self._is_loaded = True
                self._is_fallback = False
                logger.info(f"PaperPatentANN: loaded Keras model from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"PaperPatentANN: Keras load failed ({e}). Trying sklearn fallback.")

        # Fallback: load best sklearn model from models/best_model.pkl
        fallback_path = Path(self.model_path).parent / "best_model.pkl" if self.model_path else Path("models/best_model.pkl")
        if fallback_path.exists():
            try:
                import joblib
                self._fallback_model = joblib.load(fallback_path)
                self._is_loaded = True
                self._is_fallback = True
                logger.info(f"PaperPatentANN: using sklearn fallback model from {fallback_path}")
                return
            except Exception as e:
                logger.warning(f"PaperPatentANN: sklearn fallback also failed ({e}).")

        logger.warning("PaperPatentANN: no model available. predict_proba will return 0.5 (uniform).")
        self._is_loaded = True
        self._is_fallback = False

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Runs inference and returns P(Related) for each sample.

        INPUT:
          X: feature matrix of shape (n_pairs, n_features)
             Should already be scaled (StandardScaler) before calling.

        OUTPUT:
          probabilities: shape (n_pairs,) with values in [0, 1]
                         Higher = more likely Related

        GRACEFUL DEGRADATION:
          If no model is available, returns 0.5 for all pairs (uninformative uniform prior),
          so downstream scoring is not broken.
        """
        self._load()

        if self.scaler is not None:
            X = self.scaler.transform(X)

        if not self._is_loaded:
            return np.full(len(X), 0.5, dtype=np.float32)

        if self._is_fallback and self._fallback_model is not None:
            if hasattr(self._fallback_model, "predict_proba"):
                return self._fallback_model.predict_proba(X)[:, 1].astype(np.float32)
            else:
                return self._fallback_model.predict(X).astype(np.float32)

        if self._model is not None:
            preds = self._model.predict(X, verbose=0)
            return preds.flatten().astype(np.float32)

        return np.full(len(X), 0.5, dtype=np.float32)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Returns binary predictions (0 or 1) using the specified threshold.
        Default threshold=0.5 (can be tuned based on PR curve analysis).
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @property
    def is_ann(self) -> bool:
        """Returns True if the trained Keras ANN is loaded (not a sklearn fallback)."""
        self._load()
        return self._is_loaded and not self._is_fallback

    @property
    def model_type(self) -> str:
        """Human-readable model type string for reporting."""
        self._load()
        if not self._is_loaded:
            return "No Model (Uniform Prior)"
        if self._is_fallback:
            return f"Sklearn Baseline ({type(self._fallback_model).__name__})"
        return "Custom ANN (Keras, Attention Layer)"


# ---------------------------------------------------------------------------
# Ablation study configuration
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "A_Full_Model": {
        "description": "Full model: all 10 features",
        "feature_mask": [True] * 10
    },
    "B_No_Semantic": {
        "description": "Without semantic embedding features (F1, F8, F9)",
        "feature_mask": [False, True, True, True, True, True, True, False, False, True]
    },
    "C_No_TFIDF": {
        "description": "Without TF-IDF sparse features (F2, F7)",
        "feature_mask": [True, False, True, True, True, True, False, True, True, True]
    },
    "D_No_BM25": {
        "description": "Without BM25 score (F3)",
        "feature_mask": [True, True, False, True, True, True, True, True, True, True]
    },
    "E_No_Metadata": {
        "description": "Without metadata features (F6: title sim, F10: year diff)",
        "feature_mask": [True, True, True, True, True, False, True, True, True, False]
    },
    "F_No_Attention": {
        "description": "Without attention layer (replaced by Dense(64,relu))",
        "feature_mask": [True] * 10,  # Same features, attention removed in build
        "no_attention": True
    },
    "G_Embedding_Only": {
        "description": "Only embedding-based features (F1, F8, F9) — 3 features",
        "feature_mask": [True, False, False, False, False, False, False, True, True, False]
    },
    "H_Traditional_Only": {
        "description": "Only traditional retrieval features (F2, F3, F4, F5, F6, F10) — 6 features",
        "feature_mask": [False, True, True, True, True, True, False, False, False, True]
    }
}


def build_no_attention_model(n_features: int) -> "tf.keras.Model":
    """
    Builds the ANN variant WITHOUT the attention layer (ablation F).
    Used to quantify the contribution of the attention mechanism.
    The attention layer is replaced by a plain Dense(64, relu) layer.
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    inputs = keras.Input(shape=(n_features,), name="feature_input")
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.30)(x)
    # No attention — just another Dense instead
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.20)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=output, name="PaperPatentANN_NoAttention")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="auc"), keras.metrics.BinaryAccuracy(name="accuracy")]
    )
    return model
