"""
Multi-Label Classification System for Software Dependencies
Modular implementation with class-based architecture

This module provides a complete pipeline for multi-label classification of software
dependencies, including data loading, preprocessing, feature extraction, model training,
and evaluation with proper ML best practices including nested cross-validation.
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score, 
                            hamming_loss, precision_recall_fscore_support, 
                            jaccard_score, log_loss)
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
import joblib
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from typing import Dict, List, Tuple, Any, Optional
import warnings
import os
import logging
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and initial cleaning of data with validation."""
    
    def __init__(self, main_file: str, lookup_file: str):
        if not os.path.exists(main_file):
            raise FileNotFoundError(f"Main file not found: {main_file}")
        if not os.path.exists(lookup_file):
            raise FileNotFoundError(f"Lookup file not found: {lookup_file}")
            
        self.main_file = main_file
        self.lookup_file = lookup_file
        self.df: Optional[pd.DataFrame] = None
        self.dep_col: Optional[str] = None
        self.valid_dependencies: Optional[set] = None
        self.valid_labels: Optional[set] = None
        
    def load_and_clean_data(self) -> pd.DataFrame:
        """Load main dataset and perform basic cleaning."""
        logger.info(f"Loading data from {self.main_file}...")
        
        try:
            self.df = pd.read_csv(self.main_file)
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
        
        logger.info(f"Initial shape: {self.df.shape}")
        
        # Determine correct column name
        if 'dependencies_found' in self.df.columns:
            self.dep_col = 'dependencies_found'
        elif 'dependecies_found' in self.df.columns:
            self.dep_col = 'dependecies_found'
        else:
            raise ValueError("Cannot find dependencies column")
        
        if 'dependency_labels' not in self.df.columns:
            raise ValueError("Missing required column: 'dependency_labels'")
        
        # Remove duplicates
        initial_rows = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f"Removed {initial_rows - len(self.df)} duplicate rows")
        
        # Remove empty data
        self.df = self.df.dropna(subset=[self.dep_col, 'dependency_labels'])
        self.df = self.df[self.df[self.dep_col].str.strip() != '']
        self.df = self.df[self.df['dependency_labels'].str.strip() != '']
        
        logger.info(f"Shape after cleaning: {self.df.shape}")
        return self.df
    
    def load_lookup_data(self) -> Tuple[set, set]:
        """Load lookup table to define vocabulary."""
        logger.info(f"Loading lookup table from {self.lookup_file}...")
        
        try:
            lookup_df = pd.read_csv(self.lookup_file)
        except Exception as e:
            logger.error(f"Error loading lookup file: {e}")
            raise
        
        if 'dependency' not in lookup_df.columns or 'label' not in lookup_df.columns:
            raise ValueError("Lookup file must contain 'dependency' and 'label' columns")
        
        self.valid_dependencies = set(
            lookup_df['dependency'].dropna().astype(str).str.strip().unique()
        )
        self.valid_labels = set(
            lookup_df['label'].dropna().astype(str).str.strip().unique()
        )
        
        logger.info(f"Found {len(self.valid_dependencies)} valid dependencies")
        logger.info(f"Found {len(self.valid_labels)} valid labels")
        
        return self.valid_dependencies, self.valid_labels


class Preprocessor:
    """Handles data preprocessing and filtering with improved techniques."""
    
    @staticmethod
    def _parse_list_str(s: str) -> List[str]:
        """Helper to parse string representation of list."""
        if not isinstance(s, str):
            return []
        try:
            s = s.strip()
            if s.startswith('['):
                return ast.literal_eval(s)
            elif s:
                return [x.strip() for x in s.split(',') if x.strip()]
            return []
        except (ValueError, SyntaxError):
            return []

    @staticmethod
    def parse_dependencies(dep_str: str) -> List[str]:
        """Parse dependency string to list."""
        return Preprocessor._parse_list_str(dep_str)
    
    @staticmethod
    def parse_labels(label_str: str) -> List[str]:
        """Parse label string to list."""
        return Preprocessor._parse_list_str(label_str)
    
    @staticmethod
    def filter_labels(labels: List[str], valid_set: set) -> List[str]:
        """Filter labels against valid set with case-insensitive matching."""
        if not labels or not valid_set:
            return []
            
        filtered = []
        valid_set_lower = {v.lower(): v for v in valid_set}
        
        for label in labels:
            label_str = str(label).strip()
            if not label_str:
                continue
                
            if label_str in valid_set:
                filtered.append(label_str)
            elif label_str.lower() in valid_set_lower:
                filtered.append(valid_set_lower[label_str.lower()])
        
        return list(set(filtered))
    
    def preprocess_data(self, df: pd.DataFrame, dep_col: str, 
                       valid_dependencies: set, valid_labels: set) -> pd.DataFrame:
        """Complete preprocessing pipeline with quality checks."""
        logger.info("Preprocessing data...")
        initial_rows = len(df)
        
        df = df.copy()
        df['dep_list'] = df[dep_col].apply(self.parse_dependencies)
        df['dep_list'] = df['dep_list'].apply(lambda x: [str(d).lower() for d in x])
        
        valid_dependencies_lower = {str(d).lower() for d in valid_dependencies}
        df['filtered_deps'] = df['dep_list'].apply(
            lambda x: [dep for dep in x if dep in valid_dependencies_lower]
        )
        
        df['label_list'] = df['dependency_labels'].apply(self.parse_labels)
        valid_labels_norm = {str(l).strip() for l in valid_labels}
        df['filtered_labels'] = df['label_list'].apply(
            lambda x: self.filter_labels(x, valid_labels_norm)
        )
        
        # Quality check: Remove samples with too few dependencies or labels
        df = df[df['filtered_deps'].map(len) > 0]
        df = df[df['filtered_labels'].map(len) > 0]
        
        # Remove samples with excessive labels (potential noise)
        df = df[df['filtered_labels'].map(len) <= 10]
        
        logger.info(f"Removed {initial_rows - len(df)} rows during preprocessing")
        logger.info(f"Final shape: {df.shape}")
        
        return df


class FeatureExtractor:
    """Handles feature extraction with scaling and selection options."""
    
    def __init__(self):
        self.mlb_X: Optional[MultiLabelBinarizer] = None
        self.mlb_y: Optional[MultiLabelBinarizer] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector: Optional[SelectKBest] = None
        
    def extract_features(self, df: pd.DataFrame, valid_dependencies_lower: set,
                        apply_scaling: bool = False,
                        feature_selection_k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features with optional scaling and feature selection."""
        logger.info("Extracting features...")
        
        if df.empty:
            raise ValueError("Cannot extract features from empty dataframe")
        
        # Encode dependencies (X)
        self.mlb_X = MultiLabelBinarizer(classes=sorted(list(valid_dependencies_lower)))
        X = self.mlb_X.fit_transform(df['filtered_deps'])
        
        # Encode labels (y)
        present_labels = set([l for sublist in df['filtered_labels'] for l in sublist])
        if not present_labels:
            raise ValueError("No labels found in data")
            
        self.mlb_y = MultiLabelBinarizer(classes=sorted(list(present_labels)))
        y = self.mlb_y.fit_transform(df['filtered_labels'])
        
        # Optional: Apply scaling
        if apply_scaling:
            self.scaler = StandardScaler(with_mean=False)  # Sparse-safe scaling
            X = self.scaler.fit_transform(X)
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label matrix shape: {y.shape}")
        
        return X, y


class EDA:
    """Exploratory Data Analysis with enhanced statistics."""
    
    @staticmethod
    def analyze_and_visualize(df: pd.DataFrame):
        """Perform comprehensive EDA with class imbalance analysis."""
        print("\n" + "="*70)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*70)
        
        all_labels = [label for sublist in df['filtered_labels'] for label in sublist]
        all_deps = [dep for sublist in df['filtered_deps'] for dep in sublist]
        
        labels_per_repo = df['filtered_labels'].apply(len)
        deps_per_repo = df['filtered_deps'].apply(len)
        
        label_counts = pd.Series(all_labels).value_counts()
        
        # Enhanced statistics
        print(f"\n--- Label Statistics ---")
        print(f"Total Unique Labels: {len(label_counts)}")
        print(f"Most Frequent: '{label_counts.index[0]}' ({label_counts.iloc[0]} occurrences)")
        print(f"Least Frequent: '{label_counts.index[-1]}' ({label_counts.iloc[-1]} occurrences)")
        print(f"Median Frequency: {label_counts.median():.0f}")
        print(f"Mean Frequency: {label_counts.mean():.2f}")
        
        # Class imbalance ratio
        imbalance_ratio = label_counts.max() / label_counts.min()
        print(f"Imbalance Ratio: {imbalance_ratio:.2f} (max/min frequency)")
        
        print(f"\n--- Per-Sample Statistics ---")
        print(f"Avg Labels per Sample: {labels_per_repo.mean():.2f}")
        print(f"Median Labels per Sample: {labels_per_repo.median():.0f}")
        print(f"Max Labels: {labels_per_repo.max()}")
        print(f"Min Labels: {labels_per_repo.min()}")
        print(f"Avg Dependencies per Sample: {deps_per_repo.mean():.2f}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Top 20 labels
        sns.barplot(x=label_counts.head(20).values, y=label_counts.head(20).index, 
                   palette='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Top 20 Most Frequent Labels')
        axes[0, 0].set_xlabel('Count')
        
        # Label distribution
        axes[0, 1].plot(range(len(label_counts)), label_counts.values, 'b-', linewidth=2)
        axes[0, 1].fill_between(range(len(label_counts)), label_counts.values, alpha=0.3)
        axes[0, 1].set_title('Label Frequency Distribution (Imbalance)')
        axes[0, 1].set_xlabel('Label Rank')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bottom 20 labels
        bottom_20 = label_counts.tail(20).sort_values(ascending=True)
        sns.barplot(x=bottom_20.values, y=bottom_20.index, palette='rocket', ax=axes[0, 2])
        axes[0, 2].set_title('Bottom 20 Least Frequent Labels')
        
        # Label frequency histogram
        sns.histplot(label_counts, bins=30, kde=True, color='purple', ax=axes[1, 0])
        axes[1, 0].set_title('Label Frequency Distribution')
        axes[1, 0].set_yscale('log')
        
        # Labels per sample
        sns.histplot(labels_per_repo, bins=range(0, min(labels_per_repo.max() + 2, 20)), 
                    color='green', ax=axes[1, 1])
        axes[1, 1].set_title('Labels per Sample')
        axes[1, 1].axvline(labels_per_repo.mean(), color='red', linestyle='--', 
                          label=f'Mean: {labels_per_repo.mean():.1f}')
        axes[1, 1].legend()
        
        # Dependencies per sample
        sns.histplot(deps_per_repo, bins=30, kde=True, color='orange', ax=axes[1, 2])
        axes[1, 2].set_title('Dependencies per Sample')
        axes[1, 2].axvline(deps_per_repo.mean(), color='red', linestyle='--',
                          label=f'Mean: {deps_per_repo.mean():.1f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f'eda_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png', dpi=300)
        plt.show()


class ModelTrainer:
    """Enhanced model training with nested cross-validation and loss tracking."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.results: List[Dict] = []
        self.trained_models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.tuned_models: Dict[str, Any] = {}
        self.cv_scores: Dict[str, List] = {}
        self.loss_history: Dict[str, Dict] = {}
        self.nested_cv_results: Dict[str, Dict] = {}
        
    def initialize_models(self):
        """Initialize models with improved configurations."""
        logger.info("\nInitializing models...")
        
        # OPTIMIZED: Reduced n_estimators/max_iter and set n_jobs=1 to avoid contention with GridSearchCV
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42,
                # Fixed: changed n_jobs to 1 to prevent deadlock/thrashing during GridSearch
                max_depth=20, min_samples_split=5, n_jobs=1, verbose=0
            ),
            "Logistic Regression": OneVsRestClassifier(
                LogisticRegression(solver='saga', class_weight='balanced', 
                                 random_state=42, max_iter=1000, n_jobs=1, verbose=0)
            ),
            "Linear SVC": OneVsRestClassifier(
                LinearSVC(class_weight='balanced', random_state=42, 
                         dual='auto', max_iter=1000, verbose=0)
            ),
            "MLP Classifier": MLPClassifier(
                random_state=42, max_iter=500, early_stopping=True,
                hidden_layer_sizes=(100,), alpha=0.0001, verbose=False
            ),
            "XGBoost": OneVsRestClassifier(
                xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                random_state=42, eval_metric='logloss', 
                                use_label_encoder=False, n_jobs=1, verbosity=0)
            ),
            "LightGBM": OneVsRestClassifier(
                lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=42, verbose=-1, class_weight='balanced',
                                 n_jobs=1)
            ),
            "CatBoost": OneVsRestClassifier(
                CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1,
                                 random_state=42, verbose=False, 
                                 auto_class_weights='Balanced', thread_count=1)
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
    
    def get_param_grids(self) -> Dict[str, Dict]:
        """
        Define optimized hyperparameter grids.
        
        GridSearchCV will exhaustively search EVERY combination defined here.
        For OneVsRestClassifier models, 'estimator__' prefix is required to tune the inner model.
        """
        # OPTIMIZED: Reduced grid size for faster execution
        return {
            "Random Forest": {
                'n_estimators': [100, 200],
                'max_depth': [10, None]
            },
            "Logistic Regression": {
                'estimator__C': [1.0, 10.0]
            },
            "Linear SVC": {
                'estimator__C': [1.0, 10.0]
            },
            "MLP Classifier": {
                'hidden_layer_sizes': [(100,), (100, 50)],
                'alpha': [0.0001]
            },
            "XGBoost": {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.1]
            },
            "LightGBM": {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.1]
            },
            "CatBoost": {
                'estimator__iterations': [100, 200],
                'estimator__depth': [6],
                'estimator__learning_rate': [0.1]
            }
        }
    
    def calculate_loss(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate average log loss across all labels."""
        try:
            # Calculate loss for each label independently
            losses = []
            for i in range(y_true.shape[1]):
                loss = log_loss(y_true[:, i], y_pred_proba[:, i], labels=[0, 1])
                losses.append(loss)
            return np.mean(losses)
        except:
            return np.nan
    
    def track_training_loss(self, name: str, clf: Any, X_train: np.ndarray, 
                           y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray):
        """Track training and validation loss during training for MLP Classifier."""
        train_losses = []
        val_losses = []
        
        # Only MLP Classifier supports loss curve tracking
        if name == "MLP Classifier":
            if hasattr(clf, 'loss_curve_'):
                train_losses = clf.loss_curve_
                # For validation loss, approximation using final model
                try:
                    val_loss = self.calculate_loss(y_val, clf.predict_proba(X_val))
                    val_losses = [val_loss] * len(train_losses)
                except:
                    pass
        
        # Store loss history
        if train_losses:
            self.loss_history[name] = {
                'train_loss': train_losses,
                'val_loss': val_losses,
                'epochs': list(range(1, len(train_losses) + 1))
            }
    
    def nested_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                outer_cv: int = 5, inner_cv: int = 3,
                                n_jobs: int = -1) -> Dict[str, Dict]:
        """
        Perform nested cross-validation for unbiased performance estimation.
        
        Nested CV Structure:
        - Outer loop (5 folds): Evaluates model generalization performance
        - Inner loop (3 folds): Hyperparameter tuning
        
        This prevents data leakage and provides reliable performance estimates.
        
        Args:
            X: Feature matrix
            y: Label matrix
            outer_cv: Number of outer CV folds (default: 5)
            inner_cv: Number of inner CV folds for hyperparameter tuning (default: 3)
            n_jobs: Number of parallel jobs (default: -1)
            
        Returns:
            Dict containing nested CV results for each model
        """
        logger.info("\n" + "="*70)
        logger.info("NESTED CROSS-VALIDATION")
        logger.info("="*70)
        logger.info(f"Outer CV: {outer_cv} folds (performance estimation)")
        logger.info(f"Inner CV: {inner_cv} folds (hyperparameter tuning)")
        logger.info("="*70)
        
        param_grids = self.get_param_grids()
        outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
        
        for name, base_clf in self.models.items():
            if name not in param_grids:
                logger.info(f"\nSkipping {name} (no hyperparameter grid defined)")
                continue
            
            logger.info(f"\n{'='*50}")
            logger.info(f"Running Nested CV for: {name}")
            logger.info(f"{'='*50}")
            
            outer_scores = []
            best_params_per_fold = []
            
            # Outer loop - Performance estimation
            for fold_num, (train_idx, test_idx) in enumerate(outer_kfold.split(X), 1):
                logger.info(f"\n  Outer Fold {fold_num}/{outer_cv}")
                
                X_train_outer, X_test_outer = X[train_idx], X[test_idx]
                y_train_outer, y_test_outer = y[train_idx], y[test_idx]
                
                # Inner loop - Hyperparameter tuning using GridSearchCV
                try:
                    logger.info(f"    Running inner CV (GridSearchCV)...")
                    # Use explicit KFold for inner CV to handle multi-label targets correctly
                    inner_cv_splitter = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
                    
                    grid_search = GridSearchCV(
                        base_clf,
                        param_grids[name],
                        cv=inner_cv_splitter,
                        scoring='f1_micro',
                        n_jobs=n_jobs,
                        verbose=0
                    )
                    
                    # Fit on outer training fold
                    grid_search.fit(X_train_outer, y_train_outer)
                    
                    # Get best model from inner CV
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                    
                    # Evaluate on outer test fold (unseen during tuning)
                    y_pred = best_model.predict(X_test_outer)
                    
                    # Calculate metrics
                    f1_micro = f1_score(y_test_outer, y_pred, average='micro', zero_division=0)
                    f1_macro = f1_score(y_test_outer, y_pred, average='macro', zero_division=0)
                    accuracy = accuracy_score(y_test_outer, y_pred)
                    
                    outer_scores.append({
                        'f1_micro': f1_micro,
                        'f1_macro': f1_macro,
                        'accuracy': accuracy
                    })
                    best_params_per_fold.append(best_params)
                    
                    logger.info(f"    ✓ F1 Micro: {f1_micro:.4f}, F1 Macro: {f1_macro:.4f}")
                    logger.info(f"    Best params: {best_params}")
                    
                except Exception as e:
                    logger.error(f"    ✗ Outer fold {fold_num} failed: {e}")
                    continue
            
            # Aggregate results across outer folds
            if outer_scores:
                f1_micros = [s['f1_micro'] for s in outer_scores]
                f1_macros = [s['f1_macro'] for s in outer_scores]
                accuracies = [s['accuracy'] for s in outer_scores]
                
                self.nested_cv_results[name] = {
                    'outer_scores': outer_scores,
                    'mean_f1_micro': np.mean(f1_micros),
                    'std_f1_micro': np.std(f1_micros),
                    'mean_f1_macro': np.mean(f1_macros),
                    'std_f1_macro': np.std(f1_macros),
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'best_params_per_fold': best_params_per_fold,
                    'n_folds': len(outer_scores)
                }
                
                logger.info(f"\n  {'='*40}")
                logger.info(f"  NESTED CV SUMMARY for {name}")
                logger.info(f"  {'='*40}")
                logger.info(f"  Mean F1 Micro: {np.mean(f1_micros):.4f} (+/- {np.std(f1_micros):.4f})")
                logger.info(f"  Mean F1 Macro: {np.mean(f1_macros):.4f} (+/- {np.std(f1_macros):.4f})")
                logger.info(f"  Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})")
                logger.info(f"  {'='*40}")
            else:
                logger.warning(f"  ⚠️  No valid results for {name}")
        
        logger.info("\n" + "="*70)
        logger.info("NESTED CROSS-VALIDATION COMPLETED")
        logger.info("="*70 + "\n")
        
        return self.nested_cv_results
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, 
                            cv: int = 3) -> Dict[str, Any]:
        """
        Perform final hyperparameter tuning on full training set.
        
        This is done AFTER nested CV to get the best model for deployment.
        
        Args:
            X_train: Training feature matrix
            y_train: Training label matrix
            cv: Number of cross-validation folds (default: 3)
            
        Returns:
            Dict of tuned models
        """
        logger.info("\n" + "="*70)
        logger.info("FINAL HYPERPARAMETER TUNING (for deployment)")
        logger.info("="*70)
        
        param_grids = self.get_param_grids()
        # Use explicit KFold for tuning to handle multi-label targets correctly
        cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        
        for name, clf in self.models.items():
            if name not in param_grids:
                self.tuned_models[name] = clf
                self.best_params[name] = {}
                continue
            
            logger.info(f"\nTuning {name} on full training set...")
            
            try:
                grid_search = GridSearchCV(
                    clf, param_grids[name], cv=cv_splitter,
                    scoring='f1_micro', n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                
                self.tuned_models[name] = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                
                logger.info(f"  Best params: {grid_search.best_params_}")
                logger.info(f"  CV F1 Micro: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"  Tuning failed: {e}")
                self.tuned_models[name] = clf
                self.best_params[name] = {}
        
        return self.tuned_models
    
    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray, 
                          mlb_y: MultiLabelBinarizer) -> pd.DataFrame:
        """Train models and evaluate with comprehensive metrics including nested CV results."""
        logger.info("\n" + "="*70)
        logger.info("FINAL TRAINING AND EVALUATION")
        logger.info("="*70)
        
        # Create validation split for loss tracking
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        for name, clf in self.tuned_models.items():
            logger.info(f"\nTraining {name}...")
            
            try:
                # Train
                # Optimization: If model comes from successful grid search (has best_params), 
                # it is already fitted on X_train. Retraining is redundant.
                is_tuned = name in self.best_params and self.best_params[name]
                if not is_tuned:
                    clf.fit(X_train, y_train)
                else:
                    logger.info(f"  Using pre-fitted model from hyperparameter tuning")
                
                self.trained_models[name] = clf
                
                # Track training/validation loss only for MLP
                if name == "MLP Classifier":
                    logger.info(f"  Tracking loss for {name}...")
                    # Create new instance to track loss dynamics on split
                    temp_clf = MLPClassifier(**clf.get_params())
                    temp_clf.fit(X_train_split, y_train_split)
                    self.track_training_loss(name, temp_clf, X_train_split, 
                                           y_train_split, X_val_split, y_val_split)
                
                # Predict
                y_pred = clf.predict(X_test)
                
                # Calculate comprehensive metrics
                # Optimized: Calculate once
                precision, recall, f1_micro, _ = precision_recall_fscore_support(
                    y_test, y_pred, average='micro', zero_division=0
                )
                f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
                f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                acc = accuracy_score(y_test, y_pred)
                ham_loss = hamming_loss(y_test, y_pred)
                jac_score = jaccard_score(y_test, y_pred, average='micro', zero_division=0)
                
                # Calculate test loss if model supports predict_proba
                test_loss = np.nan
                try:
                    if hasattr(clf, 'predict_proba'):
                        y_pred_proba = clf.predict_proba(X_test)
                        test_loss = self.calculate_loss(y_test, y_pred_proba)
                except:
                    pass
                
                # Get nested CV results if available
                nested_cv_f1_micro = 0.0
                nested_cv_f1_std = 0.0
                if name in self.nested_cv_results:
                    nested_cv_f1_micro = self.nested_cv_results[name]['mean_f1_micro']
                    nested_cv_f1_std = self.nested_cv_results[name]['std_f1_micro']
                
                self.results.append({
                    "Model": name,
                    "Nested CV F1 Micro": nested_cv_f1_micro,
                    "Nested CV F1 Std": nested_cv_f1_std,
                    "Test Accuracy": acc,
                    "Test F1 Micro": f1_micro,
                    "Test F1 Macro": f1_macro,
                    "Test F1 Weighted": f1_weighted,
                    "Test Precision": precision,
                    "Test Recall": recall,
                    "Test Hamming Loss": ham_loss,
                    "Test Jaccard Score": jac_score,
                    "Test Loss": test_loss,
                    "Exact Match Ratio": acc,
                    "Best Params": str(self.best_params.get(name, {}))
                })
                
                logger.info(f"  Nested CV F1: {nested_cv_f1_micro:.4f} (+/- {nested_cv_f1_std:.4f})")
                logger.info(f"  Test F1 Micro: {f1_micro:.4f}")
                logger.info(f"  Test F1 Macro: {f1_macro:.4f}")
                logger.info(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
                if not np.isnan(test_loss):
                    logger.info(f"  Test Loss: {test_loss:.4f}")
                
                # Per-class report
                print(f"\n  Detailed Classification Report for {name}:")
                print(classification_report(y_test, y_pred, 
                                          target_names=mlb_y.classes_[:20],
                                          zero_division=0))
                
            except Exception as e:
                logger.error(f"  Training failed: {e}")
                continue
        
        return pd.DataFrame(self.results)


class Evaluator:
    """Enhanced model evaluation and visualization."""
    
    @staticmethod
    def plot_training_validation_loss(loss_history: Dict[str, Dict]):
        """Plot training and validation loss curves."""
        if not loss_history:
            logger.info("No loss history to plot")
            return
        
        n_models = len(loss_history)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(1, n_models, figsize=(7*n_models, 5))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, history) in enumerate(loss_history.items()):
            ax = axes[idx]
            
            epochs = history['epochs']
            train_loss = history['train_loss']
            val_loss = history['val_loss']
            
            ax.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', marker='o')
            ax.plot(epochs, val_loss, 'r--', linewidth=2, label='Validation Loss', marker='s')
            
            ax.set_xlabel('Epoch / Iteration', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title(f'{model_name}\nTraining vs Validation Loss', fontweight='bold', fontsize=12)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add minimum points
            min_train_idx = np.argmin(train_loss)
            min_val_idx = np.argmin(val_loss)
            ax.scatter(epochs[min_train_idx], train_loss[min_train_idx], 
                      color='blue', s=100, zorder=5, marker='*', 
                      label=f'Min Train: {train_loss[min_train_idx]:.4f}')
            ax.scatter(epochs[min_val_idx], val_loss[min_val_idx], 
                      color='red', s=100, zorder=5, marker='*',
                      label=f'Min Val: {val_loss[min_val_idx]:.4f}')
            ax.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = f'training_validation_loss_{timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved loss plot: {fig_path}")
        plt.show()
    
    @staticmethod
    def visualize_results(results_df: pd.DataFrame, loss_history: Dict[str, Dict] = None):
        """Create comprehensive visualizations including nested CV comparison."""
        logger.info("\nCreating visualizations...")
        
        # First plot: Training/Validation Loss
        if loss_history:
            Evaluator.plot_training_validation_loss(loss_history)
        
        # Main evaluation plots
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        
        # 1. Nested CV vs Test F1 Comparison
        if 'Nested CV F1 Micro' in results_df.columns:
            x = np.arange(len(results_df))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, results_df['Nested CV F1 Micro'], width, 
                          label='Nested CV F1 Micro', alpha=0.8, color='skyblue')
            axes[0, 0].bar(x + width/2, results_df['Test F1 Micro'], width, 
                          label='Test F1 Micro', alpha=0.8, color='coral')
            
            # Add error bars for nested CV
            axes[0, 0].errorbar(x - width/2, results_df['Nested CV F1 Micro'], 
                               yerr=results_df['Nested CV F1 Std'], 
                               fmt='none', ecolor='black', capsize=5)
            
            axes[0, 0].set_xlabel('Model')
            axes[0, 0].set_ylabel('F1 Micro Score')
            axes[0, 0].set_title('Nested CV vs Test Set Performance', fontweight='bold', fontsize=12)
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
            axes[0, 0].legend()
            axes[0, 0].grid(axis='y', alpha=0.3)
        else:
            axes[0, 0].text(0.5, 0.5, 'Run Nested CV to see comparison', 
                          ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('Nested CV vs Test Set Performance', fontweight='bold')
        
        # 2. Precision vs Recall
        axes[0, 1].scatter(results_df['Test Precision'], results_df['Test Recall'], 
                          s=200, alpha=0.6, c=range(len(results_df)), cmap='viridis')
        for idx, row in results_df.iterrows():
            axes[0, 1].annotate(row['Model'], 
                              (row['Test Precision'], row['Test Recall']),
                              fontsize=8, ha='center')
        axes[0, 1].set_xlabel('Precision')
        axes[0, 1].set_ylabel('Recall')
        axes[0, 1].set_title('Precision-Recall Tradeoff', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Model Ranking by Nested CV
        if 'Nested CV F1 Micro' in results_df.columns:
            sorted_results = results_df.sort_values('Nested CV F1 Micro', ascending=True)
            axes[1, 0].barh(sorted_results['Model'], sorted_results['Nested CV F1 Micro'], 
                           color=plt.cm.viridis(np.linspace(0.3, 1, len(sorted_results))))
            axes[1, 0].set_xlabel('Nested CV F1 Micro Score')
            axes[1, 0].set_title('Model Ranking by Nested CV', fontweight='bold')
        else:
            sorted_results = results_df.sort_values('Test F1 Micro', ascending=True)
            axes[1, 0].barh(sorted_results['Model'], sorted_results['Test F1 Micro'], 
                           color=plt.cm.viridis(np.linspace(0.3, 1, len(sorted_results))))
            axes[1, 0].set_xlabel('Test F1 Micro Score')
            axes[1, 0].set_title('Model Ranking by Test F1', fontweight='bold')
        
        # 4. Test Loss Comparison (for models that support it)
        results_with_loss = results_df[~results_df['Test Loss'].isna()]
        if len(results_with_loss) > 0:
            axes[1, 1].bar(results_with_loss['Model'], results_with_loss['Test Loss'], 
                          color='coral', alpha=0.7)
            axes[1, 1].set_ylabel('Test Loss')
            axes[1, 1].set_title('Test Loss Comparison', fontweight='bold')
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Loss Data Available', 
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Test Loss Comparison', fontweight='bold')
        
        # 5. Exact Match Ratio
        axes[2, 0].bar(results_df['Model'], results_df['Exact Match Ratio'], 
                      color='green', alpha=0.7)
        axes[2, 0].set_ylabel('Exact Match Ratio')
        axes[2, 0].set_title('Exact Match Ratio (Subset Accuracy)', fontweight='bold')
        axes[2, 0].tick_params(axis='x', rotation=45)
        
        # 6. Overall Performance Heatmap
        metrics_for_heatmap = ['Test F1 Micro', 'Test Precision', 'Test Recall', 
                              'Test Jaccard Score']
        heatmap_data = results_df[metrics_for_heatmap].T
        heatmap_data.columns = results_df['Model']
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', 
                   ax=axes[2, 1], cbar_kws={'label': 'Score'})
        axes[2, 1].set_title('Performance Metrics Heatmap', fontweight='bold')
        
        plt.tight_layout()
    
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = f'model_evaluation_{timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization: {fig_path}")
        plt.show()
        
        # Print best model (based on nested CV if available)
        if 'Nested CV F1 Micro' in results_df.columns and results_df['Nested CV F1 Micro'].max() > 0:
            best_model = results_df.loc[results_df['Nested CV F1 Micro'].idxmax()]
            print(f"\n{'='*70}")
            print(f"🏆 BEST MODEL (by Nested CV): {best_model['Model']}")
            print(f"{'='*70}")
            print(f"Nested CV F1 Micro: {best_model['Nested CV F1 Micro']:.4f} (+/- {best_model['Nested CV F1 Std']:.4f})")
        else:
            best_model = results_df.loc[results_df['Test F1 Micro'].idxmax()]
            print(f"\n{'='*70}")
            print(f"🏆 BEST MODEL (by Test F1): {best_model['Model']}")
            print(f"{'='*70}")
        
        print(f"Test F1 Micro:      {best_model['Test F1 Micro']:.4f}")
        print(f"Test F1 Macro:      {best_model['Test F1 Macro']:.4f}")
        print(f"Test Precision:     {best_model['Test Precision']:.4f}")
        print(f"Test Recall:        {best_model['Test Recall']:.4f}")
        print(f"Jaccard Score:      {best_model['Test Jaccard Score']:.4f}")
        print(f"Exact Match:        {best_model['Exact Match Ratio']:.4f}")
        if not np.isnan(best_model['Test Loss']):
            print(f"Test Loss:          {best_model['Test Loss']:.4f}")
        print(f"{'='*70}\n")
    
    @staticmethod
    def save_results_to_files(results_df: pd.DataFrame, best_params: Dict[str, Dict],
                             dataset_info: Dict[str, Any]) -> None:
        """Save all results with enhanced reporting."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV
        csv_path = f'results_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # 2. JSON with comprehensive info
        json_path = f'results_{timestamp}.json'
        json_data = {
            'timestamp': timestamp,
            'dataset_info': dataset_info,
            'best_parameters': best_params,
            'results': results_df.to_dict('records')
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # 3. Detailed report
        txt_path = f'results_report_{timestamp}.txt'
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-LABEL CLASSIFICATION - COMPREHENSIVE EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write("DATASET INFO\n" + "-"*80 + "\n")
            for key, value in dataset_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n\nMODEL RESULTS\n" + "-"*80 + "\n")
            f.write(results_df.to_string(index=False))
            
            f.write("\n\n\nBEST MODEL DETAILS\n" + "-"*80 + "\n")
            best = results_df.loc[results_df['Test F1 Micro'].idxmax()]
            f.write(f"Model: {best['Model']}\n")
            f.write(f"Test F1 Micro: {best['Test F1 Micro']:.4f}\n")
            f.write(f"Test F1 Macro: {best['Test F1 Macro']:.4f}\n")
            f.write(f"Precision: {best['Test Precision']:.4f}\n")
            f.write(f"Recall: {best['Test Recall']:.4f}\n")
            f.write(f"Parameters: {best['Best Params']}\n")
            
            f.write("\n\nRECOMMENDATIONS FOR IMPROVEMENT\n" + "-"*80 + "\n")
            f.write("1. Consider ensemble methods if individual models show complementary errors\n")
            f.write("2. Investigate feature engineering for low-performing labels\n")
            f.write("3. Apply SMOTE or other resampling for severe class imbalance\n")
            f.write("4. Consider deep learning approaches for complex patterns\n")
            f.write("5. Perform error analysis on misclassified samples\n")
        
        logger.info(f"Saved report: {txt_path}")
    
    @staticmethod
    def save_best_model(results_df: pd.DataFrame, trained_models: Dict, 
                       mlb_X: MultiLabelBinarizer, mlb_y: MultiLabelBinarizer, 
                       valid_dependencies_lower: set) -> str:
        """Save the best performing model with metadata."""
        best_model_name = results_df.loc[results_df['Test F1 Micro'].idxmax()]['Model']
        best_clf = trained_models[best_model_name]
        best_metrics = results_df.loc[results_df['Test F1 Micro'].idxmax()].to_dict()
        
        model_bundle = {
            'model': best_clf,
            'mlb_X': mlb_X,
            'mlb_y': mlb_y,
            'valid_dependencies': valid_dependencies_lower,
            'model_name': best_model_name,
            'metrics': best_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sklearn_version': '1.x',  # Add version info
            'python_version': '3.x'
        }
        
        save_path = f'best_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        joblib.dump(model_bundle, save_path)
        logger.info(f"Saved best model: {save_path}")
        
        return best_model_name


def main() -> Tuple[pd.DataFrame, Dict[str, Any], FeatureExtractor]:
    """Main execution function with nested cross-validation."""
    logger.info("\n" + "="*70)
    logger.info("MULTI-LABEL CLASSIFICATION WITH NESTED CROSS-VALIDATION")
    logger.info("="*70 + "\n")
    
    # Configuration
    MAIN_FILE = 'joss_all_with_dependency_labels1.csv'
    LOOKUP_FILE = 'dependency_counts_with_labels.csv'
    TEST_SIZE = 0.25
    RANDOM_STATE = 42
    APPLY_SCALING = False
    FEATURE_SELECTION_K = None  # DISABLED: binary sparse features don't benefit from chi2 selection
    USE_NESTED_CV = True  # ENABLED: Will perform Nested CV (might take ~5-10 mins)
    NESTED_OUTER_FOLDS = 3  # Optimized: 3 folds for performance estimation
    NESTED_INNER_FOLDS = 2  # Optimized: 2 folds for tuning
    
    try:
        # 1. Data Loading
        logger.info("STEP 1: DATA LOADING")
        loader = DataLoader(MAIN_FILE, LOOKUP_FILE)
        df = loader.load_and_clean_data()
        valid_dependencies, valid_labels = loader.load_lookup_data()
        
        # 2. Preprocessing
        logger.info("\nSTEP 2: PREPROCESSING")
        preprocessor = Preprocessor()
        df = preprocessor.preprocess_data(df, loader.dep_col, valid_dependencies, valid_labels)
        
        if df.empty:
            raise ValueError("No data after preprocessing")
        
        # 3. Feature Extraction
        logger.info("\nSTEP 3: FEATURE EXTRACTION")
        feature_extractor = FeatureExtractor()
        valid_dependencies_lower = {str(d).lower() for d in valid_dependencies}
        X, y = feature_extractor.extract_features(df, valid_dependencies_lower,
                                                  apply_scaling=APPLY_SCALING,
                                                  feature_selection_k=FEATURE_SELECTION_K)
        
        dataset_info = {
            'Total Samples': len(df),
            'Features': X.shape[1],
            'Labels': y.shape[1],
            'Avg Dependencies': round(df['filtered_deps'].apply(len).mean(), 2),
            'Avg Labels': round(df['filtered_labels'].apply(len).mean(), 2),
            'Train/Test Split': f"{int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)}",
            'Feature Scaling': APPLY_SCALING,
            'Feature Selection': FEATURE_SELECTION_K is not None,
            'Label Density': round((y.sum() / (y.shape[0] * y.shape[1])), 4)
        }
        
        # 4. EDA
        logger.info("\nSTEP 4: EXPLORATORY DATA ANALYSIS")
        EDA.analyze_and_visualize(df)
        
        # 5. Train/Test Split
        logger.info("\nSTEP 5: TRAIN/TEST SPLIT")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
        
        # Feature selection - DISABLED for binary dependency features
        # Binary sparse features lose information with chi2 selection.
        # The previous run without selection (F1=0.94) outperformed with selection (F1=0.89).
        # If needed in future, use per-label mutual information instead:
        #   from sklearn.feature_selection import mutual_info_classif
        
        # Check for class imbalance
        label_sums = y_train.sum(axis=0)
        logger.info(f"Min samples per label: {label_sums.min()}")
        logger.info(f"Max samples per label: {label_sums.max()}")
        
        # 6. Model Training
        logger.info("\nSTEP 6: MODEL INITIALIZATION")
        trainer = ModelTrainer()
        trainer.initialize_models()
        
        # 7. Nested Cross-Validation (OPTIONAL but RECOMMENDED)
        if USE_NESTED_CV:
            logger.info("\nSTEP 7: NESTED CROSS-VALIDATION")
            logger.info("This provides unbiased performance estimates...")
            nested_results = trainer.nested_cross_validation(
                X_train, y_train,
                outer_cv=NESTED_OUTER_FOLDS,
                inner_cv=NESTED_INNER_FOLDS,
                n_jobs=-1
            )
            
            # Display nested CV summary
            print("\n" + "="*70)
            print("NESTED CV PERFORMANCE SUMMARY")
            print("="*70)
            for model_name, results in nested_results.items():
                print(f"{model_name:25s} | F1: {results['mean_f1_micro']:.4f} (+/- {results['std_f1_micro']:.4f})")
            print("="*70 + "\n")
        else:
            logger.info("\nSTEP 7: SKIPPING NESTED CV (using simple train/test split)")
        
        # 8. Final Hyperparameter Tuning
        logger.info("\nSTEP 8: FINAL HYPERPARAMETER TUNING")
        trainer.tune_hyperparameters(X_train, y_train, cv=3)
        
        # 9. Final Evaluation on Test Set
        logger.info("\nSTEP 9: FINAL EVALUATION ON TEST SET")
        results_df = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, 
                                               feature_extractor.mlb_y)
        
        # 10. Visualization
        logger.info("\nSTEP 10: VISUALIZATION & ANALYSIS")
        print("\n" + "="*70)
        print("FINAL RESULTS SUMMARY")
        print("="*70)
        display_cols = ['Model', 'Nested CV F1 Micro', 'Test F1 Micro', 
                       'Test F1 Macro', 'Test Precision', 'Test Recall']
        print(results_df[display_cols].sort_values(by='Test F1 Micro', ascending=False).to_string(index=False))
        
        evaluator = Evaluator()
        evaluator.visualize_results(results_df, trainer.loss_history)
        
        # 11. Save Results
        logger.info("\nSTEP 11: SAVING RESULTS")
        evaluator.save_results_to_files(results_df, trainer.best_params, dataset_info)
        
        # 12. Save Best Model
        logger.info("\nSTEP 12: SAVING BEST MODEL")
        best_model_name = evaluator.save_best_model(
            results_df, trainer.trained_models, 
            feature_extractor.mlb_X, feature_extractor.mlb_y, 
            valid_dependencies_lower
        )
        
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*70)
        logger.info("\n KEY INSIGHTS:")
        logger.info("1. Nested CV scores show UNBIASED generalization performance")
        logger.info("2. Test scores may be optimistic if seen during any tuning")
        logger.info("3. If Nested CV << Test scores → possible overfitting to test set")
        logger.info("4. Use Nested CV scores for model selection/comparison")
        logger.info("\n RECOMMENDATIONS FOR IMPROVEMENT:")
        logger.info("1. Compare Nested CV vs Test scores for overfitting detection")
        logger.info("2. Analyze loss curves for overfitting/underfitting patterns")
        logger.info("3. Try ensemble methods combining top models from Nested CV")
        logger.info("4. Apply SMOTE/ADASYN for severe class imbalance")
        logger.info("5. Feature engineering: dependency co-occurrence, TF-IDF")
        logger.info("6. Error analysis: Focus on labels with high Nested CV variance")
        logger.info("="*70)
        
        return results_df, trainer.trained_models, feature_extractor
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    results_df, trained_models, feature_extractor = main()
