"""
Multi-Label Classification System for Software Dependencies
Modular implementation with class-based architecture

This module provides a complete pipeline for multi-label classification of software
dependencies, including data loading, preprocessing, feature extraction, model training,
and evaluation.

Classes:
    DataLoader: Handles loading and initial cleaning of data
    Preprocessor: Handles data preprocessing and filtering
    FeatureExtractor: Handles feature extraction and encoding
    EDA: Exploratory Data Analysis and Visualization
    ModelTrainer: Handles model training and cross-validation
    Evaluator: Handles model evaluation and visualization

Functions:
    main: Main execution function orchestrating the entire pipeline
"""

import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score, hamming_loss)
from sklearn.preprocessing import MultiLabelBinarizer
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

# Hardcoded list of JOSS IDs for testing (all 461 IDs)
LLM_TEST_JOSS_IDS = [
    1361, 4633, 7291, 7155, 4045, 6515, 5854, 4115, 6876, 5182, 4741, 786, 1412, 1247, 4987,
    5338, 4740, 3290, 177, 1322, 1169, 8855, 8504, 8456, 8481, 8454, 7471, 8061, 7477, 7668,
    7301, 7059, 6169, 6802, 6830, 6298, 6227, 6016, 5959, 5673, 5076, 5255, 5582, 5593, 5092,
    4724, 5269, 5415, 5300, 5149, 5126, 4953, 5028, 4774, 4879, 4866, 4622, 4455, 4212, 4383,
    4080, 3417, 2784, 2392, 2075, 1931, 1738, 1675, 1463, 979, 613, 53, 8037, 6733, 5867,
    5701, 4607, 4568, 3330, 4078, 3864, 3617, 3634, 3323, 2607, 1881, 1704, 1427, 1306, 1330,
    1143, 8816, 7061, 7887, 7769, 7009, 7363, 7830, 6274, 7358, 7031, 6756, 6912, 6241, 5756,
    5916, 5201, 6027, 5668, 6038, 5358, 5352, 5017, 4930, 5100, 4753, 4206, 4398, 4017, 3738,
    3199, 3646, 3695, 3434, 3351, 3221, 3127, 2973, 3021, 3089, 2641, 2107, 2804, 2006, 1896,
    1837, 785, 1108, 1026, 638, 427, 188, 7948, 8121, 7300, 7490, 7313, 6919, 6427, 6934,
    6860, 6120, 5984, 5810, 5511, 5422, 5510, 5410, 5372, 5093, 4868, 4996, 4880, 3829, 3308,
    2879, 2825, 2757, 2675, 2103, 1832, 1511, 1414, 1191, 855, 512, 2565, 989, 3896, 2449,
    1745, 7256, 6877, 6306, 5763, 8311, 6393, 6011, 4840, 4306, 3786, 3324, 7094, 247, 7971,
    2685, 6586, 3318, 2048, 2079, 1281, 662, 7869, 7917, 7522, 6768, 6667, 7201, 6507, 6624,
    6340, 6270, 5389, 4969, 4650, 4706, 4362, 4354, 3447, 3097, 2815, 2668, 2306, 1969, 2122,
    1352, 1341, 847, 754, 370, 8497, 6925, 7079, 6781, 6533, 6695, 6294, 6031, 5496, 5015,
    5329, 4655, 4762, 3428, 4265, 4014, 3658, 3782, 3168, 3191, 2594, 2554, 2049, 1820, 1229,
    774, 809, 5848, 5155, 8164, 5549, 5671, 4406, 3484, 1399, 1056, 6284, 7633, 6713, 2604,
    7852, 7360, 7305, 6971, 5567, 5225, 3723, 2241, 1983, 1525, 1596, 1576, 990, 259, 34,
    7312, 6495, 6469, 6058, 3865, 3521, 3313, 2940, 2180, 1910, 1272, 900, 5561, 2927, 1444,
    883, 5573, 4947, 4335, 2071, 7343, 6500, 5505, 3900, 1035, 550, 332, 135, 7697, 7932,
    6943, 6215, 5687, 5613, 5453, 3820, 2409, 2362, 2800, 2473, 1768, 1665, 1661, 1593, 911,
    657, 338, 653, 4937, 216, 6918, 2651, 1926, 6868, 5844, 5334, 4056, 3358, 1566, 765, 648,
    314, 8063, 7211, 7024, 5987, 5575, 5750, 4848, 3888, 3659, 1342, 8341, 7012, 6305, 6033,
    5879, 4015, 2285, 790, 176, 51, 4036, 6696, 1041, 5220, 7484, 6224, 5026, 1285, 4945,
    3440, 2367, 828, 2783, 7958, 5202, 5450, 6881, 280, 6948, 4843, 1097, 8391, 7719, 4693,
    2663, 8401, 7586, 5417, 4591, 1423, 6724, 4991, 4460, 4142, 4900, 4904, 3500, 3556, 1850,
    1761, 1486, 937, 4719, 7881, 7491, 2050, 3519, 373, 4369, 2290, 1063, 153, 8330, 3098,
    1693, 729, 597, 258, 4032, 1493, 7562, 6235, 2403, 2076, 695, 2917, 4458, 4304, 2752, 596,
    3389, 7104, 2331, 1207, 8564, 7946, 5572, 6181, 5174, 4886, 248, 6653, 4305, 2564, 2645,
    7478, 6638, 5194, 2976, 3040, 1584, 331, 281, 29
]


class DataLoader:
    """Handles loading and initial cleaning of data.
    
    This class is responsible for loading the main dataset and lookup table,
    performing basic cleaning operations, and identifying valid dependencies
    and labels from the lookup data.
    
    Attributes:
        main_file (str): Path to the main CSV file containing dependency data
        lookup_file (str): Path to the lookup CSV file with valid dependencies and labels
        df (pd.DataFrame): The loaded and cleaned main dataset
        dep_col (str): Name of the column containing dependencies
        valid_dependencies (set): Set of valid dependency names from lookup table
        valid_labels (set): Set of valid label names from lookup table
    """
    
    def __init__(self, main_file: str, lookup_file: str):
        """Initialize the DataLoader with file paths.
        
        Args:
            main_file: Path to the main CSV file containing dependency data
            lookup_file: Path to the lookup CSV file with valid dependencies and labels
            
        Raises:
            FileNotFoundError: If either file does not exist
        """
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
        """Load main dataset and perform basic cleaning.
        
        Loads the CSV file, identifies the dependency column name (handles both
        'dependencies_found' and 'dependecies_found'), and removes rows with
        empty dependencies or labels.
        
        Returns:
            pd.DataFrame: Cleaned dataframe with non-empty dependencies and labels
            
        Raises:
            ValueError: If neither 'dependencies_found' nor 'dependecies_found' column exists
            pd.errors.EmptyDataError: If the CSV file is empty
        """
        logger.info(f"Loading data from {self.main_file}...")
        
        try:
            self.df = pd.read_csv(self.main_file)
        except pd.errors.EmptyDataError as e:
            logger.error(f"Empty CSV file: {self.main_file}")
            raise
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
            raise ValueError(
                "Cannot find dependencies column. Expected 'dependencies_found' or 'dependecies_found'"
            )
        
        logger.info(f"Using dependency column: '{self.dep_col}'")
        
        # Validate required columns
        if 'dependency_labels' not in self.df.columns:
            raise ValueError("Missing required column: 'dependency_labels'")
        
        # Remove rows with empty dependencies or labels
        initial_rows = len(self.df)
        self.df = self.df.dropna(subset=[self.dep_col, 'dependency_labels'])
        self.df = self.df[self.df[self.dep_col].str.strip() != '']
        self.df = self.df[self.df['dependency_labels'].str.strip() != '']
        
        rows_removed = initial_rows - len(self.df)
        logger.info(f"Removed {rows_removed} rows with empty data")
        logger.info(f"Shape after cleaning: {self.df.shape}")
        
        return self.df
    
    def load_lookup_data(self) -> Tuple[set, set]:
        """Load lookup table to define vocabulary.
        
        Reads the lookup CSV file and extracts unique valid dependencies and labels
        that will be used to filter the main dataset.
        
        Returns:
            Tuple[set, set]: A tuple containing:
                - Set of valid dependency names
                - Set of valid label names
                
        Raises:
            ValueError: If required columns are missing from lookup file
        """
        logger.info(f"Loading lookup table from {self.lookup_file}...")
        
        try:
            lookup_df = pd.read_csv(self.lookup_file)
        except Exception as e:
            logger.error(f"Error loading lookup file: {e}")
            raise
        
        # Validate required columns
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
        
        if not self.valid_dependencies:
            logger.warning("No valid dependencies found in lookup file")
        if not self.valid_labels:
            logger.warning("No valid labels found in lookup file")
        
        return self.valid_dependencies, self.valid_labels
    
    def load_llm_joss_ids(self, llm_results_file: str) -> List[int]:
        """Load JOSS IDs from LLM evaluation results file.
        
        Args:
            llm_results_file: Path to the LLM results CSV with 'joss_id' column
            
        Returns:
            List[int]: List of JOSS IDs used in LLM evaluation
        """
        logger.info(f"Loading LLM JOSS IDs from {llm_results_file}...")
        
        if not os.path.exists(llm_results_file):
            raise FileNotFoundError(f"LLM results file not found: {llm_results_file}")
        
        llm_df = pd.read_csv(llm_results_file)
        
        if 'joss_id' not in llm_df.columns:
            raise ValueError("LLM results file must contain 'joss_id' column")
        
        joss_ids = llm_df['joss_id'].dropna().astype(int).tolist()
        logger.info(f"Found {len(joss_ids)} JOSS IDs from LLM evaluation")
        return joss_ids


class Preprocessor:
    """Handles data preprocessing and filtering.
    
    This class provides static methods for parsing string representations of
    dependencies and labels, filtering them against valid sets, and applying
    the complete preprocessing pipeline to the dataset.
    """
    
    def __init__(self):
        """Initialize the Preprocessor."""
        pass
    
    @staticmethod
    def parse_dependencies(dep_str: str) -> List[str]:
        """Parse dependency string to list.
        
        Handles both list-formatted strings (e.g., "['dep1', 'dep2']") and
        comma-separated strings (e.g., "dep1, dep2").
        
        Args:
            dep_str: String representation of dependencies
            
        Returns:
            List[str]: List of dependency names, empty list if parsing fails
        """
        if not isinstance(dep_str, str):
            return []
            
        try:
            dep_str = dep_str.strip()
            if dep_str.startswith('['):
                return ast.literal_eval(dep_str)
            elif dep_str:
                return [x.strip() for x in dep_str.split(',') if x.strip()]
            return []
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse dependencies: {dep_str[:50]}... Error: {e}")
            return []
    
    @staticmethod
    def parse_labels(label_str: str) -> List[str]:
        """Parse label string to list.
        
        Handles both list-formatted strings (e.g., "['Label1', 'Label2']") and
        comma-separated strings (e.g., "Label1, Label2").
        
        Args:
            label_str: String representation of labels
            
        Returns:
            List[str]: List of label names, empty list if parsing fails
        """
        if not isinstance(label_str, str):
            return []
            
        try:
            label_str = label_str.strip()
            if label_str.startswith('['):
                return ast.literal_eval(label_str)
            elif label_str:
                return [x.strip() for x in label_str.split(',') if x.strip()]
            return []
        except (ValueError, SyntaxError) as e:
            logger.debug(f"Failed to parse labels: {label_str[:50]}... Error: {e}")
            return []
    
    @staticmethod
    def filter_labels(labels: List[str], valid_set: set) -> List[str]:
        """Filter labels against valid set with case-insensitive matching.
        
        Attempts exact match first, then falls back to case-insensitive matching
        to handle variations in label capitalization.
        
        Args:
            labels: List of label strings to filter
            valid_set: Set of valid label strings
            
        Returns:
            List[str]: Filtered list of valid labels with duplicates removed
        """
        if not labels or not valid_set:
            return []
            
        filtered = []
        valid_set_lower = {v.lower(): v for v in valid_set}
        
        for label in labels:
            label_str = str(label).strip()
            if not label_str:
                continue
                
            # Exact match
            if label_str in valid_set:
                filtered.append(label_str)
            # Case-insensitive match
            elif label_str.lower() in valid_set_lower:
                filtered.append(valid_set_lower[label_str.lower()])
        
        return list(set(filtered))
    
    def preprocess_data(self, df: pd.DataFrame, dep_col: str, 
                       valid_dependencies: set, valid_labels: set) -> pd.DataFrame:
        """Complete preprocessing pipeline.
        
        Applies the full preprocessing workflow:
        1. Parse dependencies and labels from strings
        2. Normalize dependencies to lowercase
        3. Filter against valid sets
        4. Remove rows with no valid dependencies or labels
        
        Args:
            df: Input dataframe to preprocess
            dep_col: Name of the column containing dependencies
            valid_dependencies: Set of valid dependency names
            valid_labels: Set of valid label names
            
        Returns:
            pd.DataFrame: Preprocessed dataframe with filtered dependencies and labels
        """
        logger.info("Preprocessing data...")
        initial_rows = len(df)
        
        # Process dependencies
        df = df.copy()  # Avoid SettingWithCopyWarning
        df['dep_list'] = df[dep_col].apply(self.parse_dependencies)
        df['dep_list'] = df['dep_list'].apply(lambda x: [str(d).lower() for d in x])
        
        valid_dependencies_lower = {str(d).lower() for d in valid_dependencies}
        df['filtered_deps'] = df['dep_list'].apply(
            lambda x: [dep for dep in x if dep in valid_dependencies_lower]
        )
        
        # Process labels
        df['label_list'] = df['dependency_labels'].apply(self.parse_labels)
        valid_labels_norm = {str(l).strip() for l in valid_labels}
        df['filtered_labels'] = df['label_list'].apply(
            lambda x: self.filter_labels(x, valid_labels_norm)
        )
        
        # Filter out rows with no valid data
        df = df[df['filtered_deps'].map(len) > 0]
        df = df[df['filtered_labels'].map(len) > 0]
        
        rows_removed = initial_rows - len(df)
        logger.info(f"Removed {rows_removed} rows with no valid dependencies or labels")
        logger.info(f"Shape after preprocessing: {df.shape}")
        
        if df.empty:
            logger.warning("No data remaining after preprocessing!")
        
        return df


class FeatureExtractor:
    """Handles feature extraction and encoding.
    
    This class transforms preprocessed dependency and label lists into binary
    encoded matrices suitable for machine learning using MultiLabelBinarizer.
    
    Attributes:
        mlb_X (MultiLabelBinarizer): Encoder for dependency features
        mlb_y (MultiLabelBinarizer): Encoder for label targets
    """
    
    def __init__(self):
        """Initialize the FeatureExtractor."""
        self.mlb_X: Optional[MultiLabelBinarizer] = None
        self.mlb_y: Optional[MultiLabelBinarizer] = None
        
    def extract_features(self, df: pd.DataFrame, 
                        valid_dependencies_lower: set) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features using MultiLabelBinarizer.
        
        Converts lists of dependencies and labels into binary encoded matrices
        where each column represents the presence/absence of a specific
        dependency or label.
        
        Args:
            df: Dataframe with 'filtered_deps' and 'filtered_labels' columns
            valid_dependencies_lower: Set of valid lowercase dependency names
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - X: Binary feature matrix (samples x dependencies)
                - y: Binary label matrix (samples x labels)
                
        Raises:
            ValueError: If required columns are missing or data is empty
        """
        logger.info("Extracting features...")
        
        if df.empty:
            raise ValueError("Cannot extract features from empty dataframe")
        
        required_cols = ['filtered_deps', 'filtered_labels']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Encode dependencies (X)
        self.mlb_X = MultiLabelBinarizer(classes=sorted(list(valid_dependencies_lower)))
        X = self.mlb_X.fit_transform(df['filtered_deps'])
        
        # Encode labels (y)
        present_labels = set([l for sublist in df['filtered_labels'] for l in sublist])
        if not present_labels:
            raise ValueError("No labels found in data")
            
        self.mlb_y = MultiLabelBinarizer(classes=sorted(list(present_labels)))
        y = self.mlb_y.fit_transform(df['filtered_labels'])
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Label matrix shape: {y.shape}")
        logger.info(f"Number of features: {X.shape[1]}")
        logger.info(f"Number of labels: {y.shape[1]}")
        
        return X, y


class EDA:
    """Exploratory Data Analysis and Visualization.
    
    This class provides static methods for analyzing data distributions,
    identifying class imbalances, and creating comprehensive visualizations
    of the dataset characteristics.
    """
    
    @staticmethod
    def analyze_and_visualize(df: pd.DataFrame):
        """Perform comprehensive EDA.
        
        Analyzes the dataset to understand:
        - Label frequency distributions and class imbalance
        - Dependencies per repository
        - Labels per repository (label cardinality)
        
        Creates a 2x3 grid of visualizations showing:
        1. Top 20 most frequent labels
        2. Label frequency distribution curve
        3. Bottom 20 least frequent labels
        4. Histogram of label frequencies
        5. Distribution of labels per repository
        6. Distribution of dependencies per repository
        
        Args:
            df: Dataframe with 'filtered_deps' and 'filtered_labels' columns
        """
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Flatten lists for global stats
        all_labels = [label for sublist in df['filtered_labels'] for label in sublist]
        all_deps = [dep for sublist in df['filtered_deps'] for dep in sublist]
        
        labels_per_repo = df['filtered_labels'].apply(len)
        deps_per_repo = df['filtered_deps'].apply(len)
        
        label_counts = pd.Series(all_labels).value_counts()
        dep_counts = pd.Series(all_deps).value_counts()
        
        # Print statistics
        print(f"\n--- Label Statistics ---")
        print(f"Total Unique Labels: {len(label_counts)}")
        print(f"Most Frequent: '{label_counts.index[0]}' ({label_counts.iloc[0]} occurrences)")
        print(f"Least Frequent: '{label_counts.index[-1]}' ({label_counts.iloc[-1]} occurrences)")
        
        print(f"\n--- Per-Repository Statistics ---")
        print(f"Avg Labels per Repo: {labels_per_repo.mean():.2f}")
        print(f"Max Labels: {labels_per_repo.max()}")
        print(f"Avg Dependencies per Repo: {deps_per_repo.mean():.2f}")
        
        # Visualizations
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # Top 20 labels
        sns.barplot(x=label_counts.head(20).values, y=label_counts.head(20).index, 
                   palette='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Top 20 Most Frequent Labels')
        axes[0, 0].set_xlabel('Count')
        
        # Label frequency distribution
        axes[0, 1].plot(range(len(label_counts)), label_counts.values, 'b-', linewidth=2)
        axes[0, 1].fill_between(range(len(label_counts)), label_counts.values, alpha=0.3)
        axes[0, 1].set_title('Label Frequency Distribution (Imbalance)')
        axes[0, 1].set_xlabel('Label Rank')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bottom 20 labels
        bottom_20 = label_counts.tail(20).sort_values(ascending=True)
        sns.barplot(x=bottom_20.values, y=bottom_20.index, palette='rocket', ax=axes[0, 2])
        axes[0, 2].set_title('Bottom 20 Least Frequent Labels')
        
        # Histogram of label frequencies
        sns.histplot(label_counts, bins=30, kde=True, color='purple', ax=axes[1, 0])
        axes[1, 0].set_title('Label Frequency Histogram')
        axes[1, 0].set_yscale('log')
        
        # Labels per repository
        sns.histplot(labels_per_repo, bins=range(0, labels_per_repo.max() + 2), 
                    color='green', ax=axes[1, 1])
        axes[1, 1].set_title('Labels per Repository')
        axes[1, 1].axvline(labels_per_repo.mean(), color='red', linestyle='--', 
                          label=f'Mean: {labels_per_repo.mean():.1f}')
        axes[1, 1].legend()
        
        # Dependencies per repository
        sns.histplot(deps_per_repo, bins=30, kde=True, color='orange', ax=axes[1, 2])
        axes[1, 2].set_title('Dependencies per Repository')
        axes[1, 2].axvline(deps_per_repo.mean(), color='red', linestyle='--',
                          label=f'Mean: {deps_per_repo.mean():.1f}')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()


class ModelTrainer:
    """Handles model training and cross-validation.
    
    This class manages the initialization, training, and evaluation of multiple
    machine learning models including traditional ML models, gradient boosting
    models, and a custom ensemble.
    
    Attributes:
        models (Dict[str, Any]): Dictionary mapping model names to model instances
        results (List[Dict]): List of result dictionaries for each trained model
        trained_models (Dict[str, Any]): Dictionary of trained model instances
        best_params (Dict[str, Dict]): Dictionary storing best hyperparameters for each model
        tuned_models (Dict[str, Any]): Dictionary of hyperparameter-tuned model instances
        nested_cv_scores (Dict[str, List]): Dictionary storing nested CV scores for each model
    """
    
    def __init__(self):
        """Initialize the ModelTrainer."""
        self.models: Dict[str, Any] = {}
        self.results: List[Dict] = []
        self.trained_models: Dict[str, Any] = {}
        self.best_params: Dict[str, Dict] = {}
        self.tuned_models: Dict[str, Any] = {}
        self.nested_cv_scores: Dict[str, List] = {}
        
    def initialize_models(self):
        """Initialize all models with balanced class weights."""
        logger.info("\nInitializing models...")
        
        self.models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=100, class_weight='balanced', random_state=42
            ),
            "Logistic Regression": OneVsRestClassifier(
                LogisticRegression(solver='liblinear', class_weight='balanced', 
                                 random_state=42, max_iter=1000)
            ),
            "Linear SVC": OneVsRestClassifier(
                LinearSVC(class_weight='balanced', random_state=42, dual='auto', max_iter=2000)
            ),
            "MLP Classifier": MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
            "XGBoost": OneVsRestClassifier(
                xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                random_state=42, eval_metric='logloss', use_label_encoder=False)
            ),
            "LightGBM": OneVsRestClassifier(
                lgb.LGBMClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=42, verbose=-1, class_weight='balanced')
            ),
            "CatBoost": OneVsRestClassifier(
                CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1,
                                 random_state=42, verbose=False, auto_class_weights='Balanced')
            )
        }
        
        logger.info(f"Initialized {len(self.models)} models")
        
    def get_param_grids(self) -> Dict[str, Dict]:
        """Define minimal hyperparameter grids for fast tuning."""
        return {
            "Random Forest": {
                'n_estimators': [100, 200],
                'max_depth': [10, None]
            },
            "Logistic Regression": {
                'estimator__C': [0.1, 1.0, 10.0]
            },
            "Linear SVC": {
                'estimator__C': [0.1, 1.0, 10.0]
            },
            "MLP Classifier": {
                'hidden_layer_sizes': [(100,), (50, 50)],
                'alpha': [0.0001, 0.001]
            },
            "XGBoost": {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.1, 0.2]
            },
            "LightGBM": {
                'estimator__n_estimators': [100, 200],
                'estimator__max_depth': [5, 7],
                'estimator__learning_rate': [0.1, 0.2]
            },
            "CatBoost": {
                'estimator__iterations': [100, 200],
                'estimator__depth': [6, 8],
                'estimator__learning_rate': [0.1, 0.2]
            }
        }
    
    def nested_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                outer_cv: int = 3, inner_cv: int = 2) -> Dict[str, Dict]:
        """Perform nested cross-validation for unbiased performance estimation."""
        logger.info("\n" + "="*70)
        logger.info("NESTED CROSS-VALIDATION")
        logger.info("="*70)
        logger.info(f"Outer CV: {outer_cv} folds, Inner CV: {inner_cv} folds")
        
        param_grids = self.get_param_grids()
        outer_kfold = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
        nested_results = {}
        
        for name, clf in self.models.items():
            if name not in param_grids:
                continue
            
            logger.info(f"\nNested CV for {name}...")
            outer_scores = []
            
            for fold_num, (train_idx, test_idx) in enumerate(outer_kfold.split(X), 1):
                logger.info(f"  Fold {fold_num}/{outer_cv}")
                
                X_train_outer, X_test_outer = X[train_idx], X[test_idx]
                y_train_outer, y_test_outer = y[train_idx], y[test_idx]
                
                try:
                    grid_search = GridSearchCV(
                        clf, param_grids[name], cv=inner_cv,
                        scoring='f1_micro', n_jobs=-1, verbose=0
                    )
                    grid_search.fit(X_train_outer, y_train_outer)
                    
                    y_pred = grid_search.predict(X_test_outer)
                    f1_micro = f1_score(y_test_outer, y_pred, average='micro', zero_division=0)
                    outer_scores.append(f1_micro)
                    logger.info(f"    F1: {f1_micro:.4f}")
                    
                except Exception as e:
                    logger.error(f"    Failed: {e}")
                    continue
            
            if outer_scores:
                nested_results[name] = {
                    'outer_scores': outer_scores,
                    'mean_score': np.mean(outer_scores),
                    'std_score': np.std(outer_scores)
                }
                logger.info(f"  Mean: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores):.4f})")
        
        self.nested_cv_scores = nested_results
        logger.info("\nNested CV completed\n")
        return nested_results
    
    def tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, cv: int = 2) -> Dict[str, Any]:
        """Perform final hyperparameter tuning on full training set."""
        logger.info("\n" + "="*70)
        logger.info("HYPERPARAMETER TUNING")
        logger.info("="*70)
        
        param_grids = self.get_param_grids()
        
        for name, clf in self.models.items():
            if name not in param_grids:
                self.tuned_models[name] = clf
                self.best_params[name] = {}
                continue
            
            logger.info(f"\nTuning {name}...")
            
            try:
                grid_search = GridSearchCV(
                    clf, param_grids[name], cv=cv,
                    scoring='f1_micro', n_jobs=-1, verbose=0
                )
                grid_search.fit(X_train, y_train)
                
                self.tuned_models[name] = grid_search.best_estimator_
                self.best_params[name] = grid_search.best_params_
                
                logger.info(f"  Best params: {grid_search.best_params_}")
                logger.info(f"  CV F1: {grid_search.best_score_:.4f}")
                
            except Exception as e:
                logger.error(f"  Tuning failed: {e}")
                self.tuned_models[name] = clf
                self.best_params[name] = {}
        
        logger.info("\nTuning completed\n")
        return self.tuned_models
    
    def train_and_evaluate(self, X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray, 
                          mlb_y: MultiLabelBinarizer) -> pd.DataFrame:
        """Train tuned models and evaluate on test set."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING AND EVALUATION")
        logger.info("="*70)
        
        for name, clf in self.tuned_models.items():
            logger.info(f"\nTraining {name}...")
            
            # Get nested CV score
            nested_cv_mean = 0.0
            nested_cv_std = 0.0
            if name in self.nested_cv_scores:
                nested_cv_mean = self.nested_cv_scores[name]['mean_score']
                nested_cv_std = self.nested_cv_scores[name]['std_score']
            
            # Train and evaluate
            try:
                clf.fit(X_train, y_train)
                self.trained_models[name] = clf
                
                y_pred = clf.predict(X_test)
                
                self.results.append({
                    "Model": name,
                    "Test Accuracy": accuracy_score(y_test, y_pred),
                    "Test F1 Micro": f1_score(y_test, y_pred, average='micro', zero_division=0),
                    "Test F1 Macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
                    "Test Hamming Loss": hamming_loss(y_test, y_pred),
                    "Nested CV F1 (Mean)": nested_cv_mean,
                    "Nested CV F1 (Std)": nested_cv_std,
                    "Best Params": str(self.best_params.get(name, {}))
                })
                
                logger.info(f"  Test F1 Micro: {self.results[-1]['Test F1 Micro']:.4f}")
                
            except Exception as e:
                logger.error(f"  Training failed: {e}")
                continue
        
        return pd.DataFrame(self.results)


class Evaluator:
    """Handles model evaluation and visualization."""
    
    @staticmethod
    def visualize_results(results_df: pd.DataFrame):
        """Create comprehensive visualizations."""
        logger.info("\nCreating visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance comparison
        results_melted = results_df.melt(
            id_vars="Model",
            value_vars=["Test F1 Micro", "Test F1 Macro", "Test Accuracy"],
            var_name="Metric", value_name="Score"
        )
        sns.barplot(data=results_melted, x="Model", y="Score", hue="Metric", 
                   palette="viridis", ax=axes[0, 0])
        axes[0, 0].set_title("Model Performance Comparison", fontweight='bold')
        axes[0, 0].set_ylim(0, 1.0)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. F1 Micro ranking
        sorted_results = results_df.sort_values('Test F1 Micro', ascending=True)
        axes[0, 1].barh(sorted_results['Model'], sorted_results['Test F1 Micro'], 
                       color=plt.cm.viridis(np.linspace(0.3, 1, len(sorted_results))))
        axes[0, 1].set_xlabel('Test F1 Micro Score')
        axes[0, 1].set_title('Model Ranking', fontweight='bold')
        
        # 3. Hamming Loss
        sorted_by_hl = results_df.sort_values('Test Hamming Loss')
        axes[1, 0].barh(sorted_by_hl['Model'], sorted_by_hl['Test Hamming Loss'], color='coral')
        axes[1, 0].set_xlabel('Hamming Loss (lower is better)')
        axes[1, 0].set_title('Hamming Loss Comparison', fontweight='bold')
        
        # 4. Nested CV vs Test
        comp_df = results_df[['Model', 'Nested CV F1 (Mean)', 'Test F1 Micro']].copy()
        melt_df = comp_df.melt(id_vars='Model', var_name='Metric', value_name='F1')
        sns.barplot(data=melt_df, x='Model', y='F1', hue='Metric', palette='muted', ax=axes[1, 1])
        axes[1, 1].set_title('CV vs Test Performance', fontweight='bold')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fig_path = f'model_comparison_{timestamp}.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization: {fig_path}")
        plt.show()
        
        # Print best model
        best_model = results_df.loc[results_df['Test F1 Micro'].idxmax()]
        print(f"\n{'='*50}")
        print(f"🏆 BEST MODEL: {best_model['Model']}")
        print(f"{'='*50}")
        print(f"Test F1 Micro:  {best_model['Test F1 Micro']:.4f}")
        print(f"Test F1 Macro:  {best_model['Test F1 Macro']:.4f}")
        print(f"Test Accuracy:  {best_model['Test Accuracy']:.4f}")
        print(f"Nested CV F1:   {best_model['Nested CV F1 (Mean)']:.4f}")
        print(f"{'='*50}\n")
    
    @staticmethod
    def save_results_to_files(results_df: pd.DataFrame, best_params: Dict[str, Dict],
                             trainer: Any, dataset_info: Dict[str, Any]) -> None:
        """Save all results to multiple file formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. CSV
        csv_path = f'model_results_{timestamp}.csv'
        results_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV: {csv_path}")
        
        # 2. JSON
        json_path = f'results_{timestamp}.json'
        json_data = {
            'best_parameters': best_params,
            'nested_cv_scores': {
                name: {
                    'mean': float(scores['mean_score']),
                    'std': float(scores['std_score']),
                    'scores': [float(s) for s in scores['outer_scores']]
                }
                for name, scores in trainer.nested_cv_scores.items()
            },
            'dataset_info': dataset_info
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        logger.info(f"Saved JSON: {json_path}")
        
        # 3. Text report
        txt_path = f'results_report_{timestamp}.txt'
        with open(txt_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("MULTI-LABEL CLASSIFICATION RESULTS\n")
            f.write("="*80 + "\n\n")
            
            f.write("DATASET INFO\n" + "-"*80 + "\n")
            for key, value in dataset_info.items():
                f.write(f"{key}: {value}\n")
            
            f.write("\n\nRESULTS\n" + "-"*80 + "\n")
            f.write(results_df.to_string(index=False))
            
            f.write("\n\n\nBEST MODEL\n" + "-"*80 + "\n")
            best = results_df.loc[results_df['Test F1 Micro'].idxmax()]
            f.write(f"Model: {best['Model']}\n")
            f.write(f"Test F1 Micro: {best['Test F1 Micro']:.4f}\n")
            f.write(f"Parameters: {best['Best Params']}\n")
        
        logger.info(f"Saved report: {txt_path}")
    
    @staticmethod
    def save_best_model(results_df: pd.DataFrame, trained_models: Dict, 
                       mlb_X: MultiLabelBinarizer, mlb_y: MultiLabelBinarizer, 
                       valid_dependencies_lower: set) -> str:
        """Save the best performing model."""
        best_model_name = results_df.loc[results_df['Test F1 Micro'].idxmax()]['Model']
        best_clf = trained_models[best_model_name]
        
        model_bundle = {
            'model': best_clf,
            'mlb_X': mlb_X,
            'mlb_y': mlb_y,
            'valid_dependencies': valid_dependencies_lower,
            'model_name': best_model_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        save_path = 'best_model.joblib'
        joblib.dump(model_bundle, save_path)
        logger.info(f"Saved best model: {save_path}")
        
        return best_model_name


def split_by_joss_ids(X: np.ndarray, y: np.ndarray, df: pd.DataFrame,
                      test_joss_ids: List[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test based on JOSS IDs.
    
    Ensures the same papers evaluated by the LLM form the holdout test set,
    enabling unbiased comparison between ML and LLM predictions.
    
    Args:
        X: Feature matrix
        y: Label matrix
        df: Preprocessed dataframe (must have 'joss_id' or index-aligned with X/y)
        test_joss_ids: List of JOSS IDs to use as test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Detect JOSS ID column
    id_col = None
    for col in ['joss_id', 'id', 'paper_id', 'ID']:
        if col in df.columns:
            id_col = col
            break
    
    if id_col is None:
        logger.warning("No JOSS ID column found in dataframe. Falling back to random split.")
        return None, None, None, None
    
    df_reset = df.reset_index(drop=True)
    test_joss_ids_set = set(test_joss_ids)
    
    test_mask = df_reset[id_col].astype(int).isin(test_joss_ids_set)
    train_mask = ~test_mask
    
    matched = test_mask.sum()
    logger.info(f"Matched {matched}/{len(test_joss_ids)} LLM JOSS IDs in main dataset")
    
    if matched == 0:
        logger.warning("No JOSS IDs matched. Falling back to random split.")
        return None, None, None, None
    
    train_idx = df_reset.index[train_mask].tolist()
    test_idx = df_reset.index[test_mask].tolist()
    
    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    logger.info(f"JOSS ID-based split: Train={len(train_idx)}, Test={len(test_idx)}")
    
    # Print the actual matched test JOSS IDs
    matched_test_ids = sorted(df_reset.loc[test_idx, id_col].astype(int).tolist())
    print(f"\n{'='*60}")
    print(f"TEST SET JOSS IDs ({len(matched_test_ids)} papers):")
    print(matched_test_ids)
    print(f"{'='*60}\n")
    
    return X_train, X_test, y_train, y_test


def main() -> Tuple[pd.DataFrame, Dict[str, Any], FeatureExtractor]:
    """Main execution function orchestrating the entire pipeline."""
    logger.info("\n" + "="*70)
    logger.info("MULTI-LABEL CLASSIFICATION PIPELINE")
    logger.info("="*70 + "\n")
    
    # Configuration
    MAIN_FILE = 'joss_all_with_dependency_labels1.csv'
    LOOKUP_FILE = 'dependency_counts_with_labels.csv'
    LLM_RESULTS_FILE = 'llm_results.csv'  # Optional fallback
    TEST_SIZE = 0.30
    RANDOM_STATE = 42
    PERFORM_NESTED_CV = True
    
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
        X, y = feature_extractor.extract_features(df, valid_dependencies_lower)
        
        # 4. EDA
        logger.info("\nSTEP 4: EDA")
        EDA.analyze_and_visualize(df)
        
        # 5. Train/Test Split
        logger.info("\nSTEP 5: TRAIN/TEST SPLIT")
        logger.info(f"Using hardcoded LLM JOSS IDs ({len(LLM_TEST_JOSS_IDS)} IDs) as holdout test set...")
        
        X_train, X_test, y_train, y_test = split_by_joss_ids(X, y, df, LLM_TEST_JOSS_IDS)
        split_method = "joss_id"
        
        if X_train is None:
            logger.warning("No JOSS IDs matched — falling back to random split")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )
            split_method = "random"
        else:
            logger.info(f"✅ JOSS ID split: TRAIN={len(X_train)}, TEST={len(X_test)} "
                       f"({len(X_test)/len(X)*100:.1f}% test)")
        
        dataset_info = {
            'Total Samples': len(df),
            'Features': X.shape[1],
            'Labels': y.shape[1],
            'Train Samples': len(X_train),
            'Test Samples': len(X_test),
            'Split Method': split_method,
            'Avg Dependencies': df['filtered_deps'].apply(len).mean(),
            'Avg Labels': df['filtered_labels'].apply(len).mean(),
            'Train/Test Split': f"{len(X_train)}/{len(X_test)}"
        }
        
        # 6. Model Training
        logger.info("\nSTEP 6: MODEL TRAINING")
        trainer = ModelTrainer()
        trainer.initialize_models()
        
        # 7. Nested CV
        if PERFORM_NESTED_CV:
            logger.info("\nSTEP 7: NESTED CV")
            trainer.nested_cross_validation(X_train, y_train, outer_cv=3, inner_cv=2)
        
        # 8. Hyperparameter Tuning
        logger.info("\nSTEP 8: HYPERPARAMETER TUNING")
        trainer.tune_hyperparameters(X_train, y_train, cv=2)
        
        # 9. Final Training & Evaluation
        logger.info("\nSTEP 9: EVALUATION")
        results_df = trainer.train_and_evaluate(X_train, X_test, y_train, y_test, feature_extractor.mlb_y)
        
        # 10. Visualization
        logger.info("\nSTEP 10: VISUALIZATION")
        print(f"\nSplit Method: {split_method.upper()}")
        print("\nResults:\n", results_df[['Model', 'Test F1 Micro', 'Test F1 Macro', 'Test Accuracy']].to_string(index=False))
        
        evaluator = Evaluator()
        evaluator.visualize_results(results_df)
        
        # 11. Save Results
        logger.info("\nSTEP 11: SAVING RESULTS")
        evaluator.save_results_to_files(results_df, trainer.best_params, trainer, dataset_info)
        
        # 12. Save Best Model
        logger.info("\nSTEP 12: SAVING BEST MODEL")
        best_model_name = evaluator.save_best_model(
            results_df, trainer.trained_models,
            feature_extractor.mlb_X, feature_extractor.mlb_y,
            valid_dependencies_lower
        )
        
        logger.info("\n" + "="*70)
        logger.info(f"PIPELINE COMPLETED | Split: {split_method}")
        logger.info("="*70)
        
        return results_df, trainer.trained_models, feature_extractor
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    results_df, trained_models, feature_extractor = main()
