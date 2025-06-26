import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional, Dict, List, Union
import warnings

warnings.filterwarnings('ignore')

class DataCleaningAgent:
    def __init__(self, 
                 missing_threshold: float = 0.5,
                 outlier_method: str = 'iqr',
                 outlier_threshold: float = 1.5,
                 handle_categorical: bool = True):
        """
        Initialize the Data Cleaning Agent with configurable parameters.
        
        Args:
            missing_threshold (float): Maximum ratio of missing values allowed before removing column
            outlier_method (str): Method for detecting outliers ('iqr' or 'zscore')
            outlier_threshold (float): Threshold for outlier detection
            handle_categorical (bool): Whether to encode categorical variables
        """
        self.missing_threshold = missing_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.handle_categorical = handle_categorical
        self.label_encoders = {}
        self.column_stats = {}
    
    def _analyze_column_relationships(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Analyze relationships between columns to determine which columns are most correlated.
        """
        relationships = {}
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        if len(numerical_cols) > 1:
            corr_matrix = df[numerical_cols].corr().abs()
            for col in numerical_cols:
                # Get top 3 correlated features excluding self
                related_cols = corr_matrix[col].sort_values(ascending=False)[1:4]
                relationships[col] = list(related_cols.index)
        
        return relationships

    def _determine_imputation_strategy(self, df: pd.DataFrame, column: str, missing_ratio: float) -> Dict:
        """
        Determine the best imputation strategy for a given column based on various factors.
        """
        if missing_ratio >= self.missing_threshold:
            return {'strategy': 'drop_column', 'ratio': missing_ratio}
        
        column_data = df[column]
        dtype = column_data.dtype
        non_null_count = column_data.count()
        unique_ratio = column_data.nunique() / non_null_count if non_null_count > 0 else 0
        
        strategy_info = {
            'ratio': missing_ratio,
            'dtype': str(dtype),
            'unique_ratio': unique_ratio
        }
        
        # Rule-based strategy selection
        if dtype in ['int64', 'float64']:
            if missing_ratio < 0.05:  # Very few missing values
                if column_data.skew() > 1:
                    strategy_info['strategy'] = 'median'
                else:
                    strategy_info['strategy'] = 'mean'
            elif unique_ratio > 0.9:  # High cardinality
                strategy_info['strategy'] = 'knn'
            elif missing_ratio < 0.3:  # Moderate missing values
                strategy_info['strategy'] = 'iterative'
            else:  # Many missing values
                strategy_info['strategy'] = 'random_forest'
        else:  # Categorical
            if unique_ratio < 0.05:  # Very few unique values
                strategy_info['strategy'] = 'most_frequent'
            elif missing_ratio < 0.1:
                strategy_info['strategy'] = 'most_frequent'
            else:
                strategy_info['strategy'] = 'random_forest_classifier'
        
        return strategy_info

    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """
        Analyze missing values and determine the best strategy for each column.
        """
        missing_stats = {}
        relationships = self._analyze_column_relationships(df)
        
        for column in df.columns:
            missing_ratio = df[column].isnull().mean()
            if missing_ratio > 0:
                strategy_info = self._determine_imputation_strategy(df, column, missing_ratio)
                strategy_info['related_columns'] = relationships.get(column, [])
                missing_stats[column] = strategy_info
        print('#' * 100)
        print(missing_stats)
        
        return missing_stats
    
    def _impute_with_random_forest(self, df: pd.DataFrame, column: str, related_columns: List[str]) -> pd.Series:
        """
        Impute missing values using Random Forest.
        """
        df_temp = df.copy()
        missing_mask = df_temp[column].isnull()
        
        # If no related columns provided, use all other numeric columns
        if not related_columns:
            related_columns = [col for col in df_temp.select_dtypes(include=['int64', 'float64']).columns 
                             if col != column and df_temp[col].isnull().sum() == 0]
        
        if not related_columns:
            return df_temp[column]
        
        # Prepare training data
        X_train = df_temp.loc[~missing_mask, related_columns]
        y_train = df_temp.loc[~missing_mask, column]
        X_missing = df_temp.loc[missing_mask, related_columns]
        
        if df_temp[column].dtype in ['int64', 'float64']:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        df_temp.loc[missing_mask, column] = model.predict(X_missing)
        
        return df_temp[column]

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values based on the analysis results.
        """
        df_clean = df.copy()
        missing_stats = self._analyze_missing_values(df)
        
        # Drop columns with too many missing values
        columns_to_drop = [col for col, stats in missing_stats.items() 
                          if stats['strategy'] == 'drop_column']
        print("columns_to_drop:", columns_to_drop)
        df_clean = df_clean.drop(columns=columns_to_drop)
        
        # First pass: handle simple imputation strategies
        simple_imputation_columns = {
            'mean': [],
            'median': [],
            'most_frequent': []
        }
        
        for column, stats in missing_stats.items():
            if stats['strategy'] in simple_imputation_columns:
                simple_imputation_columns[stats['strategy']].append(column)
        
        # Apply simple imputation strategies in batches
        for strategy, columns in simple_imputation_columns.items():
            if columns:
                imputer = SimpleImputer(strategy=strategy)
                df_clean[columns] = imputer.fit_transform(df_clean[columns])
        
        # Second pass: handle more complex imputation strategies
        for column, stats in missing_stats.items():
            if column in df_clean.columns and df_clean[column].isnull().any():
                if stats['strategy'] == 'knn':
                    # KNN imputation
                    imputer = KNNImputer(n_neighbors=5)
                    related_cols = stats['related_columns']
                    if related_cols:
                        cols_to_use = [column] + related_cols
                        scaler = StandardScaler()
                        df_scaled = pd.DataFrame(
                            scaler.fit_transform(df_clean[cols_to_use]),
                            columns=cols_to_use
                        )
                        df_clean[column] = imputer.fit_transform(df_scaled)[:, 0]
                    else:
                        df_clean[column] = imputer.fit_transform(df_clean[[column]])
                
                elif stats['strategy'] in ['random_forest', 'random_forest_classifier']:
                    # Random Forest imputation
                    df_clean[column] = self._impute_with_random_forest(
                        df_clean, 
                        column,
                        stats['related_columns']
                    )
                
                elif stats['strategy'] == 'iterative':
                    # Iterative imputation using related columns
                    related_cols = stats['related_columns']
                    if related_cols:
                        initial_imputer = SimpleImputer(strategy='mean')
                        df_clean[column] = initial_imputer.fit_transform(df_clean[[column]])
                        df_clean[column] = self._impute_with_random_forest(
                            df_clean,
                            column,
                            related_cols
                        )
        
        return df_clean
    
    def _detect_outliers(self, series: pd.Series) -> np.ndarray:
        """
        Detect outliers using the specified method.
        """
        if self.outlier_method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.outlier_threshold * IQR
            upper_bound = Q3 + self.outlier_threshold * IQR
            outliers = (series < lower_bound) | (series > upper_bound)
        else:  # zscore
            z_scores = np.abs((series - series.mean()) / series.std())
            outliers = z_scores > self.outlier_threshold
            
        return outliers
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers in numerical columns.
        """
        df_clean = df.copy()
        numerical_columns = df_clean.select_dtypes(include=['int64', 'float64']).columns
        
        for column in numerical_columns:
            outliers = self._detect_outliers(df_clean[column])
            if outliers.any():
                # Store outlier stats
                self.column_stats[column] = {
                    'outliers_detected': outliers.sum(),
                    'outlier_ratio': outliers.mean()
                }
                
                # Rule-based handling
                if outliers.mean() < 0.05:  # Less than 5% outliers
                    # Remove outliers
                    df_clean.loc[outliers, column] = np.nan
                else:
                    # Cap outliers
                    Q1 = df_clean[column].quantile(0.25)
                    Q3 = df_clean[column].quantile(0.75)
                    IQR = Q3 - Q1
                    df_clean[column] = df_clean[column].clip(
                        lower=Q1 - 1.5 * IQR,
                        upper=Q3 + 1.5 * IQR
                    )
        
        return df_clean
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle duplicate records.
        """
        # First, check for exact duplicates
        exact_duplicates = df.duplicated()
        df_clean = df.drop_duplicates()
        
        # Store duplicate stats
        self.column_stats['duplicates'] = {
            'exact_duplicates_removed': exact_duplicates.sum()
        }
        
        return df_clean
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables using LabelEncoder.
        """
        df_clean = df.copy()
        categorical_columns = df_clean.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if column not in self.label_encoders:
                self.label_encoders[column] = LabelEncoder()
                df_clean[column] = self.label_encoders[column].fit_transform(df_clean[column].astype(str))
        
        return df_clean
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the input DataFrame using all specified methods.
        
        Args:
            df (pd.DataFrame): Input DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        # Reset stats
        self.column_stats = {}
        
        # Handle missing values
        df_clean = self._handle_missing_values(df)
        
        # Handle outliers
        #TODO: make boolean switch
        df_clean = self._handle_outliers(df_clean)
        
        # Handle duplicates
        df_clean = self._handle_duplicates(df_clean)
        
        # Handle categorical variables if specified
        # if self.handle_categorical:
        #     df_clean = self._encode_categorical(df_clean)
        
        return df_clean
    
    def get_cleaning_stats(self) -> Dict:
        """
        Get statistics about the cleaning process.
        """
        return self.column_stats
    
    def inverse_transform_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Inverse transform encoded categorical variables back to original values.
        
        Args:
            df (pd.DataFrame): DataFrame with encoded values
            columns (List[str]): List of columns to inverse transform
            
        Returns:
            pd.DataFrame: DataFrame with decoded values
        """
        df_decoded = df.copy()
        for column in columns:
            if column in self.label_encoders:
                df_decoded[column] = self.label_encoders[column].inverse_transform(df[column])
        
        return df_decoded

# # Example usage:
# df = pd.read_csv('datasets/Smoking/train.csv')
# # Initialize the agent
# cleaner = DataCleaningAgent(
#     missing_threshold=0.8,
#     outlier_method='iqr',
#     outlier_threshold=1.5,
#     handle_categorical=True
# )

# # Clean the data
# df_clean = cleaner.clean(df)

# # Get cleaning statistics
# stats = cleaner.get_cleaning_stats()
