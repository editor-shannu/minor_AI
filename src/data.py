import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import yaml
from pathlib import Path

class DataLoader(ABC):
    """Abstract base class for different data sources."""
    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass

class CSVDataLoader(DataLoader):
    def __init__(self, file_path: str):
        self.file_path = file_path
    
    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)

class DiseaseSalesData:
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.disease_loader = CSVDataLoader(self.config['disease_data_path'])
        self.sales_loader = CSVDataLoader(self.config['sales_data_path'])
    
    def _load_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_and_merge(self) -> pd.DataFrame:
        """CHANGE HERE: Update column names to match your datasets"""
        disease_df = self.disease_loader.load()
        sales_df = self.sales_loader.load()
        
        merged_df = pd.merge(
            disease_df, sales_df, 
            on=['date', 'location'],  # CHANGE: Your merge keys
            how='inner'
        )
        return self.preprocess(merged_df)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # CHANGE HERE: Custom feature engineering
        df['disease_trend'] = df.groupby('location')['cases'].pct_change().fillna(0)
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
        
        return df.fillna(0)
