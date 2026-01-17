# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# import joblib
# import pandas as pd
# import numpy as np
# from abc import ABC, abstractmethod
# from typing import Dict, Tuple, Any
# from pathlib import Path

# class Predictor(ABC):
#     @abstractmethod
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         pass
    
#     @abstractmethod
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         pass

# class RandomForestPredictor(Predictor):
#     def __init__(self):
#         self.model = RandomForestRegressor(n_estimators=100, random_state=42)
#         self.scaler = StandardScaler()
    
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         X_scaled = self.scaler.fit_transform(X)
#         self.model.fit(X_scaled, y)
    
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         X_scaled = self.scaler.transform(X)
#         return self.model.predict(X_scaled)

# class MedicineSalesModel:
#     def __init__(self, model_type: str = 'rf'):
#         self.model_type = model_type
#         self.predictor = RandomForestPredictor()
#         self.feature_columns = None
#         self.label_encoders = {}
#         self.is_trained = False
    
#     def prepare_features(self, df: pd.DataFrame, predict_mode=False) -> Tuple[np.ndarray, np.ndarray]:
#         feature_cols = ['cases', 'disease_trend', 'month', 'year']
#         target_col = 'sales_volume'
        
#         df_work = df.copy()
        
#         if 'date' in df_work.columns:
#             df_work['date'] = pd.to_datetime(df_work['date'])
#             df_work['month'] = df_work['date'].dt.month
#             df_work['year'] = df_work['date'].dt.year
        
#         if 'disease_trend' not in df_work.columns and 'cases' in df_work.columns:
#             df_work['disease_trend'] = df_work.groupby('location')['cases'].pct_change().fillna(0)
        
#         if 'location' in df_work.columns:
#             if predict_mode and self.is_trained:
#                 le = self.label_encoders.get('location', LabelEncoder())
#                 try:
#                     df_work['location_encoded'] = le.transform(df_work['location'].astype(str))
#                 except ValueError:
#                     known_locations = {v: i for i, v in enumerate(le.classes_)}
#                     df_work['location_encoded'] = df_work['location'].astype(str).map(known_locations).fillna(0).astype(int)
#             else:
#                 le = LabelEncoder()
#                 df_work['location_encoded'] = le.fit_transform(df_work['location'].astype(str))
#                 self.label_encoders['location'] = le
#             feature_cols.append('location_encoded')
        
#         available_features = [col for col in feature_cols if col in df_work.columns]
#         self.feature_columns = available_features
        
#         X = df_work[available_features].fillna(0).values
        
#         if not predict_mode:
#             if target_col not in df_work.columns:
#                 raise ValueError(f"Target '{target_col}' missing")
#             y = df_work[target_col].values
#             return X, y
#         return X, None
    
#     def train(self, df: pd.DataFrame):
#         X, y = self.prepare_features(df, predict_mode=False)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         self.predictor.fit(X_train, y_train)
#         self.is_trained = True
        
#         train_score = self.predictor.model.score(X_train, y_train)
#         test_score = self.predictor.model.score(X_test, y_test)
        
#         return {
#             'train_r2': round(train_score, 3),
#             'test_r2': round(test_score, 3),
#             'features': self.feature_columns,
#             'n_samples': len(df)
#         }
    
#     def predict(self, df: pd.DataFrame) -> np.ndarray:
#         if not self.is_trained:
#             raise ValueError("Model must be trained first!")
#         X, _ = self.prepare_features(df, predict_mode=True)
#         return self.predictor.predict(X)
    
#     def save(self, path: str):
#         Path(path).parent.mkdir(exist_ok=True, parents=True)
#         joblib.dump(self, path)
    
#     @classmethod
#     def load(cls, path: str):
#         return joblib.load(path) 

import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from abc import ABC, abstractmethod

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.cluster import KMeans, DBSCAN

from sklearn.metrics import r2_score, accuracy_score


# -----------------------------------
# Base predictor interface (OOP)
# -----------------------------------
class Predictor(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass


# -----------------------------------
# Regression predictors
# -----------------------------------
class RandomForestPredictor(Predictor):
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


class LinearRegressionPredictor(Predictor):
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = LinearRegression()

    def fit(self, X: np.ndarray, y: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return self.model.predict(Xs)


# ======================================================
# ✅ MAIN MODEL CLASS
# ======================================================
class MedicineSalesModel:
    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type.lower()

        self.feature_columns: Optional[list] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.is_trained: bool = False

        # regression
        self.predictor: Optional[Predictor] = None

        # classification
        self.classifier: Optional[LogisticRegression] = None
        self.cls_scaler = StandardScaler()

        # clustering
        self.clusterer = None

        # anomaly
        self.anomaly_detector = None

        # init based on model_type
        if self.model_type == "rf":
            self.predictor = RandomForestPredictor()

        elif self.model_type == "linear":
            self.predictor = LinearRegressionPredictor()

        elif self.model_type == "logistic":
            self.classifier = LogisticRegression(max_iter=2000)

        elif self.model_type == "kmeans":
            self.clusterer = KMeans(n_clusters=3, random_state=42, n_init="auto")

        elif self.model_type == "dbscan":
            self.clusterer = DBSCAN(eps=0.5, min_samples=5)

        elif self.model_type == "isolation":
            self.anomaly_detector = IsolationForest(random_state=42, contamination=0.02)

        else:
            raise ValueError("Invalid model_type. Use: rf / linear / logistic / kmeans / dbscan / isolation")

    # -----------------------------------
    # Feature engineering
    # -----------------------------------
    def prepare_features(self, df: pd.DataFrame, predict_mode=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        df_work = df.copy()

        # required
        if "date" in df_work.columns:
            df_work["date"] = pd.to_datetime(df_work["date"], errors="coerce")
            df_work["month"] = df_work["date"].dt.month
            df_work["year"] = df_work["date"].dt.year

        if "disease_trend" not in df_work.columns and "cases" in df_work.columns and "location" in df_work.columns:
            df_work["disease_trend"] = df_work.groupby("location")["cases"].pct_change().fillna(0)

        # encode location
        if "location" in df_work.columns:
            if predict_mode and self.is_trained and "location" in self.label_encoders:
                le = self.label_encoders["location"]
                known = {v: i for i, v in enumerate(le.classes_)}
                df_work["location_encoded"] = df_work["location"].astype(str).map(known).fillna(0).astype(int)
            else:
                le = LabelEncoder()
                df_work["location_encoded"] = le.fit_transform(df_work["location"].astype(str))
                self.label_encoders["location"] = le

        features = ["cases", "disease_trend", "month", "year", "location_encoded"]
        self.feature_columns = features

        # safe fill
        for c in features:
            if c not in df_work.columns:
                df_work[c] = 0

        X = df_work[features].fillna(0).values

        # regression
        if self.model_type in ["rf", "linear"]:
            if not predict_mode:
                if "sales_volume" not in df_work.columns:
                    raise ValueError("Target column missing: sales_volume")
                y = df_work["sales_volume"].values
                return X, y
            return X, None

        # classification
        if self.model_type == "logistic":
            if not predict_mode:
                if "sales_volume" not in df_work.columns:
                    raise ValueError("Target column missing: sales_volume")

                # Low/Med/High bins
                y_class = pd.qcut(df_work["sales_volume"], q=3, labels=[0, 1, 2]).astype(int).values
                return X, y_class
            return X, None

        # unsupervised
        return X, None

    # -----------------------------------
    # Regression
    # -----------------------------------
    def train_regression(self, df: pd.DataFrame) -> Dict[str, Any]:
        X, y = self.prepare_features(df, predict_mode=False)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.predictor.fit(X_train, y_train)
        self.is_trained = True

        pred_train = self.predictor.predict(X_train)
        pred_test = self.predictor.predict(X_test)

        return {
            "model": self.model_type,
            "train_r2": round(r2_score(y_train, pred_train), 3),
            "test_r2": round(r2_score(y_test, pred_test), 3),
            "features": self.feature_columns
        }

    def predict_regression(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        X, _ = self.prepare_features(df, predict_mode=True)
        return self.predictor.predict(X)

    # -----------------------------------
    # Classification
    # -----------------------------------
    def train_classification(self, df: pd.DataFrame) -> Dict[str, Any]:
        X, y = self.prepare_features(df, predict_mode=False)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_s = self.cls_scaler.fit_transform(X_train)
        X_test_s = self.cls_scaler.transform(X_test)

        self.classifier.fit(X_train_s, y_train)
        self.is_trained = True

        preds = self.classifier.predict(X_test_s)

        return {
            "model": "logistic",
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "features": self.feature_columns
        }

    def predict_classification(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("Model not trained yet!")
        X, _ = self.prepare_features(df, predict_mode=True)
        Xs = self.cls_scaler.transform(X)
        return self.classifier.predict(Xs)

    # -----------------------------------
    # Clustering
    # -----------------------------------
    def fit_clustering(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.prepare_features(df, predict_mode=True)

        if self.model_type == "kmeans":
            self.clusterer.fit(X)
            self.is_trained = True
            return self.clusterer.labels_

        if self.model_type == "dbscan":
            labels = self.clusterer.fit_predict(X)
            self.is_trained = True
            return labels

        raise ValueError("Not a clustering model.")

    # -----------------------------------
    # Anomaly Detection
    # -----------------------------------
    def fit_anomaly(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self.prepare_features(df, predict_mode=True)
        self.anomaly_detector.fit(X)
        self.is_trained = True
        return self.anomaly_detector.predict(X)

    # -----------------------------------
    # Save / Load
    # -----------------------------------
    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)
