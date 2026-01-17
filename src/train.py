# medicine_sales_prediction/src/train.py
import sys
import os
from pathlib import Path

# 🔥 FIX: Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from src.data import DiseaseSalesData
from src.model import MedicineSalesModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    config_path = 'configs/data.yaml'
    model_save_path = 'models/medicine_sales_model.pkl'
    
    try:
        data_handler = DiseaseSalesData(config_path)
        df = data_handler.load_and_merge()
        logger.info(f"✅ Loaded {len(df)} records")
        logger.info(f"📊 Columns: {df.columns.tolist()}")
        logger.info(f"📈 Shape: {df.shape}")
        
        model = MedicineSalesModel()
        scores = model.train(df)
        logger.info(f"🎯 Scores: {scores}")
        
        Path(model_save_path).parent.mkdir(exist_ok=True)
        model.save(model_save_path)
        logger.info(f"💾 Model saved: {model_save_path}")
        
    except FileNotFoundError as e:
        logger.error(f"❌ File missing: {e}")
        logger.info("💡 Create sample data/ folder with CSVs or update configs/data.yaml")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.info("💡 Check your CSV columns match expected format")

if __name__ == "__main__":
    main()
