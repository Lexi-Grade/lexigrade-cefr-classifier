import os
import dotenv


class Settings:
    def __init__(self):
        dotenv.load_dotenv()
        self.universal_cefr_spanish_datasets = os.getenv('UNIVERSAL_CEFR_SPANISH_DATASETS')
        self.universal_cefr_english_datasets = os.getenv('UNIVERSAL_CEFR_ENGLISH_DATASETS')
        self.raw_path = os.getenv('DATASETS_RAW_PATH')
        self.split_path = os.getenv('DATASETS_SPLIT_PATH')
        self.models_path = os.getenv('MODELS_PATH')
settings = Settings()