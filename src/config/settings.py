import os
import dotenv


class Settings:
    def __init__(self):
        dotenv.load_dotenv()
        self.universal_cefr_spanish_datasets = os.getenv('UNIVERSAL_CEFR_SPANISH_DATASETS')
        self.universal_cefr_english_datasets = os.getenv('UNIVERSAL_CEFR_ENGLISH_DATASETS')
        self.raw_base_path = os.getenv('DATASETS_RAW_BASE_PATH')
        self.split_base_path = os.getenv('DATASETS_SPLIT_BASE_PATH')
        self.models_base_path = os.getenv('MODELS_BASE_PATH')
settings = Settings()