import os
import pickle
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from loguru import logger

WEIGHTS_PATH = os.environ['WEIGHTS_PATH']
HYSTERESIS_WEIGHTS_PATH = os.path.join(WEIGHTS_PATH, "сatboost_coer_sat_mr")


class AbstractHysteresisModel(ABC):

    def __init__(
            self,
            features_weights_path: str,
            scaler_weights_path: str,
            model_weights_path: str
            ) -> None:
        super().__init__()

        features = self.__load_weights(features_weights_path)
        if features is None:
            raise Exception(f"Input features are not set for {self.model_name}")
        
        features_to_drop = ['orig_c1', 'orig_c2']
        for feat in features_to_drop:
            if feat in features.columns:
                features.drop(feat, axis=1, inplace=True)
        self.feature_names = features.columns

        self.scaler = self.__load_weights(scaler_weights_path)
        if self.scaler is None:
            raise Exception(f"Scaler weights are not set for {self.model_name}")

        self.model = self.__load_weights(model_weights_path)
        if self.model is None:
            raise Exception(f"Model weights are not set for {self.model_name}")
        
        logger.info(f"Loaded {self.model_name} weights.")

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    def __load_weights(self, path: str):
        try:
            obj = joblib.load(path)
            return obj
        except pickle.UnpicklingError as e:
            logger.error(f"Cannot load the object on: {path}: \n {e}")
            return

    def _get_featured_data(self, data: pd.DataFrame) -> pd.DataFrame | None:
        try:
            data = data[self.feature_names]
        except pd.errors.IndexingError as e:
            logger.error(f"Cannot prepare data for {self.model_name}: \n {e}")
            return
        return data

    @abstractmethod
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def scale_output(self, value: float) -> float:
        pass

    def predict(self, data):
        scaled_data = self.scaler.transform(
                self.prepare_data(data)
            )
        predicted_value = self.model.predict(scaled_data).item()
        return self.scale_output(predicted_value)


class CoercivityModel(AbstractHysteresisModel):
    def __init__(
            self,
            features_weights_path: str,
            scaler_weights_path: str,
            model_weights_path: str
            ) -> None:
        super().__init__(features_weights_path, scaler_weights_path, model_weights_path)
    
    @property
    def model_name(self) -> str:
        return "coercivity model"
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        featured_data = self._get_featured_data(data)

        if featured_data is None:
            raise Exception("Cannot get X for model %s", self.model_name)

        columns_to_drop = ['coer_oe']
        for col in columns_to_drop:
            if col in featured_data.columns:
                featured_data.drop(col, axis=1, inplace=True)
        return featured_data
    
    def scale_output(self, value: float) -> float:
        return np.power(10, value).item()


class RemanenceModel(AbstractHysteresisModel):
    def __init__(
            self,
            features_weights_path: str,
            scaler_weights_path: str,
            model_weights_path: str
            ) -> None:
        super().__init__(features_weights_path, scaler_weights_path, model_weights_path)

    @property
    def model_name(self) -> str:
        return "remanence model"

    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        featured_data = self._get_featured_data(data)

        if featured_data is None:
            raise Exception("Cannot get X for model %s", self.model_name)

        columns_to_drop = ['mr (emu/g)']
        for col in columns_to_drop:
            if col in featured_data.columns:
                featured_data.drop(col, axis=1, inplace=True)
        return featured_data

    def scale_output(self, value: float) -> float:
        return np.power(10, value).item()


class SaturationModel(AbstractHysteresisModel):
    def __init__(
            self,
            features_weights_path: str,
            scaler_weights_path: str,
            model_weights_path: str
            ) -> None:
        super().__init__(features_weights_path, scaler_weights_path, model_weights_path)
    
    @property
    def model_name(self) -> str:
        return "saturation model"

    def prepare_data(self, data: pd.DataFrame):
        featured_data = self._get_featured_data(data)

        if featured_data is None:
            raise Exception("Cannot get X for model %s", self.model_name)

        columns_to_drop = ['sat_em_g']
        for col in columns_to_drop:
            if col in featured_data.columns:
                featured_data.drop(col, axis=1, inplace=True)
        return featured_data

    def scale_output(self, value: float) -> float:
        return np.power(10, value).item()


class HysteresisModelsRouter:
    def __init__(self) -> None:
        self.coercivity = CoercivityModel(
            features_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "input_data.pkl"),
            scaler_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "scaler_coer.pkl"),
            model_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "catboost_model_coer.pkl")
        )

        self.remanence = RemanenceModel(
            features_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "input_data.pkl"),
            scaler_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "scaler_mr.pkl"),
            model_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "catboost_model_mr.pkl")
        )

        self.saturation = SaturationModel(
            features_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "input_data.pkl"),
            scaler_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "scaler_sat.pkl"),
            model_weights_path=os.path.join(HYSTERESIS_WEIGHTS_PATH, "catboost_model_sat.pkl")
        )


__all__ = [
    "HysteresisModelsRouter",
    "CoercivityModel",
    "RemanenceModel",
    "SaturationModel",
]
