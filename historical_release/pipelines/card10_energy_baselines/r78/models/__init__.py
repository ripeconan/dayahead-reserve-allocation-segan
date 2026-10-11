"""Model adapters for R78 energy baselines."""

from .base import ModelAdapter, PointForecast
from .catboost_energy import CatBoostEnergyAdapter
from .lightgbm_energy import LightGBMEnergyAdapter
from .nyiso_p7 import NyisoP7Adapter
from .neuralforecast_energy import NHITSQuantileAdapter, PatchTSTQuantileAdapter
from .proloaf_native import ProLoaFNativeAdapter
from .ridge_lag2 import RidgeLag2Adapter
from .seasonal_naive import SeasonalNaiveAdapter
from .tft_quantile import TFTQuantileAdapter

ADAPTERS = {
    "seasonal_naive_d7": SeasonalNaiveAdapter,
    "nyiso_p7_official": NyisoP7Adapter,
    "ridge_lag2": RidgeLag2Adapter,
    "lightgbm_energy": LightGBMEnergyAdapter,
    "catboost_energy_quantile": CatBoostEnergyAdapter,
    "tft_quantile": TFTQuantileAdapter,
    "proloaf_native": ProLoaFNativeAdapter,
    "nhits_quantile": NHITSQuantileAdapter,
    "patchtst_quantile": PatchTSTQuantileAdapter,
}

__all__ = [
    "ADAPTERS",
    "CatBoostEnergyAdapter",
    "LightGBMEnergyAdapter",
    "ModelAdapter",
    "NyisoP7Adapter",
    "NHITSQuantileAdapter",
    "PatchTSTQuantileAdapter",
    "PointForecast",
    "ProLoaFNativeAdapter",
    "RidgeLag2Adapter",
    "SeasonalNaiveAdapter",
    "TFTQuantileAdapter",
]
