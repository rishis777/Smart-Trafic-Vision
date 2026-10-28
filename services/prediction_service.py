"""PredictionService: Machine learning model for congestion forecasting
and bottleneck hotspot identification using scikit-learn.
"""

from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from traffic_ai.config import DATA_DIR, DIRECTIONS
from traffic_ai.db.models import TrafficData


class PredictionService:
    """Predicts future lane density and discovers temporal congestion hotspots."""

    def __init__(self, model_file: Optional[Path] = None):
        self.model_file = model_file or (DATA_DIR / "congestion_model.joblib")
        self.model: Optional[RandomForestRegressor] = None
        self.hotspots: List[Dict[str, Any]] = []
        self.direction_map = {d: i for i, d in enumerate(DIRECTIONS)}
        self._load_or_train_initial()

    def _load_or_train_initial(self) -> None:
        """Load cached model or train baseline predictor."""
        if self.model_file.exists():
            try:
                data = joblib.load(self.model_file)
                self.model = data.get("model")
                self.hotspots = data.get("hotspots", [])
                return
            except Exception:
                pass

        # Train a default statistical model with realistic synthetic baseline
        self.train_baseline()

    def train_baseline(self) -> Dict[str, Any]:
        """Train baseline regression model on synthetic historical distribution."""
        X = []
        y = []

        # Generate realistic traffic curve across 7 days, 24 hours, 4 directions
        for day in range(7):
            is_weekend = int(day in [5, 6])
            for hour in range(24):
                # Rush hours: 8-10am and 5-8pm
                morning_rush = np.exp(-0.5 * ((hour - 8.5) / 1.5) ** 2)
                evening_rush = np.exp(-0.5 * ((hour - 18.0) / 1.8) ** 2)
                base_curve = 0.15 + 0.55 * (morning_rush + evening_rush)
                if is_weekend:
                    base_curve = 0.15 + 0.35 * np.exp(-0.5 * ((hour - 14.0) / 3.0) ** 2)

                for direction, d_idx in self.direction_map.items():
                    # North/South corridors have higher flow
                    corridor_bias = 0.12 if direction in ["NORTH", "SOUTH"] else 0.0
                    density = float(np.clip(base_curve + corridor_bias + np.random.normal(0, 0.03), 0.05, 0.95))

                    X.append([hour, day, is_weekend, d_idx])
                    y.append(density)

        X = np.array(X)
        y = np.array(y)

        model = RandomForestRegressor(n_estimators=40, max_depth=8, random_state=42)
        model.fit(X, y)
        self.model = model

        # Derive hotspots
        self.hotspots = self._compute_hotspots()
        self._save_model()

        preds = model.predict(X)
        return {
            "status": "TRAINED",
            "samples": len(X),
            "r2_score": round(float(r2_score(y, preds)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 3),
            "hotspots_count": len(self.hotspots)
        }

    def train_on_database_records(self) -> Dict[str, Any]:
        """Train or fine-tune model using real recorded observations in SQLite."""
        records = TrafficData.query.all()
        if len(records) < 50:
            return self.train_baseline()

        X = []
        y = []
        for r in records:
            if not r.timestamp or r.direction not in self.direction_map:
                continue
            hour = r.timestamp.hour
            weekday = r.timestamp.weekday()
            is_weekend = int(weekday in [5, 6])
            d_idx = self.direction_map[r.direction]

            X.append([hour, weekday, is_weekend, d_idx])
            y.append(r.density)

        X = np.array(X)
        y = np.array(y)

        model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X, y)
        self.model = model

        self.hotspots = self._compute_hotspots()
        self._save_model()

        preds = model.predict(X)
        return {
            "status": "TRAINED_FROM_DB",
            "samples": len(X),
            "r2_score": round(float(r2_score(y, preds)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 3),
            "hotspots_count": len(self.hotspots)
        }

    def predict_congestion(self, direction: str, hour: int, day_of_week: int) -> Dict[str, Any]:
        """Predict expected traffic density and severity for given hour and day."""
        if self.model is None:
            self.train_baseline()

        d_idx = self.direction_map.get(direction.upper(), 0)
        is_weekend = int(day_of_week in [5, 6])

        features = np.array([[hour, day_of_week, is_weekend, d_idx]])
        pred_density = float(np.clip(self.model.predict(features)[0], 0.05, 1.0))
        pred_density = round(pred_density, 3)

        if pred_density < 0.25:
            level = "LOW"
            rec = "Normal cycle timings sufficient."
        elif pred_density < 0.55:
            level = "MODERATE"
            rec = "Standard green wave recommended."
        elif pred_density < 0.80:
            level = "HIGH"
            rec = "Preemptively extend green allocation +15s."
        else:
            level = "SEVERE"
            rec = "Activate congestion rerouting and deploy traffic warden."

        return {
            "direction": direction,
            "hour": hour,
            "day_of_week": day_of_week,
            "predicted_density": pred_density,
            "density_level": level,
            "operational_recommendation": rec
        }

    def _compute_hotspots(self) -> List[Dict[str, Any]]:
        """Identify critical bottlenecks across the weekly cycle."""
        hotspots = []
        for day in range(7):
            day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][day]
            for hour in range(24):
                for direction in DIRECTIONS:
                    res = self.predict_congestion(direction, hour, day)
                    if res["predicted_density"] >= 0.65:
                        hotspots.append({
                            "day": day_name,
                            "hour_window": f"{hour:02d}:00 - {(hour+1)%24:02d}:00",
                            "direction": direction,
                            "expected_density": res["predicted_density"],
                            "level": res["density_level"]
                        })

        hotspots.sort(key=lambda x: x["expected_density"], reverse=True)
        return hotspots[:10]  # Top 10 bottlenecks

    def get_hotspots(self) -> List[Dict[str, Any]]:
        return self.hotspots

    def _save_model(self) -> None:
        try:
            self.model_file.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump({"model": self.model, "hotspots": self.hotspots}, self.model_file)
        except Exception:
            pass
