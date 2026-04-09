from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .runtime_context import LocalizedFeatureProvider


class ContextProvider(Protocol):
    name: str

    def resolve(
        self,
        region: str | None,
        state: str | None,
        target_time: str | None = None,
    ) -> dict[str, Any]:
        ...

    def get_region_catalog(self) -> list[dict[str, Any]]:
        ...

    def get_temporal_catalog(self) -> dict[str, Any]:
        ...

    def get_provider_status(self) -> dict[str, Any]:
        ...


@dataclass
class HistoricalTrainingContextProvider:
    dataset_path: Path | None
    numeric_features: list[str]
    categorical_features: list[str]
    label_columns: list[str]

    def __post_init__(self) -> None:
        self.name = "historical_training_context"
        self._provider = LocalizedFeatureProvider(
            dataset_path=self.dataset_path,
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            label_columns=self.label_columns,
        )

    def resolve(
        self,
        region: str | None,
        state: str | None,
        target_time: str | None = None,
    ) -> dict[str, Any]:
        payload = self._provider.resolve(region=region, state=state, target_time=target_time)
        payload["selected_provider"] = self.name
        return payload

    def get_region_catalog(self) -> list[dict[str, Any]]:
        return self._provider.get_region_catalog()

    def get_temporal_catalog(self) -> dict[str, Any]:
        catalog = dict(self._provider.get_temporal_catalog())
        catalog["selected_provider"] = self.name
        return catalog

    def get_provider_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "historical",
            "available": bool(self._provider.available),
            "buffer_strategy": self._provider.buffer_strategy,
            "live_weather_status": self._provider.live_weather_status,
        }


@dataclass
class PendingLiveWeatherProvider:
    name: str = "live_weather_pending"

    def resolve(
        self,
        region: str | None,
        state: str | None,
        target_time: str | None = None,
    ) -> dict[str, Any]:
        return {
            "available": False,
            "resolved_region": region or "",
            "resolved_state": state or "",
            "region_key": None,
            "target_time": target_time,
            "target_month": None,
            "target_season": None,
            "match_level": "unavailable",
            "context_confidence": 0.0,
            "buffer_strategy": "live_observed_plus_forecast",
            "data_source": "provider_not_enabled",
            "feature_defaults": {},
            "validation_bands": {},
            "feature_provenance": {},
            "crop_prior": [],
            "observation_counts": {},
            "selected_provider": self.name,
            "provider_message": "Live weather provider is not enabled in this environment.",
        }

    def get_region_catalog(self) -> list[dict[str, Any]]:
        return []

    def get_temporal_catalog(self) -> dict[str, Any]:
        return {
            "autofill_enabled": False,
            "buffer_strategy": "live_observed_plus_forecast",
            "live_weather_status": "planned_pending_external_approval",
            "selected_provider": self.name,
        }

    def get_provider_status(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": "live_weather",
            "available": False,
            "buffer_strategy": "live_observed_plus_forecast",
            "live_weather_status": "planned_pending_external_approval",
        }


@dataclass
class CompositeContextProvider:
    providers: list[ContextProvider]

    def resolve(
        self,
        region: str | None,
        state: str | None,
        target_time: str | None = None,
    ) -> dict[str, Any]:
        statuses = [provider.get_provider_status() for provider in self.providers]
        fallback_payload: dict[str, Any] | None = None
        for provider in self.providers:
            payload = provider.resolve(region=region, state=state, target_time=target_time)
            payload["provider_stack"] = statuses
            payload["selected_provider"] = payload.get("selected_provider", getattr(provider, "name", "unknown"))
            if payload.get("available"):
                return payload
            fallback_payload = payload
        if fallback_payload is not None:
            return fallback_payload
        return {
            "available": False,
            "resolved_region": region or "",
            "resolved_state": state or "",
            "provider_stack": statuses,
            "selected_provider": "none",
        }

    def get_region_catalog(self) -> list[dict[str, Any]]:
        for provider in self.providers:
            catalog = provider.get_region_catalog()
            if catalog:
                return catalog
        return []

    def get_temporal_catalog(self) -> dict[str, Any]:
        base: dict[str, Any] = {}
        statuses = [provider.get_provider_status() for provider in self.providers]
        for provider in self.providers:
            catalog = provider.get_temporal_catalog()
            if catalog:
                base.update(catalog)
        base["provider_stack"] = statuses
        return base

    def get_provider_status(self) -> dict[str, Any]:
        return {
            "name": "composite_context_provider",
            "kind": "composite",
            "providers": [provider.get_provider_status() for provider in self.providers],
        }


def build_default_context_provider(
    dataset_path: Path | None,
    numeric_features: list[str],
    categorical_features: list[str],
    label_columns: list[str],
) -> CompositeContextProvider:
    return CompositeContextProvider(
        providers=[
            PendingLiveWeatherProvider(),
            HistoricalTrainingContextProvider(
                dataset_path=dataset_path,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                label_columns=label_columns,
            ),
        ]
    )
