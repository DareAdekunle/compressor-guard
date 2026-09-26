"""
Shared maintenance-economics model (README §4).

An alert is worth something only if it arrives early enough to act on, which includes
getting the spare part:

    lead >= L_parts + L_schedule   -> value = C_unplanned - C_planned
    0 < lead < L_parts + L_schedule -> value = C_unplanned - C_planned - C_expedite
    missed / not before onset       -> value = 0 (the failure is paid in full)

    net value = sum(value of caught failures) - false_alerts * C_inspection

All figures in config/costs.yaml are illustrative assumptions.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import numpy as np

from compressor_guard.config import load_costs


@dataclass(frozen=True)
class CostModel:
    unplanned_failure: float = 25_000.0
    planned_maintenance: float = 4_000.0
    expedite_penalty: float = 5_000.0
    false_inspection: float = 500.0
    parts_delivery_hours: float = 48.0
    scheduling_hours: float = 12.0

    @classmethod
    def from_yaml(cls, path: Union[str, Path] = "config/costs.yaml") -> "CostModel":
        c = load_costs(path)
        return cls(**{**c.get("costs", {}), **c.get("lead_times", {})})

    def with_parts_lead(self, hours: float) -> "CostModel":
        return replace(self, parts_delivery_hours=float(hours))

    @property
    def required_lead_hours(self) -> float:
        return self.parts_delivery_hours + self.scheduling_hours

    @property
    def full_value(self) -> float:
        return self.unplanned_failure - self.planned_maintenance

    @property
    def expedited_value(self) -> float:
        return self.unplanned_failure - self.planned_maintenance - self.expedite_penalty

    def value_of_warning(self, lead_hours: Optional[float]) -> float:
        """Value of catching one failure with the given lead time (None/<=0 -> missed)."""
        if lead_hours is None or not np.isfinite(lead_hours) or lead_hours <= 0:
            return 0.0
        return self.full_value if lead_hours >= self.required_lead_hours else self.expedited_value

    def net_value(self, lead_times: Iterable[Optional[float]], n_false_alerts: float) -> float:
        caught = sum(self.value_of_warning(lt) for lt in lead_times)
        return caught - n_false_alerts * self.false_inspection

    def break_even_false_alerts(self, lead_times: Iterable[Optional[float]]) -> float:
        """Number of false alerts at which the programme's net value falls to zero."""
        return sum(self.value_of_warning(lt) for lt in lead_times) / self.false_inspection


def parts_lead_scenarios(path: Union[str, Path] = "config/costs.yaml") -> Sequence[float]:
    return load_costs(path).get("sensitivity", {}).get("parts_scenarios_hours", [24.0, 48.0, 168.0, 336.0])
