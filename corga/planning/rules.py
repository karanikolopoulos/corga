from dataclasses import field, dataclass

from corga.planning.geo import WalkingModel


@dataclass(frozen=True)
class PlanningRules:
    walking: WalkingModel = field(default_factory=WalkingModel)
    max_leg_minutes: int = 40
    max_wait_minutes: int = 20
    max_stops: int = 5
    walk_penalty: float = 0.2  # ranking cost per minute walked: 5 minutes weigh as much as one place

    def assumptions(self) -> list[str]:
        return [
            str(self.walking),
            "All stops are visited on foot.",
            "Opening hours and visit durations are sample data, not live visitor information.",
        ]
