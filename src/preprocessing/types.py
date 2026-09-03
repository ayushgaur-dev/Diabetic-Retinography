"""Types for adaptive enhancement (SIH26038 Phase 3). Plain stdlib
dataclasses, matching the src/quality convention (no new dependencies)."""

from dataclasses import dataclass, field


ENHANCEMENT_VERSION = "1.0.0"

IMPROVED = "IMPROVED"
UNCHANGED = "UNCHANGED"
WORSE = "WORSE"


@dataclass
class EnhancementResult:
    """Structured outcome of one adaptive enhancement attempt."""

    enhanced_image: object  # RGB uint8 array, or None when rejected/bypassed
    operations_applied: list = field(default_factory=list)
    before_quality: dict = field(default_factory=dict)  # QualityResult.to_dict()
    after_quality: dict = field(default_factory=dict)  # QualityResult.to_dict() or {}
    enhancement_successful: bool = False
    comparison: str = UNCHANGED  # IMPROVED / UNCHANGED / WORSE
    changed_quality_dimensions: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    bypassed: bool = False  # True when input was GOOD (no enhancement attempted)
    rejected: bool = False  # True when input was UNGRADABLE (no enhancement attempted)
    enhancement_version: str = ENHANCEMENT_VERSION

    def to_dict(self):
        return {
            "operations_applied": list(self.operations_applied),
            "before_quality": self.before_quality,
            "after_quality": self.after_quality,
            "enhancement_successful": self.enhancement_successful,
            "comparison": self.comparison,
            "changed_quality_dimensions": list(self.changed_quality_dimensions),
            "warnings": list(self.warnings),
            "bypassed": self.bypassed,
            "rejected": self.rejected,
            "enhancement_version": self.enhancement_version,
            "enhanced_shape": (list(self.enhanced_image.shape)
                               if self.enhanced_image is not None else None),
        }
