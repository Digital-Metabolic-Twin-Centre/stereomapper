from dataclasses import dataclass

@dataclass
class StereoCounts:
    num_stereogenic_elements: int
    num_tetra_matches: int
    num_tetra_flips: int
    num_db_matches: int
    num_db_flips: int
    num_missing: int
    num_unspecified: int

    @classmethod
    def from_stereo_elements(cls, d: dict) -> "StereoCounts":
        return cls(
            num_tetra_matches=d.get("tetra_matches", 0),
            num_tetra_flips=d.get("tetra_flips", 0),
            num_db_matches=d.get("db_matches", 0),
            num_db_flips=d.get("db_flips", 0),
            num_missing=d.get("missing_centres", 0),
            num_unspecified=d.get("unspecified", 0),
            num_stereogenic_elements=d.get("total_stereo", 0),
        )
    
    def as_classification_kwargs(self) -> dict:
        return {
            "num_tetra_matches": self.num_tetra_matches,
            "num_tetra_flips": self.num_tetra_flips,
            "num_db_matches": self.num_db_matches,
            "num_db_flips": self.num_db_flips,
            "num_missing": self.num_missing,
            "num_unspecified": self.num_unspecified,
            "num_stereogenic_elements": self.num_stereogenic_elements,
        }