import scoreq


class ScoreQ:
    def __init__(self):
        self.models = {
            "natural_nr": scoreq.Scoreq(data_domain="natural", mode="nr"),
            "natural_ref": scoreq.Scoreq(data_domain="natural", mode="ref"),
            "synthetic_nr": scoreq.Scoreq(data_domain="synthetic", mode="nr"),
            "synthetic_ref": scoreq.Scoreq(data_domain="synthetic", mode="ref"),
        }

    @staticmethod
    def _to_float(value):
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def __call__(self, audio_path: str, name: str = "synthetic_nr", reference_path: str | None = None):
        assert name in ["natural_nr", "natural_ref", "synthetic_nr", "synthetic_ref"]
        mode = "ref" if name.endswith("_ref") else "nr"
        if mode == "ref" and reference_path is None:
            raise ValueError(f"reference_path is required for {name}")

        pred = self.models[name].predict(
            test_path=audio_path,
            ref_path=reference_path if mode == "ref" else None,
        )
        return self._to_float(pred)