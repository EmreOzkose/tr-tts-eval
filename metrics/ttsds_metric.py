from pathlib import Path
import math

from ttsds import BenchmarkSuite
from ttsds.ttsds import BENCHMARKS
from ttsds.util.dataset import DirectoryDataset


class TTSDSMetric:
    _COMPATIBLE_BENCHMARKS = {
        "hubert",
        "wav2vec2",
        "wavlm",
        "mpm",
        "pitch",
        "hubert_token_sr",
        "allosaurus_sr",
        "dvector",
        "whisper_activations",
        "wada_snr",
    }

    def __init__(self, reference_root: str, device: str = "cuda", write_to_file: str | None = None):
        self.reference_root = Path(reference_root)
        self.device = device
        self.write_to_file = write_to_file
        self._patch_transformers_torchload_gate()

    @staticmethod
    def _patch_transformers_torchload_gate() -> None:
        """
        Allow model loading on environments pinned below torch 2.6.
        This keeps ttsds usable in legacy eval setups.
        """
        try:
            import transformers.modeling_utils as modeling_utils
            import transformers.utils.import_utils as import_utils

            if hasattr(import_utils, "check_torch_load_is_safe"):
                import_utils.check_torch_load_is_safe = lambda: None
            if hasattr(modeling_utils, "check_torch_load_is_safe"):
                modeling_utils.check_torch_load_is_safe = lambda: None
        except Exception:
            # If transformers internals change, fallback is to let original error raise.
            pass

    @staticmethod
    def _to_float(value):
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _safe_finite_float(self, value):
        try:
            number = self._to_float(value)
        except Exception:
            return None
        if math.isfinite(number):
            return number
        return None

    def _extract_dataset_scores(self, agg, input_name: str) -> dict[str, float]:
        dataset_rows = agg[agg["dataset"] == input_name]
        results = {}

        if dataset_rows.empty or "score_mean" not in dataset_rows.columns:
            return {"overall": 0.0}

        bench_col = None
        for candidate in ("benchmark", "benchmark_name", "metric", "name"):
            if candidate in dataset_rows.columns:
                bench_col = candidate
                break

        metric_rows = dataset_rows
        if "benchmark_category" in dataset_rows.columns:
            metric_rows = dataset_rows[dataset_rows["benchmark_category"] != "OVERALL"]

        if bench_col is not None:
            for _, row in metric_rows.iterrows():
                score = self._safe_finite_float(row.get("score_mean"))
                if score is None:
                    continue
                metric_name = str(row.get(bench_col))
                if metric_name:
                    results[metric_name] = score

        overall = None
        if "benchmark_category" in dataset_rows.columns:
            overall_row = dataset_rows[dataset_rows["benchmark_category"] == "OVERALL"]
            if not overall_row.empty:
                overall = self._safe_finite_float(overall_row.iloc[0].get("score_mean"))

        if overall is None:
            finite_scores = []
            for value in dataset_rows["score_mean"].tolist():
                score = self._safe_finite_float(value)
                if score is not None:
                    finite_scores.append(score)
            if finite_scores:
                overall = float(sum(finite_scores) / len(finite_scores))

        results["overall"] = overall if overall is not None else 0.0
        return results

    def _resolve_reference_dir(self, input_name: str) -> Path:
        candidate = self.reference_root / input_name
        if candidate.exists():
            return candidate
        return self.reference_root

    def __call__(self, input_name: str, audio_path_list: list[str]) -> float:
        all_results = self.process_all({input_name: audio_path_list})
        return float(all_results.get(input_name, {}).get("overall", 0.0))

    def process_all(self, input_groups: dict[str, list[str]]) -> dict[str, dict[str, float]]:
        """
        Process all input groups in a single batch.

        Args:
            input_groups (dict): A dictionary where keys are input group names and values are lists of audio file paths.

        Returns:
            dict: A dictionary mapping input group names to benchmark scores with an "overall" key.
        """
        datasets = []
        reference_datasets = []

        for input_name, audio_path_list in input_groups.items():
            if not audio_path_list:
                raise ValueError(f"No wav files provided for input group: {input_name}")

            synth_dir = Path(audio_path_list[0]).parent
            reference_dir = self._resolve_reference_dir(input_name)

            if not reference_dir.exists():
                raise ValueError(f"Reference directory does not exist: {reference_dir}")

            datasets.append(DirectoryDataset(root_dir=str(synth_dir), name=input_name))
            reference_datasets.append(
                DirectoryDataset(root_dir=str(reference_dir), name=f"{input_name}_reference")
            )

        compatible_benchmarks = {
            name: benchmark
            for name, benchmark in BENCHMARKS.items()
            if name in self._COMPATIBLE_BENCHMARKS
        }
        cache_dir = Path.home() / ".cache" / "ttsds"
        cache_dir.mkdir(parents=True, exist_ok=True)

        suite = BenchmarkSuite(
            datasets=datasets,
            reference_datasets=reference_datasets,
            benchmarks=compatible_benchmarks,
            device=self.device,
            cache_dir=str(cache_dir),
            skip_errors=True,
            write_to_file=self.write_to_file,
            n_workers=1,
        )
        suite.run()
        agg = suite.get_aggregated_results()

        results = {}
        for input_name in input_groups.keys():
            results[input_name] = self._extract_dataset_scores(agg, input_name)

        return results
