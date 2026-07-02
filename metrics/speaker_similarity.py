import torch
import wespeaker


class SpeakerSimilarity:
    def __init__(self, model_name: str = "english", device: str | None = None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = wespeaker.load_model(model_name)
        self.model.set_device(self.device)
        self._patch_compute_features_device()
        self.embedding_cache = {}

    def _patch_compute_features_device(self):
        original_compute_features = self.model.compute_features

        def compute_features_on_model_device(wavform, sample_rate=16000, cmn=True):
            feat = original_compute_features(wavform, sample_rate=sample_rate, cmn=cmn)
            if isinstance(feat, torch.Tensor):
                return feat.to(self.model.device)
            return feat

        self.model.compute_features = compute_features_on_model_device

    def _get_embedding(self, path: str):
        key = str(path)
        if key not in self.embedding_cache:
            self.embedding_cache[key] = self.model.extract_embedding(key)

        return self.embedding_cache[key]

    def _compute_similarity(self, path1: str, path2: str):
        emb1 = self._get_embedding(path1)
        emb2 = self._get_embedding(path2)
        if emb1 is None or emb2 is None:
            return 0.0
        return float(self.model.cosine_similarity(emb1, emb2))

    def __call__(self, path1: str, path2: str):
        try:
            return self._compute_similarity(path1, path2)
        except RuntimeError as exc:
            msg = str(exc)
            mismatch = "Input type" in msg and "weight type" in msg and "should be the same" in msg
            if mismatch and self.device != "cpu":
                self.device = "cpu"
                self.model.set_device("cpu")
                self._patch_compute_features_device()
                self.embedding_cache.clear()
                return self._compute_similarity(path1, path2)
            raise