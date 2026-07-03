

class AudioboxPQ:
    def __init__(self):
        from audiobox_aesthetics.infer import initialize_predictor
        self.predictor = initialize_predictor()

    def __call__(self, audio_path: str, name="PQ"):
        assert name in ["CE", "CU", "PC", "PQ"]
        score = self.predictor.forward([{"path":audio_path}])
        return score[0][name]