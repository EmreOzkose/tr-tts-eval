import torch
import torchaudio


class Squim:
    def __init__(self):
        self.objective = torchaudio.pipelines.SQUIM_OBJECTIVE.get_model()
        self.subjective = torchaudio.pipelines.SQUIM_SUBJECTIVE.get_model()
        self.resampler = {}

    def __call__(self, audio_path: str, name: str = "pesq", reference_for_mos: str | None = None):
        assert name in ["pesq", "stoi", "si_sdr", "mos"]
        if name == "mos":
            assert reference_for_mos is not None, "Reference audio path needed for MOS score"
            reference_waveform, reference_sample_rate = torchaudio.load(reference_for_mos)
            if reference_sample_rate != 16000:
                if reference_sample_rate not in self.resampler:
                    self.resampler[reference_sample_rate] = torchaudio.transforms.Resample(reference_sample_rate, 16000)
                reference_waveform = self.resampler[reference_sample_rate](reference_waveform)
            self.subjective.set_reference(reference_waveform)
        
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            if sample_rate not in self.resampler:
                self.resampler[sample_rate] = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = self.resampler[sample_rate](waveform)

        with torch.inference_mode():
            if name == "mos":
                _, _, score = self.subjective(waveform)
                return float(score.item())

            pesq, stoi, si_sdr = self.objective(waveform)
            scores = {
                "pesq": pesq,
                "stoi": stoi,
                "si_sdr": si_sdr,
            }
            return float(scores[name].item())
