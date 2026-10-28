"""CPU ONNX speaker embedder (WeSpeaker ResNet34-LM). Vectors stay in memory only."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from backend.errors import ModelMissingError

SR = 16000
MIN_SAMPLES = SR // 2  # 0.5 s


def fbank(audio: np.ndarray) -> np.ndarray:
    """80-bin Kaldi fbank (25 ms / 10 ms, hamming) with per-utterance mean normalization -> (T, 80)."""
    import kaldi_native_fbank as knf

    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.samp_freq = SR
    opts.frame_opts.window_type = "hamming"
    opts.frame_opts.snip_edges = True
    opts.mel_opts.num_bins = 80
    f = knf.OnlineFbank(opts)
    f.accept_waveform(SR, (np.asarray(audio, dtype=np.float32) * 32768.0).tolist())
    f.input_finished()
    feats = np.stack([f.get_frame(i) for i in range(f.num_frames_ready)]).astype(np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


class SpeakerEmbedder:
    def __init__(self, model_path: Path, session=None) -> None:
        self.model_path = Path(model_path)
        self._sess = session

    def load(self) -> None:
        if self._sess is not None:
            return
        if not self.model_path.is_file():
            raise ModelMissingError(f"Thiếu mô hình nhận diện người nói: {self.model_path}")
        import onnxruntime

        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self._sess = onnxruntime.InferenceSession(
            str(self.model_path), providers=["CPUExecutionProvider"], sess_options=opts
        )

    def embed(self, audio: np.ndarray) -> np.ndarray:
        if self._sess is None:
            self.load()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.shape[0] < MIN_SAMPLES:
            audio = np.pad(audio, (0, MIN_SAMPLES - audio.shape[0]))
        out = self._sess.run(None, {"feats": fbank(audio)[None, :, :]})[0]
        v = np.asarray(out, dtype=np.float32).reshape(-1)
        return v / max(float(np.linalg.norm(v)), 1e-12)
