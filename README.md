# noiseclarity
An application to reduce background noise. Applicable for sports.
# ClearVoice — RNNoise AI Audio Denoiser

> Remove background noise while preserving speech quality.
> Powered by the [RNNoise](https://github.com/xiph/rnnoise) recurrent neural network, compiled to WebAssembly.

---

## Quick Start

**Option A — Standalone HTML (no build required)**
```bash
# Just open in a browser:
open clearvoice.html
```

**Option B — React app (full project)**
```bash
npm install
npm start
# → opens at http://localhost:3000
```

---

## Architecture

```
Audio File
    │
    ▼
Web Audio API               ← Decodes any browser-supported format
    │  (AudioContext + OfflineAudioContext for resampling)
    ▼
48kHz Mono Float32 PCM
    │
    ▼
RNNoise WASM Module         ← https://github.com/xiph/rnnoise
    │  480-sample frames (~10ms)
    │  GRU recurrent network (pre-trained on speech/noise pairs)
    │  Per-band spectral gain masking
    │  VAD score emitted per frame
    ▼
Clean PCM Float32
    │
    ▼
WAV Encoder                 ← 16-bit PCM, TPDF dithering
    │
    ▼
Downloadable .wav + Audio preview
```

### WASM Fallback
When the RNNoise WASM module cannot be loaded (CDN issues, CSP headers, etc.),
the engine automatically falls back to a pure-JS **spectral subtraction** algorithm:
- Estimates noise floor from first ~12 frames
- Applies per-band SNR-based gain masking
- Same frame-based API — telemetry continues to work

---

## ML Best Practices

### Model Architecture (RNNoise)
- **Recurrent GRU network** with hidden state preserved across frames
- **480-sample frames** (10ms @ 48kHz) — matches RNNoise's design window
- **Band gain prediction** across 22 Bark-scale critical bands
- **VAD signal** emitted alongside noise-suppressed output

### Input Preprocessing
- **Sample rate normalization**: any input → 48kHz via `OfflineAudioContext`
  (uses browser's high-quality polyphase resampler)
- **Mono mix-down**: equal-power mixing: `gain = 1/sqrt(numChannels)`
- **Dynamic range scaling**: Float32 → 16-bit range before model, back after
  (RNNoise was trained on 16-bit PCM; scaling is required for correct model behavior)

### Output Post-processing
- **TPDF dithering** on WAV encode: adds ~1 LSB of shaped noise before
  quantization, reducing quantization distortion audibility

---

## MLOps Telemetry

Every run emits a structured `telemetry` object:

```json
{
  "model": {
    "version": "rnnoise-0.2-wasm",
    "type": "rnnoise-gru-wasm",
    "frameSize": 480,
    "sampleRate": 48000
  },
  "vad": {
    "scores": [...],          // Per-frame VAD probability 0–1
    "speechFrames": 142,
    "totalFrames": 200,
    "speechRatio": 0.71,
    "meanScore": 0.64
  },
  "performance": {
    "totalTimeMs": 312,
    "audioDurationS": 2.0,
    "realTimeFactor": 0.156,  // <1.0 = faster than real-time
    "frameLatency": {
      "p50": 0.08,
      "p95": 0.23,
      "maxMs": 0.45
    }
  },
  "quality": {
    "inputRMS": 0.0412,
    "outputRMS": 0.0289,
    "estimatedNoiseReductionDB": 3.1,
    "estimatedSNRDB": 14.2
  },
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

All metrics are visible in the **MLOps Telemetry panel** in the UI.

---

## File Structure

```
rnnoise-app/
├── clearvoice.html          ← Standalone single-file version
├── src/
│   ├── utils/
│   │   └── rnnoiseEngine.js ← Core processing pipeline
│   │       ├── loadRNNoiseModule()   WASM loader (singleton)
│   │       ├── decodeAudio()         Any format → 48kHz mono PCM
│   │       ├── denoisePCM()          Frame processing + telemetry
│   │       ├── encodePCMToWAV()      Float32 → 16-bit WAV
│   │       ├── computeWaveformData() RMS waveform for display
│   │       └── computeSpectrum()     DFT spectrum for display
│   ├── hooks/
│   │   └── useDenoiser.js   ← React hook (full state machine)
│   ├── components/
│   │   ├── WaveformCanvas.js ← Canvas waveform + VAD + spectrum
│   │   └── TelemetryPanel.js ← MLOps metrics UI
│   ├── App.js               ← Main React component
│   └── App.css
└── README.md
```

---

## Browser Support

| Feature | Chrome | Firefox | Safari | Edge |
|---|---|---|---|---|
| Web Audio API | ✓ | ✓ | ✓ | ✓ |
| OfflineAudioContext | ✓ | ✓ | ✓ | ✓ |
| WebAssembly | ✓ | ✓ | ✓ | ✓ |
| ES Modules (dynamic import) | ✓ | ✓ | ✓ | ✓ |

Minimum: Chrome 66+, Firefox 60+, Safari 14+, Edge 79+

---

## Supported Audio Formats

Any format supported by the browser's `AudioContext.decodeAudioData`:
WAV, MP3, OGG, FLAC, AAC, M4A, WebM

Output is always 16-bit 48kHz mono WAV.

---

## License

The RNNoise model weights and C code are BSD-3-Clause licensed by the Xiph.Org Foundation.
See https://github.com/xiph/rnnoise/blob/main/COPYING
