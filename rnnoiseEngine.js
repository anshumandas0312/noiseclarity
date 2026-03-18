/**
 * RNNoise Processing Engine
 *
 * Architecture:
 *   - Loads RNNoise compiled to WebAssembly via @jitsi/rnnoise-wasm
 *   - Decodes audio with Web Audio API (any format → 48kHz mono Float32)
 *   - Processes in 480-sample frames (~10ms each) matching RNNoise's GRU window
 *   - Tracks per-frame Voice Activity Detection (VAD) scores emitted by the model
 *   - Encodes clean output as 16-bit PCM WAV
 *
 * MLOps instrumentation:
 *   - Frame-level VAD scores (speech probability 0–1)
 *   - SNR estimation pre/post
 *   - Processing latency per batch
 *   - Model version tracking
 */

// ─── Constants ────────────────────────────────────────────────────────────────
export const RNNOISE_SAMPLE_RATE = 48000;  // RNNoise only supports 48kHz
export const FRAME_SIZE = 480;             // 10ms @ 48kHz — fixed by RNNoise architecture
export const MODEL_VERSION = "rnnoise-0.2-wasm";

// ─── WASM loader (lazy singleton) ─────────────────────────────────────────────
let rnnoiseModule = null;
let moduleLoadPromise = null;

/**
 * Load RNNoise WASM module (singleton — loads once, reused across calls).
 * Uses the Jitsi build which includes the pre-trained RNNoise model weights.
 */
export async function loadRNNoiseModule() {
  if (rnnoiseModule) return rnnoiseModule;
  if (moduleLoadPromise) return moduleLoadPromise;

  moduleLoadPromise = (async () => {
    try {
      // Dynamic import of the Jitsi rnnoise-wasm package
      // The package exposes a factory function that initializes the WASM module
      const RNNoiseFactory = await import(
        /* webpackChunkName: "rnnoise-wasm" */
        'https://cdn.jsdelivr.net/npm/@jitsi/rnnoise-wasm@0.0.2/dist/index.js'
      ).catch(() => null);

      if (!RNNoiseFactory) {
        // Fallback: use unpkg
        throw new Error('CDN load failed');
      }

      rnnoiseModule = RNNoiseFactory.default
        ? await RNNoiseFactory.default()
        : await RNNoiseFactory();

      console.info(`[RNNoise] WASM module loaded — ${MODEL_VERSION}`);
      return rnnoiseModule;
    } catch (err) {
      console.warn('[RNNoise] WASM load via CDN failed, using polyfill engine:', err.message);
      // Return a JS polyfill that applies spectral subtraction
      // This ensures the app always works even when WASM is unavailable
      rnnoiseModule = createJSFallbackEngine();
      return rnnoiseModule;
    }
  })();

  return moduleLoadPromise;
}

// ─── JS Fallback Engine ────────────────────────────────────────────────────────
/**
 * Pure-JS noise suppression fallback using spectral subtraction.
 * Activates when WASM cannot be loaded (CORS restrictions, etc).
 * Less accurate than RNNoise but demonstrates the same frame-based API.
 */
function createJSFallbackEngine() {
  const WINDOW_SIZE = FRAME_SIZE * 2;
  const FFT_BINS = WINDOW_SIZE / 2;
  let noiseProfile = null;
  let noiseFrameCount = 0;
  const NOISE_ESTIMATE_FRAMES = 10;

  return {
    _isFallback: true,
    _version: 'spectral-subtraction-js-v1',

    createDenoiseState() {
      let localNoiseProfile = null;
      let localFrameCount = 0;

      return {
        processFrame(frame) {
          // Simple spectral subtraction:
          // 1. Estimate noise from first N frames
          // 2. Subtract noise spectrum from subsequent frames
          const magnitude = estimateMagnitude(frame);

          if (localFrameCount < NOISE_ESTIMATE_FRAMES) {
            if (!localNoiseProfile) {
              localNoiseProfile = new Float32Array(magnitude.length);
            }
            // Accumulate noise estimate
            for (let i = 0; i < magnitude.length; i++) {
              localNoiseProfile[i] += magnitude[i] / NOISE_ESTIMATE_FRAMES;
            }
            localFrameCount++;
            return 0.1; // Low VAD during noise profiling
          }

          // Apply soft-mask based on SNR per frequency bin
          let speechEnergy = 0;
          let totalEnergy = 0;
          for (let i = 0; i < frame.length; i++) {
            const noiseFloor = localNoiseProfile[Math.floor(i * FFT_BINS / frame.length)] || 0.001;
            const snr = Math.abs(frame[i]) / (noiseFloor + 1e-10);
            const gain = Math.max(0, 1 - 1 / (snr + 1));
            frame[i] *= gain;
            if (gain > 0.5) speechEnergy += frame[i] * frame[i];
            totalEnergy += frame[i] * frame[i];
          }

          // Return VAD score: ratio of post-gain speech energy
          const vad = totalEnergy > 0 ? Math.min(1, speechEnergy / totalEnergy) : 0;
          return vad;
        },
        destroy() {}
      };
    },

    destroy() {}
  };
}

function estimateMagnitude(frame) {
  // Compute rough magnitude spectrum via DFT (simplified for the fallback)
  const N = Math.min(frame.length, 64);
  const mag = new Float32Array(N / 2);
  for (let k = 0; k < N / 2; k++) {
    let re = 0, im = 0;
    for (let n = 0; n < N; n++) {
      const angle = (2 * Math.PI * k * n) / N;
      re += (frame[n] || 0) * Math.cos(angle);
      im -= (frame[n] || 0) * Math.sin(angle);
    }
    mag[k] = Math.sqrt(re * re + im * im);
  }
  return mag;
}

// ─── Audio Decoder ─────────────────────────────────────────────────────────────
/**
 * Decode any browser-supported audio format to 48kHz mono Float32Array.
 * Uses OfflineAudioContext for high-quality sample-rate conversion.
 *
 * @param {ArrayBuffer} arrayBuffer - Raw audio file bytes
 * @returns {{ pcm: Float32Array, originalSampleRate: number, duration: number }}
 */
export async function decodeAudio(arrayBuffer) {
  // Decode at native sample rate first to get duration
  const tempCtx = new AudioContext();
  const originalBuffer = await tempCtx.decodeAudioData(arrayBuffer.slice(0));
  await tempCtx.close();

  const originalSampleRate = originalBuffer.sampleRate;
  const duration = originalBuffer.duration;
  const numChannels = originalBuffer.numberOfChannels;

  // If already 48kHz, just down-mix to mono
  if (originalSampleRate === RNNOISE_SAMPLE_RATE) {
    const mono = mixDownToMono(originalBuffer, numChannels);
    return { pcm: mono, originalSampleRate, duration };
  }

  // Resample to 48kHz using OfflineAudioContext
  const outputLength = Math.ceil(duration * RNNOISE_SAMPLE_RATE);
  const offlineCtx = new OfflineAudioContext(1, outputLength, RNNOISE_SAMPLE_RATE);

  const sourceNode = offlineCtx.createBufferSource();
  sourceNode.buffer = originalBuffer;
  sourceNode.connect(offlineCtx.destination);
  sourceNode.start(0);

  const renderedBuffer = await offlineCtx.startRendering();
  const pcm = renderedBuffer.getChannelData(0);

  return { pcm, originalSampleRate, duration };
}

/**
 * Mix a multi-channel AudioBuffer down to mono Float32Array.
 * Equal-power mixing: divides by sqrt(numChannels).
 */
function mixDownToMono(audioBuffer, numChannels) {
  const length = audioBuffer.length;
  const mono = new Float32Array(length);
  const gainFactor = 1 / Math.sqrt(numChannels);

  for (let ch = 0; ch < numChannels; ch++) {
    const channelData = audioBuffer.getChannelData(ch);
    for (let i = 0; i < length; i++) {
      mono[i] += channelData[i] * gainFactor;
    }
  }
  return mono;
}

// ─── Core Denoising Pipeline ───────────────────────────────────────────────────
/**
 * Main denoising function.
 *
 * @param {Float32Array} pcm - 48kHz mono audio
 * @param {Function} onProgress - Called with (framesProcessed, totalFrames, vadScore)
 * @returns {{ cleanPCM: Float32Array, telemetry: ProcessingTelemetry }}
 */
export async function denoisePCM(pcm, onProgress) {
  const module = await loadRNNoiseModule();
  const startTime = performance.now();

  const totalFrames = Math.ceil(pcm.length / FRAME_SIZE);
  const cleanPCM = new Float32Array(pcm.length);

  // MLOps telemetry accumulators
  const vadScores = new Float32Array(totalFrames);
  const frameLatencies = [];
  let inputRMS = 0;
  let outputRMS = 0;

  // Compute input RMS for SNR estimation
  for (let i = 0; i < pcm.length; i++) {
    inputRMS += pcm[i] * pcm[i];
  }
  inputRMS = Math.sqrt(inputRMS / pcm.length);

  // Create RNNoise state (holds GRU hidden state across frames)
  const denoiseState = module.createDenoiseState();

  try {
    for (let frameIdx = 0; frameIdx < totalFrames; frameIdx++) {
      const frameStart = performance.now();
      const offset = frameIdx * FRAME_SIZE;

      // Extract frame — pad with zeros if last frame is short
      const frame = new Float32Array(FRAME_SIZE);
      const available = Math.min(FRAME_SIZE, pcm.length - offset);
      frame.set(pcm.subarray(offset, offset + available));

      // Scale to RNNoise's expected range [-32768, 32767]
      // (RNNoise was trained on 16-bit PCM; Float32 input [-1, 1] must be scaled)
      for (let i = 0; i < FRAME_SIZE; i++) {
        frame[i] *= 32768;
      }

      // ─── RNNOISE INFERENCE ───
      const vadScore = denoiseState.processFrame(frame);
      // ─────────────────────────

      // Scale back to [-1, 1] and copy to output
      for (let i = 0; i < available; i++) {
        cleanPCM[offset + i] = frame[i] / 32768;
      }

      // Record telemetry
      vadScores[frameIdx] = vadScore;
      frameLatencies.push(performance.now() - frameStart);

      // Progress callback every 50 frames (~500ms of audio)
      if (frameIdx % 50 === 0 || frameIdx === totalFrames - 1) {
        onProgress?.(frameIdx + 1, totalFrames, vadScore);
        // Yield to event loop to keep UI responsive
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }
  } finally {
    denoiseState.destroy();
  }

  // Compute output RMS
  for (let i = 0; i < cleanPCM.length; i++) {
    outputRMS += cleanPCM[i] * cleanPCM[i];
  }
  outputRMS = Math.sqrt(outputRMS / cleanPCM.length);

  const totalTime = performance.now() - startTime;

  // ─── MLOps Telemetry Report ───
  const telemetry = buildTelemetry({
    vadScores,
    frameLatencies,
    inputRMS,
    outputRMS,
    totalFrames,
    totalTime,
    pcmLength: pcm.length,
    isFallback: module._isFallback || false,
    modelVersion: module._isFallback ? module._version : MODEL_VERSION,
  });

  return { cleanPCM, telemetry };
}

// ─── Telemetry Builder ─────────────────────────────────────────────────────────
function buildTelemetry({ vadScores, frameLatencies, inputRMS, outputRMS, totalFrames, totalTime, pcmLength, isFallback, modelVersion }) {
  // VAD statistics
  let speechFrames = 0;
  let vadSum = 0;
  const VAD_THRESHOLD = 0.5;
  for (let i = 0; i < vadScores.length; i++) {
    if (vadScores[i] > VAD_THRESHOLD) speechFrames++;
    vadSum += vadScores[i];
  }

  // Frame latency statistics
  const sortedLatencies = [...frameLatencies].sort((a, b) => a - b);
  const p50 = sortedLatencies[Math.floor(sortedLatencies.length * 0.5)] || 0;
  const p95 = sortedLatencies[Math.floor(sortedLatencies.length * 0.95)] || 0;
  const maxLatency = sortedLatencies[sortedLatencies.length - 1] || 0;

  // SNR estimation (dB)
  const snrDB = inputRMS > 1e-10 && outputRMS > 1e-10
    ? 20 * Math.log10(outputRMS / (inputRMS - outputRMS + 1e-10))
    : 0;

  // Noise reduction estimate (energy removed)
  const noiseReductionDB = inputRMS > 1e-10
    ? Math.max(0, 20 * Math.log10(inputRMS / (outputRMS + 1e-10)))
    : 0;

  const audioDuration = pcmLength / RNNOISE_SAMPLE_RATE;
  const realTimeFactor = totalTime / 1000 / audioDuration;

  return {
    model: {
      version: modelVersion,
      type: isFallback ? 'spectral-subtraction-js' : 'rnnoise-gru-wasm',
      frameSize: FRAME_SIZE,
      sampleRate: RNNOISE_SAMPLE_RATE,
    },
    vad: {
      scores: Array.from(vadScores),
      speechFrames,
      totalFrames,
      speechRatio: speechFrames / totalFrames,
      meanScore: vadSum / totalFrames,
    },
    performance: {
      totalTimeMs: Math.round(totalTime),
      audioDurationS: Math.round(audioDuration * 100) / 100,
      realTimeFactor: Math.round(realTimeFactor * 1000) / 1000,
      frameLatency: { p50: Math.round(p50 * 100) / 100, p95: Math.round(p95 * 100) / 100, maxMs: Math.round(maxLatency * 100) / 100 },
    },
    quality: {
      inputRMS: Math.round(inputRMS * 10000) / 10000,
      outputRMS: Math.round(outputRMS * 10000) / 10000,
      estimatedNoiseReductionDB: Math.round(noiseReductionDB * 10) / 10,
      estimatedSNRDB: Math.round(snrDB * 10) / 10,
    },
    timestamp: new Date().toISOString(),
  };
}

// ─── WAV Encoder ───────────────────────────────────────────────────────────────
/**
 * Encode Float32Array PCM to a standard 16-bit PCM WAV ArrayBuffer.
 * WAV format: RIFF header + fmt chunk + data chunk.
 *
 * @param {Float32Array} pcm - Normalized [-1, 1] audio
 * @param {number} sampleRate - Sample rate in Hz
 * @returns {ArrayBuffer}
 */
export function encodePCMToWAV(pcm, sampleRate = RNNOISE_SAMPLE_RATE) {
  const numSamples = pcm.length;
  const numChannels = 1;
  const bitsPerSample = 16;
  const bytesPerSample = bitsPerSample / 8;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;
  const bufferSize = 44 + dataSize;

  const buffer = new ArrayBuffer(bufferSize);
  const view = new DataView(buffer);

  // RIFF header
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);          // chunk size
  view.setUint16(20, 1, true);           // PCM = 1
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Convert Float32 → Int16 with dithering
  const int16View = new Int16Array(buffer, 44);
  for (let i = 0; i < numSamples; i++) {
    // Clamp + dither (TPDF dithering for better perceptual quality)
    const dither = (Math.random() + Math.random() - 1) / 32768;
    const sample = Math.max(-1, Math.min(1, pcm[i] + dither));
    int16View[i] = Math.round(sample * 32767);
  }

  return buffer;
}

function writeString(view, offset, str) {
  for (let i = 0; i < str.length; i++) {
    view.setUint8(offset + i, str.charCodeAt(i));
  }
}

// ─── Waveform Analysis ─────────────────────────────────────────────────────────
/**
 * Downsample PCM to a fixed number of waveform display points.
 * Returns RMS amplitude per bucket for waveform visualisation.
 *
 * @param {Float32Array} pcm
 * @param {number} numPoints - Number of display points (e.g., 512)
 * @returns {Float32Array}
 */
export function computeWaveformData(pcm, numPoints = 512) {
  const blockSize = Math.floor(pcm.length / numPoints);
  const waveform = new Float32Array(numPoints);

  for (let i = 0; i < numPoints; i++) {
    let rms = 0;
    const start = i * blockSize;
    const end = Math.min(start + blockSize, pcm.length);
    for (let j = start; j < end; j++) {
      rms += pcm[j] * pcm[j];
    }
    waveform[i] = Math.sqrt(rms / (end - start));
  }

  return waveform;
}

/**
 * Compute frequency spectrum for a segment of audio (for spectrum display).
 * Uses a simple DFT on a windowed segment.
 *
 * @param {Float32Array} pcm
 * @param {number} numBands - Number of frequency bands
 * @returns {Float32Array} - Magnitude in dB per band
 */
export function computeSpectrum(pcm, numBands = 64) {
  const N = Math.min(pcm.length, 2048);
  const segment = pcm.subarray(Math.max(0, Math.floor(pcm.length / 2) - N / 2), Math.floor(pcm.length / 2) + N / 2);

  // Apply Hann window
  const windowed = new Float32Array(segment.length);
  for (let i = 0; i < segment.length; i++) {
    windowed[i] = segment[i] * 0.5 * (1 - Math.cos(2 * Math.PI * i / (segment.length - 1)));
  }

  // DFT (simplified, only compute numBands bins)
  const spectrum = new Float32Array(numBands);
  const binStep = Math.floor(N / 2 / numBands);
  for (let k = 0; k < numBands; k++) {
    let re = 0, im = 0;
    const freq = k * binStep;
    for (let n = 0; n < windowed.length; n++) {
      const angle = (2 * Math.PI * freq * n) / windowed.length;
      re += windowed[n] * Math.cos(angle);
      im -= windowed[n] * Math.sin(angle);
    }
    const magnitude = Math.sqrt(re * re + im * im) / windowed.length;
    spectrum[k] = magnitude > 1e-10 ? 20 * Math.log10(magnitude) : -80;
  }

  return spectrum;
}
