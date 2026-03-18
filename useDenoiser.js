/**
 * useDenoiser — React hook for the RNNoise denoising pipeline
 *
 * Manages all async state: file loading, decoding, processing, download.
 * Follows MLOps patterns: captures telemetry, surfaces model metadata.
 */

import { useState, useCallback, useRef } from 'react';
import {
  decodeAudio,
  denoisePCM,
  encodePCMToWAV,
  computeWaveformData,
  computeSpectrum,
  loadRNNoiseModule,
} from '../utils/rnnoiseEngine';

export const STATES = {
  IDLE: 'idle',
  LOADING_MODEL: 'loading_model',
  DECODING: 'decoding',
  PROCESSING: 'processing',
  COMPLETE: 'complete',
  ERROR: 'error',
};

const initialState = {
  status: STATES.IDLE,
  progress: 0,
  currentFrame: 0,
  totalFrames: 0,
  currentVAD: 0,
  error: null,
  // Input file metadata
  inputFile: null,
  inputMetadata: null,       // { duration, sampleRate, channels, size }
  inputWaveform: null,       // Float32Array for waveform display
  inputSpectrum: null,
  // Output
  outputBlob: null,
  outputURL: null,
  outputWaveform: null,
  outputSpectrum: null,
  // MLOps
  telemetry: null,
  modelReady: false,
};

export function useDenoiser() {
  const [state, setState] = useState(initialState);
  const abortRef = useRef(false);

  // Merge partial state updates
  const update = useCallback((patch) => {
    setState(prev => ({ ...prev, ...patch }));
  }, []);

  /**
   * Pre-warm the WASM module so first-run latency is hidden.
   */
  const preloadModel = useCallback(async () => {
    try {
      update({ status: STATES.LOADING_MODEL });
      await loadRNNoiseModule();
      update({ status: STATES.IDLE, modelReady: true });
    } catch (err) {
      console.warn('[useDenoiser] Model preload failed (will load on demand):', err);
      update({ status: STATES.IDLE });
    }
  }, [update]);

  /**
   * Process an audio file through the RNNoise pipeline.
   * @param {File} file - Audio file from file input or drag-drop
   */
  const processFile = useCallback(async (file) => {
    if (!file) return;
    abortRef.current = false;

    // Reset state
    update({
      status: STATES.LOADING_MODEL,
      progress: 0,
      error: null,
      inputFile: file,
      inputMetadata: null,
      inputWaveform: null,
      inputSpectrum: null,
      outputBlob: null,
      outputURL: null,
      outputWaveform: null,
      outputSpectrum: null,
      telemetry: null,
    });

    try {
      // 1. Load WASM module (cached after first load)
      await loadRNNoiseModule();
      if (abortRef.current) return;

      // 2. Read file as ArrayBuffer
      update({ status: STATES.DECODING, progress: 5 });
      const arrayBuffer = await file.arrayBuffer();
      if (abortRef.current) return;

      // 3. Decode audio to 48kHz mono PCM
      const { pcm, originalSampleRate, duration } = await decodeAudio(arrayBuffer);
      if (abortRef.current) return;

      // Compute input waveform & spectrum for display
      const inputWaveform = computeWaveformData(pcm, 512);
      const inputSpectrum = computeSpectrum(pcm, 64);

      update({
        progress: 15,
        inputMetadata: {
          duration: Math.round(duration * 100) / 100,
          originalSampleRate,
          channels: 1, // after mono mix-down
          size: file.size,
          name: file.name,
          type: file.type,
        },
        inputWaveform,
        inputSpectrum,
        status: STATES.PROCESSING,
        totalFrames: Math.ceil(pcm.length / 480),
      });

      // 4. Run RNNoise denoising with progress tracking
      const { cleanPCM, telemetry } = await denoisePCM(pcm, (done, total, vad) => {
        if (abortRef.current) return;
        const progressPct = 15 + Math.round((done / total) * 80);
        update({
          progress: progressPct,
          currentFrame: done,
          totalFrames: total,
          currentVAD: Math.round(vad * 100) / 100,
        });
      });

      if (abortRef.current) return;

      // 5. Encode output to WAV
      const wavBuffer = encodePCMToWAV(cleanPCM, 48000);
      const outputBlob = new Blob([wavBuffer], { type: 'audio/wav' });
      const outputURL = URL.createObjectURL(outputBlob);

      // Compute output waveform & spectrum
      const outputWaveform = computeWaveformData(cleanPCM, 512);
      const outputSpectrum = computeSpectrum(cleanPCM, 64);

      update({
        status: STATES.COMPLETE,
        progress: 100,
        outputBlob,
        outputURL,
        outputWaveform,
        outputSpectrum,
        telemetry,
        modelReady: true,
      });

    } catch (err) {
      console.error('[useDenoiser] Pipeline error:', err);
      update({
        status: STATES.ERROR,
        error: err.message || 'Processing failed',
        progress: 0,
      });
    }
  }, [update]);

  /**
   * Trigger download of the denoised WAV file.
   */
  const downloadOutput = useCallback(() => {
    if (!state.outputURL || !state.inputFile) return;
    const baseName = state.inputFile.name.replace(/\.[^/.]+$/, '');
    const a = document.createElement('a');
    a.href = state.outputURL;
    a.download = `${baseName}_denoised.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [state.outputURL, state.inputFile]);

  /**
   * Cancel in-flight processing.
   */
  const cancel = useCallback(() => {
    abortRef.current = true;
    update({ status: STATES.IDLE, progress: 0 });
  }, [update]);

  /**
   * Reset to initial state.
   */
  const reset = useCallback(() => {
    abortRef.current = true;
    if (state.outputURL) URL.revokeObjectURL(state.outputURL);
    setState(initialState);
  }, [state.outputURL]);

  return {
    ...state,
    processFile,
    downloadOutput,
    cancel,
    reset,
    preloadModel,
    isProcessing: state.status === STATES.PROCESSING || state.status === STATES.DECODING || state.status === STATES.LOADING_MODEL,
    isDone: state.status === STATES.COMPLETE,
    isError: state.status === STATES.ERROR,
  };
}
