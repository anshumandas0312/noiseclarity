import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useDenoiser, STATES } from './hooks/useDenoiser';
import { WaveformCanvas } from './components/WaveformCanvas';
import { TelemetryPanel } from './components/TelemetryPanel';
import './App.css';

const ACCEPTED_FORMATS = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/ogg', 'audio/flac', 'audio/aac', 'audio/x-wav'];

// ─── Sub-components ────────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const colors = {
    [STATES.IDLE]: '#444',
    [STATES.LOADING_MODEL]: '#f59e0b',
    [STATES.DECODING]: '#818cf8',
    [STATES.PROCESSING]: '#4ade80',
    [STATES.COMPLETE]: '#4ade80',
    [STATES.ERROR]: '#f87171',
  };
  const pulse = [STATES.PROCESSING, STATES.DECODING, STATES.LOADING_MODEL].includes(status);

  return (
    <span className={`status-dot${pulse ? ' pulse' : ''}`} style={{ background: colors[status] || '#444' }} />
  );
}

function StatusLabel({ status }) {
  const labels = {
    [STATES.IDLE]: 'Ready',
    [STATES.LOADING_MODEL]: 'Loading model…',
    [STATES.DECODING]: 'Decoding audio…',
    [STATES.PROCESSING]: 'Processing frames…',
    [STATES.COMPLETE]: 'Complete',
    [STATES.ERROR]: 'Error',
  };
  return <span>{labels[status] || status}</span>;
}

function ProgressBar({ progress, status }) {
  const isActive = [STATES.PROCESSING, STATES.DECODING, STATES.LOADING_MODEL].includes(status);
  return (
    <div className="progress-track">
      <div
        className="progress-fill"
        style={{
          width: `${progress}%`,
          background: status === STATES.ERROR
            ? 'rgba(248,113,113,0.6)'
            : status === STATES.COMPLETE
            ? 'rgba(74,222,128,0.5)'
            : 'rgba(129,140,248,0.6)',
          transition: isActive ? 'width 0.3s ease' : 'none',
        }}
      />
    </div>
  );
}

function AudioPlayer({ url, label, accentColor = '#818cf8' }) {
  return (
    <div className="audio-player">
      <span style={{ fontSize: 10, color: 'rgba(255,255,255,0.35)', textTransform: 'uppercase', letterSpacing: '0.1em' }}>
        {label}
      </span>
      <audio
        controls
        src={url}
        style={{ width: '100%', height: 32, filter: 'invert(0.85) hue-rotate(200deg)' }}
      />
    </div>
  );
}

function FormatBadge({ type }) {
  const fmt = (type || '').split('/').pop().toUpperCase().replace('MPEG', 'MP3');
  return (
    <span style={{
      fontSize: 9,
      fontFamily: 'monospace',
      padding: '2px 6px',
      background: 'rgba(129,140,248,0.12)',
      color: '#818cf8',
      borderRadius: 3,
      letterSpacing: '0.05em',
    }}>
      {fmt || 'AUDIO'}
    </span>
  );
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// ─── Drop Zone ─────────────────────────────────────────────────────────────────
function DropZone({ onFile, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    if (disabled) return;
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }, [onFile, disabled]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    if (!disabled) setDragging(true);
  }, [disabled]);

  const handleDragLeave = useCallback(() => setDragging(false), []);

  const handleChange = useCallback((e) => {
    const file = e.target.files?.[0];
    if (file) onFile(file);
  }, [onFile]);

  return (
    <div
      className={`drop-zone ${dragging ? 'drag-active' : ''} ${disabled ? 'disabled' : ''}`}
      onDrop={handleDrop}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept="audio/*"
        style={{ display: 'none' }}
        onChange={handleChange}
      />
      <div className="drop-icon">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
          <path d="M9 18V5l12-2v13"/>
          <circle cx="6" cy="18" r="3"/>
          <circle cx="18" cy="16" r="3"/>
        </svg>
      </div>
      <div className="drop-title">Drop audio file here</div>
      <div className="drop-sub">or click to browse — WAV · MP3 · OGG · FLAC · AAC</div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────────
function App() {
  const {
    status,
    progress,
    currentFrame,
    totalFrames,
    currentVAD,
    error,
    inputFile,
    inputMetadata,
    inputWaveform,
    inputSpectrum,
    outputURL,
    outputWaveform,
    outputSpectrum,
    telemetry,
    isProcessing,
    isDone,
    processFile,
    downloadOutput,
    cancel,
    reset,
    preloadModel,
  } = useDenoiser();

  // Pre-warm WASM on mount
  useEffect(() => {
    preloadModel();
  }, [preloadModel]);

  const handleFile = useCallback((file) => {
    if (!ACCEPTED_FORMATS.includes(file.type) && !file.name.match(/\.(wav|mp3|ogg|flac|aac|m4a)$/i)) {
      alert('Please upload an audio file (WAV, MP3, OGG, FLAC, AAC)');
      return;
    }
    processFile(file);
  }, [processFile]);

  const showDropZone = status === STATES.IDLE || status === STATES.ERROR;
  const showProgress = isProcessing;
  const showResults = isDone;

  return (
    <div className="app">
      {/* ── Header ── */}
      <header className="header">
        <div className="header-inner">
          <div className="logo">
            <div className="logo-mark">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
                <path d="M2 13s3-6 10-6 10 6 10 6-3 6-10 6-10-6-10-6z" stroke="#4ade80" strokeWidth="1.5"/>
                <circle cx="12" cy="13" r="3" stroke="#4ade80" strokeWidth="1.5"/>
                <path d="M12 4V2M12 24v-2M4 13H2M22 13h-2" stroke="#4ade80" strokeWidth="1" opacity="0.5"/>
              </svg>
            </div>
            <div>
              <span className="logo-name">ClearVoice</span>
              <span className="logo-tag">RNNoise · WASM · 48kHz</span>
            </div>
          </div>
          <div className="header-status">
            <StatusDot status={status} />
            <span className="status-text"><StatusLabel status={status} /></span>
            {isProcessing && (
              <button className="btn-ghost btn-sm" onClick={cancel}>cancel</button>
            )}
            {(isDone || status === STATES.ERROR) && (
              <button className="btn-ghost btn-sm" onClick={reset}>reset</button>
            )}
          </div>
        </div>
      </header>

      <main className="main">

        {/* ── Error Banner ── */}
        {error && (
          <div className="error-banner">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
            {error}
          </div>
        )}

        {/* ── Drop Zone ── */}
        {showDropZone && (
          <DropZone onFile={handleFile} disabled={isProcessing} />
        )}

        {/* ── Processing View ── */}
        {(isProcessing || inputFile) && (
          <div className="card">
            <div className="card-row">
              <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                <FormatBadge type={inputFile?.type} />
                <span className="file-name">{inputFile?.name}</span>
              </div>
              <span className="file-meta">
                {inputMetadata
                  ? `${inputMetadata.duration}s · ${inputMetadata.originalSampleRate / 1000}kHz · ${formatBytes(inputFile?.size)}`
                  : formatBytes(inputFile?.size)
                }
              </span>
            </div>

            {showProgress && (
              <>
                <ProgressBar progress={progress} status={status} />
                <div className="progress-stats">
                  <span>{status === STATES.LOADING_MODEL ? 'Initializing WASM module…' : status === STATES.DECODING ? 'Decoding to 48kHz PCM…' : `Frame ${currentFrame} / ${totalFrames}`}</span>
                  <span className="vad-live">
                    VAD {(currentVAD * 100).toFixed(0)}%
                    <span className={`vad-dot ${currentVAD > 0.5 ? 'speech' : 'noise'}`} />
                  </span>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── Waveform Comparison ── */}
        {(inputWaveform || outputWaveform) && (
          <div className="card">
            <div className="section-label">Waveform comparison</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <WaveformCanvas
                data={inputWaveform}
                color="#818cf8"
                height={72}
                label="Input (noisy)"
              />
              {outputWaveform && (
                <WaveformCanvas
                  data={outputWaveform}
                  color="#4ade80"
                  height={72}
                  label="Output (clean)"
                  highlight
                />
              )}
            </div>
          </div>
        )}

        {/* ── Results ── */}
        {showResults && (
          <div className="card results-card">
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="success-icon">✓</span>
                <span style={{ fontWeight: 500, color: '#4ade80' }}>Noise removed</span>
              </div>
              <button className="btn-primary" onClick={downloadOutput}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4"/><polyline points="7,10 12,15 17,10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Download WAV
              </button>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
              <AudioPlayer url={URL.createObjectURL(inputFile)} label="Original" />
              <AudioPlayer url={outputURL} label="Denoised" accentColor="#4ade80" />
            </div>
          </div>
        )}

        {/* ── MLOps Telemetry ── */}
        {telemetry && (
          <TelemetryPanel
            telemetry={telemetry}
            inputSpectrum={inputSpectrum}
            outputSpectrum={outputSpectrum}
          />
        )}

        {/* ── Info Footer ── */}
        {status === STATES.IDLE && !inputFile && (
          <div className="info-grid">
            <div className="info-card">
              <div className="info-icon">🧠</div>
              <div className="info-title">RNNoise GRU Model</div>
              <div className="info-desc">Recurrent neural network trained on thousands of speech/noise pairs. Runs at 480-sample frames (10ms) preserving GRU state across the full file.</div>
            </div>
            <div className="info-card">
              <div className="info-icon">⚡</div>
              <div className="info-title">WebAssembly Runtime</div>
              <div className="info-desc">Compiled from C via Emscripten. Runs near-native speed in the browser — no server upload, fully private, works offline.</div>
            </div>
            <div className="info-card">
              <div className="info-icon">📊</div>
              <div className="info-title">MLOps Telemetry</div>
              <div className="info-desc">Per-frame VAD scores, SNR estimation, latency percentiles, noise reduction in dB — full observability on every run.</div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
