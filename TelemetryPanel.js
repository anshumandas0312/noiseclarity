import React, { useState } from 'react';
import { VADChart, SpectrumBars } from './WaveformCanvas';

const MetricCard = ({ label, value, unit, sub, accent }) => (
  <div style={{
    background: 'rgba(255,255,255,0.03)',
    border: `1px solid ${accent ? 'rgba(74,222,128,0.2)' : 'rgba(255,255,255,0.06)'}`,
    borderRadius: 8,
    padding: '12px 14px',
    minWidth: 0,
  }}>
    <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.4)', letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 4 }}>
      {label}
    </div>
    <div style={{ fontSize: 22, fontWeight: 500, color: accent ? '#4ade80' : 'rgba(255,255,255,0.9)', lineHeight: 1 }}>
      {value}
      {unit && <span style={{ fontSize: 12, fontWeight: 400, color: 'rgba(255,255,255,0.4)', marginLeft: 3 }}>{unit}</span>}
    </div>
    {sub && <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.3)', marginTop: 3 }}>{sub}</div>}
  </div>
);

const Badge = ({ children, color = 'rgba(255,255,255,0.1)', textColor = 'rgba(255,255,255,0.6)' }) => (
  <span style={{
    background: color,
    color: textColor,
    fontSize: 10,
    fontFamily: 'monospace',
    padding: '2px 8px',
    borderRadius: 4,
    letterSpacing: '0.05em',
    display: 'inline-block',
  }}>
    {children}
  </span>
);

const Section = ({ title, children }) => (
  <div style={{ marginBottom: 20 }}>
    <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', letterSpacing: '0.12em', textTransform: 'uppercase', marginBottom: 10, display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
      {title}
      <span style={{ flex: 1, height: 1, background: 'rgba(255,255,255,0.06)' }} />
    </div>
    {children}
  </div>
);

export function TelemetryPanel({ telemetry, inputSpectrum, outputSpectrum }) {
  const [expanded, setExpanded] = useState(false);
  if (!telemetry) return null;

  const { model, vad, performance: perf, quality } = telemetry;
  const speechPct = Math.round(vad.speechRatio * 100);

  return (
    <div style={{
      background: 'rgba(10,12,18,0.8)',
      border: '1px solid rgba(255,255,255,0.08)',
      borderRadius: 12,
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div
        style={{ padding: '14px 18px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 10, borderBottom: '1px solid rgba(255,255,255,0.05)' }}
        onClick={() => setExpanded(e => !e)}
      >
        <div style={{ width: 8, height: 8, borderRadius: '50%', background: '#4ade80', boxShadow: '0 0 6px #4ade8088' }} />
        <span style={{ fontSize: 12, fontWeight: 500, color: 'rgba(255,255,255,0.8)', letterSpacing: '0.05em' }}>
          MLOps Telemetry
        </span>
        <Badge color="rgba(74,222,128,0.1)" textColor="#4ade80">{model.type}</Badge>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: 'rgba(255,255,255,0.3)' }}>
          {expanded ? '▲ collapse' : '▼ expand'}
        </span>
      </div>

      {/* Summary row — always visible */}
      <div style={{ padding: '12px 18px', display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 10 }}>
        <MetricCard
          label="Noise reduced"
          value={quality.estimatedNoiseReductionDB}
          unit="dB"
          accent={quality.estimatedNoiseReductionDB > 3}
        />
        <MetricCard
          label="Speech frames"
          value={speechPct}
          unit="%"
          sub={`${vad.speechFrames} / ${vad.totalFrames}`}
        />
        <MetricCard
          label="Processing time"
          value={perf.totalTimeMs < 1000 ? perf.totalTimeMs + 'ms' : (perf.totalTimeMs / 1000).toFixed(1) + 's'}
          sub={`${perf.realTimeFactor}× realtime`}
        />
        <MetricCard
          label="Audio duration"
          value={perf.audioDurationS}
          unit="s"
          sub={`${model.sampleRate / 1000}kHz`}
        />
      </div>

      {expanded && (
        <div style={{ padding: '4px 18px 18px' }}>

          <Section title="Voice activity detection">
            <div style={{ marginBottom: 6, display: 'flex', justifyContent: 'space-between', fontSize: 11, color: 'rgba(255,255,255,0.35)' }}>
              <span>Frame-level VAD scores (0–1)</span>
              <span>Mean: {(vad.meanScore * 100).toFixed(0)}% · Threshold: 0.5</span>
            </div>
            <VADChart scores={vad.scores} height={72} />
          </Section>

          {(inputSpectrum || outputSpectrum) && (
            <Section title="Frequency spectrum">
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div>
                  <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginBottom: 5 }}>Input (noisy)</div>
                  <SpectrumBars data={inputSpectrum} height={56} />
                </div>
                <div>
                  <div style={{ fontSize: 10, color: 'rgba(255,255,255,0.3)', marginBottom: 5 }}>Output (clean)</div>
                  <SpectrumBars data={outputSpectrum} height={56} />
                </div>
              </div>
            </Section>
          )}

          <Section title="Model metadata">
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <Badge>{model.version}</Badge>
              <Badge>frame {model.frameSize}smp</Badge>
              <Badge>{model.sampleRate / 1000}kHz</Badge>
              <Badge>GRU architecture</Badge>
              <Badge color="rgba(129,140,248,0.1)" textColor="#818cf8">BSD-3 licensed</Badge>
            </div>
          </Section>

          <Section title="Frame latency">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
              <MetricCard label="p50 latency" value={perf.frameLatency.p50} unit="ms" />
              <MetricCard label="p95 latency" value={perf.frameLatency.p95} unit="ms" />
              <MetricCard label="max latency" value={perf.frameLatency.maxMs} unit="ms" />
            </div>
          </Section>

          <Section title="Signal quality">
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 10 }}>
              <MetricCard label="Input RMS" value={(quality.inputRMS * 100).toFixed(2)} unit="%" />
              <MetricCard label="Output RMS" value={(quality.outputRMS * 100).toFixed(2)} unit="%" />
              <MetricCard label="Est. SNR" value={quality.estimatedSNRDB} unit="dB" />
            </div>
          </Section>

          {/* Raw JSON export */}
          <div style={{ marginTop: 12 }}>
            <details style={{ fontSize: 11 }}>
              <summary style={{ cursor: 'pointer', color: 'rgba(255,255,255,0.35)', letterSpacing: '0.06em', userSelect: 'none' }}>
                RAW JSON TELEMETRY
              </summary>
              <pre style={{
                marginTop: 8,
                padding: 12,
                background: 'rgba(0,0,0,0.4)',
                borderRadius: 6,
                fontSize: 10,
                color: 'rgba(255,255,255,0.5)',
                overflowX: 'auto',
                lineHeight: 1.6,
                maxHeight: 200,
                overflowY: 'auto',
              }}>
                {JSON.stringify({ model, performance: perf, quality, vad: { ...vad, scores: '[...array]' } }, null, 2)}
              </pre>
            </details>
          </div>
        </div>
      )}
    </div>
  );
}
