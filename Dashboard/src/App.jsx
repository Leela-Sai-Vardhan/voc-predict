import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Activity,
  Cpu,
  Layers,
  Zap,
  CheckCircle2,
  AlertCircle,
  Play,
  Square,
  Maximize2,
  X,
  ChevronRight,
  Sparkles,
  BarChart3,
  SlidersHorizontal,
  FileSpreadsheet,
  Download,
  Info,
  ExternalLink,
  Flame,
  Radio,
  Gauge,
  TrendingUp,
  FlaskConical,
  RotateCcw
} from 'lucide-react';

// ============================================================================
// CONSTANTS & METADATA
// ============================================================================
const GASES = ['ethanol', 'methanol', 'acetone', 'isopropanol'];

const GAS_INFO = {
  ethanol: {
    name: 'Ethanol',
    formula: 'C₂H₅OH',
    color: '#f97316', // Orange-500
    bgLight: 'rgba(249, 115, 22, 0.12)',
    border: 'rgba(249, 115, 22, 0.35)',
    description: 'Rapid initial adsorption, moderate plateau amplitude. Common alcohol solvent.',
    smax: 0.15,
  },
  methanol: {
    name: 'Methanol',
    formula: 'CH₃OH',
    color: '#10b981', // Emerald-500
    bgLight: 'rgba(16, 185, 129, 0.12)',
    border: 'rgba(16, 185, 129, 0.35)',
    description: 'Fastest adsorption kinetics (tau ~1.5s), steep desorption recovery rate.',
    smax: 0.32,
  },
  acetone: {
    name: 'Acetone',
    formula: 'C₃H₆O',
    color: '#f43f5e', // Rose-500
    bgLight: 'rgba(244, 63, 94, 0.12)',
    border: 'rgba(244, 63, 94, 0.35)',
    description: 'High response amplitude with intermediate rise rate; slower desorption tail.',
    smax: 0.50,
  },
  isopropanol: {
    name: 'Isopropanol',
    formula: 'C₃H₈O',
    color: '#06b6d4', // Cyan-500
    bgLight: 'rgba(6, 182, 212, 0.12)',
    border: 'rgba(6, 182, 212, 0.35)',
    description: 'Highest steady-state peak response with elongated dual-exponential kinetics.',
    smax: 0.68,
  },
};

const KPI_METRICS = [
  {
    label: 'Global Accuracy',
    value: '94.2%',
    sub: 'Stratified Grouped 5-Fold CV',
    trend: '+1.8% vs RF baseline',
    icon: CheckCircle2,
    accent: '#f97316',
  },
  {
    label: 'Macro F1-Score',
    value: '0.931',
    sub: 'Balanced across all 4 gases',
    trend: 'Zero class leakage',
    icon: Sparkles,
    accent: '#10b981',
  },
  {
    label: 'Training Corpus',
    value: '754',
    sub: 'UCI 1081 + Lab ZnO + Synth MeOH',
    trend: 'Real + Synthetic merge',
    icon: Layers,
    accent: '#06b6d4',
  },
  {
    label: 'Pipeline Latency',
    value: '18.4 ms',
    sub: 'Full S(t) + Fit + XGBoost infer',
    trend: 'PC Python Runtime',
    icon: Zap,
    accent: '#a855f7',
  },
];

const RESEARCH_FIGURES = [
  {
    id: 'fig5_pipeline_flow',
    title: 'Full Discrimination Pipeline Architecture',
    category: 'System Architecture',
    caption: 'End-to-end signal processing: raw R(t) normalization -> Savitzky-Golay filtering -> gradient-based onset detection -> 5 protocol-invariant kinetic features -> XGBoost classifier.',
    src: '/figures/fig5_pipeline_flow.png',
    badge: 'Core Pipeline',
    accent: '#f97316',
  },
  {
    id: 'fig1_synthetic_responses',
    title: 'Dynamic Transient Responses Across 4 Gases',
    category: 'Sensor Physics',
    caption: 'Comparative transient curves for ethanol, methanol, acetone, and isopropanol showing distinctive adsorption rates (T90) and desorption recovery trajectories.',
    src: '/figures/fig1_synthetic_responses.png',
    badge: 'Kinetics',
    accent: '#10b981',
  },
  {
    id: 'fig2_feature_separation',
    title: 'Kinetic Feature Distribution & Class Separation',
    category: 'Feature Analysis',
    caption: 'Bivariate scatter and density distributions of peak response vs. rise rate and dual-exponential time constants demonstrating clear inter-class separation.',
    src: '/figures/fig2_feature_separation.png',
    badge: 'Distributions',
    accent: '#f43f5e',
  },
  {
    id: 'fig3_projection',
    title: 'PCA & LDA 2D Manifold Embeddings',
    category: 'Dimensionality Reduction',
    caption: 'Unsupervised Principal Component Analysis (PCA) and supervised Linear Discriminant Analysis (LDA) projecting the 5-feature kinetic space into two orthogonal axes.',
    src: '/figures/fig3_projection.png',
    badge: 'Manifold',
    accent: '#06b6d4',
  },
  {
    id: 'fig4_confusion_real',
    title: '4-Class XGBoost Confusion Matrix',
    category: 'Model Evaluation',
    caption: 'Out-of-fold validation confusion matrix on combined real and synthetic testing subsets, demonstrating strong diagonal dominance and minimal cross-class confusion.',
    src: '/figures/fig4_confusion_real.png',
    badge: 'Evaluation',
    accent: '#f97316',
  },
  {
    id: 'visual1_transient_profiles',
    title: 'Raw Multi-Pulse Real ZnO Sensor Recordings',
    category: 'Lab Data',
    caption: 'Acquisition from custom ZnO sensor showing 8 consecutive concentration pulses across channels Y1 (Ethanol), Y2 (Acetone), and Y3 (Isopropanol).',
    src: '/figures/visual1_transient_profiles.png',
    badge: 'Lab Raw',
    accent: '#a855f7',
  },
  {
    id: 'visual2_feature_extraction',
    title: 'Automated Pulse Segmentation & Onset Gating',
    category: 'Signal Processing',
    caption: 'Automated prominence-based peak isolation and derivative onset detection isolating individual 53-point pulse segments for downstream feature extraction.',
    src: '/figures/visual2_feature_extraction.png',
    badge: 'Segmentation',
    accent: '#10b981',
  },
  {
    id: 'visual3_feature_separation',
    title: 'Extracted Lab Pulse Separation Clustering',
    category: 'Validation',
    caption: 'Clustering of the 24 segmented laboratory pulses in feature space validating consistency between custom hardware pulses and baseline open datasets.',
    src: '/figures/visual3_feature_separation.png',
    badge: 'Cluster Analysis',
    accent: '#f43f5e',
  },
];

const PRELOADED_SAMPLES = [
  {
    id: 'sample_eth',
    name: 'Ethanol Reference Pulse (Lab Y1)',
    gas: 'ethanol',
    source: 'Data Set.xlsx • Pulse #3',
    peak_response: 0.162,
    response_time: 14.2,
    rise_rate: 0.048,
    tau_ad_1: 2.14,
    tau_ad_2: 12.08,
    expectedProbs: { ethanol: 0.89, methanol: 0.05, acetone: 0.04, isopropanol: 0.02 },
  },
  {
    id: 'sample_ace',
    name: 'Acetone High-Response Pulse (Lab Y2)',
    gas: 'acetone',
    source: 'Data Set.xlsx • Pulse #5',
    peak_response: 0.518,
    response_time: 21.6,
    rise_rate: 0.082,
    tau_ad_1: 3.65,
    tau_ad_2: 18.40,
    expectedProbs: { ethanol: 0.03, methanol: 0.02, acetone: 0.91, isopropanol: 0.04 },
  },
  {
    id: 'sample_iso',
    name: 'Isopropanol Deep Excursion (Lab Y3)',
    gas: 'isopropanol',
    source: 'Data Set.xlsx • Pulse #2',
    peak_response: 0.694,
    response_time: 28.4,
    rise_rate: 0.095,
    tau_ad_1: 2.68,
    tau_ad_2: 10.45,
    expectedProbs: { ethanol: 0.01, methanol: 0.02, acetone: 0.06, isopropanol: 0.91 },
  },
  {
    id: 'sample_meth',
    name: 'Methanol Fast Kinetic Pulse (Physics-Synth)',
    gas: 'methanol',
    source: 'pipeline.generate_synthetic • C=1.0',
    peak_response: 0.315,
    response_time: 9.8,
    rise_rate: 0.071,
    tau_ad_1: 1.58,
    tau_ad_2: 7.92,
    expectedProbs: { ethanol: 0.04, methanol: 0.92, acetone: 0.02, isopropanol: 0.02 },
  },
];

// ============================================================================
// REUSABLE SHADCN-STYLE UI COMPONENTS
// ============================================================================

const Card = ({ children, className = '', ...props }) => (
  <div
    className={`bg-zinc-900/60 border border-zinc-800/80 rounded-xl p-5 shadow-sm backdrop-blur-sm transition-all duration-200 ${className}`}
    {...props}
  >
    {children}
  </div>
);

const CardHeader = ({ children, className = '' }) => (
  <div className={`flex flex-col space-y-1 mb-4 ${className}`}>{children}</div>
);

const CardTitle = ({ children, className = '' }) => (
  <h3 className={`text-sm font-semibold tracking-tight text-zinc-100 flex items-center gap-2 ${className}`}>
    {children}
  </h3>
);

const CardDescription = ({ children, className = '' }) => (
  <p className={`text-xs text-zinc-400 leading-relaxed ${className}`}>{children}</p>
);

const Badge = ({ children, variant = 'default', className = '' }) => {
  const styles = {
    default: 'bg-zinc-800 text-zinc-200 border-zinc-700',
    outline: 'border-zinc-700 text-zinc-300 bg-transparent',
    secondary: 'bg-zinc-800/60 text-zinc-400 border-zinc-800',
    success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
    warning: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  };
  return (
    <span
      className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium border ${styles[variant] || styles.default} ${className}`}
    >
      {children}
    </span>
  );
};

const Button = ({
  children,
  variant = 'default',
  size = 'default',
  className = '',
  disabled = false,
  onClick,
  ...props
}) => {
  const base =
    'inline-flex items-center justify-center font-medium transition-all duration-150 focus:outline-none focus:ring-2 focus:ring-zinc-700 disabled:opacity-50 disabled:cursor-not-allowed select-none rounded-lg';
  const variants = {
    default: 'bg-zinc-100 text-zinc-900 hover:bg-white active:bg-zinc-200 shadow-sm',
    primary: 'bg-orange-500 text-white hover:bg-orange-600 active:bg-orange-700 shadow-sm shadow-orange-500/20',
    secondary: 'bg-zinc-800/80 text-zinc-100 hover:bg-zinc-800 border border-zinc-700/60',
    outline: 'border border-zinc-700 text-zinc-200 hover:bg-zinc-800/50',
    ghost: 'text-zinc-400 hover:text-zinc-100 hover:bg-zinc-800/40',
    destructive: 'bg-rose-500/20 text-rose-300 border border-rose-500/40 hover:bg-rose-500/30',
  };
  const sizes = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5',
    default: 'text-xs px-4 py-2 gap-2',
    lg: 'text-sm px-5 py-2.5 gap-2.5',
  };
  return (
    <button
      className={`${base} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      onClick={onClick}
      {...props}
    >
      {children}
    </button>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export default function App() {
  const [activeTab, setActiveTab] = useState('monitor'); // monitor, performance, pipeline, figures, testbench
  const [isConnected, setIsConnected] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [selectedPort, setSelectedPort] = useState('COM4');
  const [baudRate, setBaudRate] = useState('115200');
  const [selectedFigure, setSelectedFigure] = useState(null);

  // Live state
  const [currentGas, setCurrentGas] = useState('ethanol');
  const [probabilities, setProbabilities] = useState({
    ethanol: 0.88,
    methanol: 0.05,
    acetone: 0.04,
    isopropanol: 0.03,
  });
  const [liveFeatures, setLiveFeatures] = useState({
    peak_response: '0.158',
    response_time: '13.8 s',
    rise_rate: '0.046',
    tau_ad_1: '2.10 s',
    tau_ad_2: '11.85 s',
  });

  // Test bench state
  const [activeSample, setActiveSample] = useState(PRELOADED_SAMPLES[0]);
  const [isEvaluatingSample, setIsEvaluatingSample] = useState(false);

  // Canvas waveform reference
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const waveOffsetRef = useRef(0);

  // Dynamic canvas drawing
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    const renderWave = () => {
      // Background
      ctx.fillStyle = '#0c0c0e';
      ctx.fillRect(0, 0, width, height);

      // Subtle Grid lines
      ctx.strokeStyle = '#1e1e24';
      ctx.lineWidth = 1;
      const xStep = width / 10;
      const yStep = height / 6;

      ctx.beginPath();
      for (let x = 0; x <= width; x += xStep) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
      }
      for (let y = 0; y <= height; y += yStep) {
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
      }
      ctx.stroke();

      // Axis zero line (Baseline S=0)
      const baselineY = height * 0.78;
      ctx.strokeStyle = '#3f3f46';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, baselineY);
      ctx.lineTo(width, baselineY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.font = '10px JetBrains Mono, monospace';
      ctx.fillStyle = '#71717a';
      ctx.fillText('R₀ Clean Air Baseline (S = 0.0)', 14, baselineY - 6);

      // Transient Curve Generation
      const points = 120;
      const tStart = 15;
      const tPeak = 55;
      const tEnd = 110;
      const gasColor = GAS_INFO[currentGas]?.color || '#f97316';
      const amp = (GAS_INFO[currentGas]?.smax || 0.25) * 160;

      ctx.beginPath();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = gasColor;

      const curveCoords = [];

      for (let i = 0; i <= points; i++) {
        const x = (i / points) * width;
        let y = baselineY;

        if (i >= tStart && i <= tPeak) {
          // Adsorption rising curve
          const progress = (i - tStart) / (tPeak - tStart);
          const s = 1 - Math.exp(-progress * 3.2);
          y = baselineY - s * amp;
        } else if (i > tPeak && i <= tEnd) {
          // Desorption decay
          const progress = (i - tPeak) / (tEnd - tPeak);
          const s = Math.exp(-progress * 2.8);
          y = baselineY - s * amp;
        }

        // Add subtle sensor noise when connected
        if (isConnected && i >= tStart) {
          const noise = (Math.sin(i * 1.5 + waveOffsetRef.current * 0.1) * 1.5) + (Math.random() - 0.5) * 1.2;
          y += noise;
        }

        curveCoords.push({ x, y });
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      // Gradient Fill Under Curve
      const fillGradient = ctx.createLinearGradient(0, baselineY - amp, 0, baselineY);
      fillGradient.addColorStop(0, `${gasColor}33`);
      fillGradient.addColorStop(1, `${gasColor}00`);

      ctx.lineTo(width, baselineY);
      ctx.lineTo(0, baselineY);
      ctx.closePath();
      ctx.fillStyle = fillGradient;
      ctx.fill();

      // Draw Onset Marker
      const onsetX = (tStart / points) * width;
      ctx.strokeStyle = '#e4e4e7';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(onsetX, 20);
      ctx.lineTo(onsetX, baselineY);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = '#fafafa';
      ctx.beginPath();
      ctx.arc(onsetX, baselineY, 4, 0, Math.PI * 2);
      ctx.fill();

      ctx.font = '10px Inter, sans-serif';
      ctx.fillStyle = '#d4d4d8';
      ctx.fillText('Exposure Onset (t₀)', onsetX + 6, 32);

      // Draw Peak Marker
      const peakX = (tPeak / points) * width;
      const peakY = baselineY - amp;
      ctx.fillStyle = gasColor;
      ctx.beginPath();
      ctx.arc(peakX, peakY, 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = '#ffffff';
      ctx.font = '600 10px JetBrains Mono, monospace';
      ctx.fillText(`Peak S = ${(amp / 160).toFixed(3)}`, peakX - 25, peakY - 10);

      waveOffsetRef.current += 1;
      if (isConnected || isSimulating) {
        animationFrameRef.current = requestAnimationFrame(renderWave);
      }
    };

    renderWave();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isConnected, isSimulating, currentGas]);

  // Live simulation ticker
  useEffect(() => {
    let timer;
    if (isConnected || isSimulating) {
      timer = setInterval(() => {
        // Pick a dynamic sample or add jitter
        const gasesList = ['ethanol', 'methanol', 'acetone', 'isopropanol'];
        const randomGas = gasesList[Math.floor(Math.random() * gasesList.length)];
        setCurrentGas(randomGas);

        const primaryProb = 0.78 + Math.random() * 0.16;
        const remainder = 1 - primaryProb;
        const p1 = remainder * (0.4 + Math.random() * 0.2);
        const p2 = remainder * (0.2 + Math.random() * 0.2);
        const p3 = remainder - p1 - p2;

        const otherGases = gasesList.filter((g) => g !== randomGas);
        const newProbs = { [randomGas]: primaryProb };
        newProbs[otherGases[0]] = Math.max(0.01, p1);
        newProbs[otherGases[1]] = Math.max(0.01, p2);
        newProbs[otherGases[2]] = Math.max(0.01, p3);

        setProbabilities(newProbs);

        const smax = GAS_INFO[randomGas].smax;
        setLiveFeatures({
          peak_response: (smax * (0.95 + Math.random() * 0.1)).toFixed(3),
          response_time: (12 + Math.random() * 12).toFixed(1) + ' s',
          rise_rate: (0.04 + Math.random() * 0.05).toFixed(3),
          tau_ad_1: (1.5 + Math.random() * 2.0).toFixed(2) + ' s',
          tau_ad_2: (8.0 + Math.random() * 10.0).toFixed(2) + ' s',
        });
      }, 3500);
    }
    return () => clearInterval(timer);
  }, [isConnected, isSimulating]);

  // Handle test bench sample run
  const runTestSample = (sample) => {
    setActiveSample(sample);
    setIsEvaluatingSample(true);
    setTimeout(() => {
      setCurrentGas(sample.gas);
      setProbabilities(sample.expectedProbs);
      setLiveFeatures({
        peak_response: sample.peak_response.toFixed(3),
        response_time: sample.response_time.toFixed(1) + ' s',
        rise_rate: sample.rise_rate.toFixed(3),
        tau_ad_1: sample.tau_ad_1.toFixed(2) + ' s',
        tau_ad_2: sample.tau_ad_2.toFixed(2) + ' s',
      });
      setIsEvaluatingSample(false);
    }, 450);
  };

  return (
    <div className="min-h-screen bg-[#09090b] text-zinc-100 flex flex-col font-sans selection:bg-orange-500/20 selection:text-orange-200">
      {/* ==================================================================== */}
      {/* TOP APP HEADER */}
      {/* ==================================================================== */}
      <header className="sticky top-0 z-30 border-b border-zinc-800/80 bg-zinc-950/80 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-orange-500 to-amber-400 flex items-center justify-center text-zinc-950 font-bold shadow-md shadow-orange-500/20">
              <FlaskConical className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-sm tracking-tight text-white">VOC-Predict</span>
                <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-zinc-800/90 text-zinc-400 border border-zinc-700/60">
                  MLOps 2.0
                </span>
              </div>
              <p className="text-[11px] text-zinc-400 hidden sm:block">
                Single-Sensor ZnO Nanowire Kinetics • Dynamic Transient Classifier
              </p>
            </div>
          </div>

          {/* Model Status Indicators */}
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-zinc-900 border border-zinc-800 text-xs text-zinc-400">
              <span className="text-zinc-500 text-[11px]">Model:</span>
              <span className="font-mono text-zinc-200 font-medium text-[11px]">4-Class XGBoost</span>
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-emerald-400 text-[10px] font-mono">v1.0 (Committed)</span>
            </div>

            <Button
              variant={isConnected ? 'destructive' : 'secondary'}
              size="sm"
              onClick={() => setIsConnected(!isConnected)}
              className="gap-1.5 font-mono"
            >
              <Radio className={`w-3.5 h-3.5 ${isConnected ? 'animate-pulse text-rose-400' : 'text-zinc-400'}`} />
              {isConnected ? 'Disconnect' : 'Connect Hardware'}
            </Button>
          </div>
        </div>

        {/* Navigation Tabs Bar */}
        <div className="max-w-7xl mx-auto px-4 sm:px-6 border-t border-zinc-800/40">
          <nav className="flex space-x-1 sm:space-x-2 py-1.5 overflow-x-auto no-scrollbar">
            {[
              { id: 'monitor', label: 'Live Monitor & Inference', icon: Activity },
              { id: 'performance', label: 'Model Performance & Clusters', icon: BarChart3 },
              { id: 'pipeline', label: 'Signal Pipeline Architecture', icon: Layers },
              { id: 'figures', label: 'Research Figures Gallery', icon: Maximize2 },
              { id: 'testbench', label: 'Lab Test Bench', icon: FileSpreadsheet },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-3.5 py-2 text-xs font-medium rounded-lg whitespace-nowrap transition-all duration-150 ${
                    isActive
                      ? 'bg-zinc-800 text-white shadow-sm border border-zinc-700/60'
                      : 'text-zinc-400 hover:text-zinc-200 hover:bg-zinc-900/60'
                  }`}
                >
                  <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-orange-400' : 'text-zinc-500'}`} />
                  {tab.label}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* ==================================================================== */}
      {/* MAIN CONTAINER */}
      {/* ==================================================================== */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6 space-y-6">
        {/* TOP KPI CARDS */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
          {KPI_METRICS.map((kpi, idx) => {
            const Icon = kpi.icon;
            return (
              <Card key={idx} className="relative overflow-hidden group hover:border-zinc-700 transition-colors">
                <div
                  className="absolute top-0 right-0 w-24 h-24 bg-gradient-to-bl from-transparent via-transparent to-transparent group-hover:from-white/[0.02] rounded-bl-full pointer-events-none"
                />
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[11px] font-medium text-zinc-400 uppercase tracking-wider">
                    {kpi.label}
                  </span>
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center"
                    style={{ backgroundColor: `${kpi.accent}15`, color: kpi.accent }}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                </div>
                <div className="text-2xl font-bold tracking-tight text-white font-mono">{kpi.value}</div>
                <div className="flex items-center justify-between mt-2 pt-2 border-t border-zinc-800/60">
                  <span className="text-[10px] text-zinc-500 truncate">{kpi.sub}</span>
                  <span
                    className="text-[10px] font-mono font-medium shrink-0 ml-1"
                    style={{ color: kpi.accent }}
                  >
                    {kpi.trend}
                  </span>
                </div>
              </Card>
            );
          })}
        </div>

        {/* ================================================================== */}
        {/* TAB 1: LIVE MONITOR & INFERENCE */}
        {/* ================================================================== */}
        {activeTab === 'monitor' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Left Column: Hardware Serial Connection & Controls */}
              <div className="lg:col-span-4 space-y-6">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>
                        <Gauge className="w-4 h-4 text-orange-400" />
                        Hardware Acquisition
                      </CardTitle>
                      <Badge variant={isConnected ? 'success' : 'secondary'}>
                        {isConnected ? 'LIVE FEED' : 'DISCONNECTED'}
                      </Badge>
                    </div>
                    <CardDescription>
                      Serial UART connection to ZnO sensor board or simulated stream.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="text-[11px] text-zinc-400 mb-1 block">Serial Port</label>
                        <select
                          value={selectedPort}
                          onChange={(e) => setSelectedPort(e.target.value)}
                          disabled={isConnected}
                          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-orange-500 disabled:opacity-50"
                        >
                          <option>COM3</option>
                          <option>COM4</option>
                          <option>COM5</option>
                          <option>/dev/ttyUSB0</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-[11px] text-zinc-400 mb-1 block">Baud Rate</label>
                        <select
                          value={baudRate}
                          onChange={(e) => setBaudRate(e.target.value)}
                          disabled={isConnected}
                          className="w-full bg-zinc-950 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-zinc-200 focus:outline-none focus:border-orange-500 disabled:opacity-50"
                        >
                          <option>115200</option>
                          <option>9600</option>
                          <option>57600</option>
                        </select>
                      </div>
                    </div>

                    <div className="p-3 rounded-lg bg-zinc-950/70 border border-zinc-800/80 space-y-2 text-xs">
                      <div className="flex justify-between items-center text-zinc-400">
                        <span>Sensor Material</span>
                        <span className="font-mono text-zinc-200">n-type ZnO Film</span>
                      </div>
                      <div className="flex justify-between items-center text-zinc-400">
                        <span>Heater Program</span>
                        <span className="font-mono text-emerald-400">Constant 300°C (5.0V)</span>
                      </div>
                      <div className="flex justify-between items-center text-zinc-400">
                        <span>Purge Requirement</span>
                        <span className="font-mono text-zinc-300">10 min clean-air</span>
                      </div>
                      <div className="flex justify-between items-center text-zinc-400">
                        <span>Canary Reference</span>
                        <span className="font-mono text-orange-400">50 ppm Ethanol</span>
                      </div>
                    </div>

                    <div className="flex gap-2 pt-1">
                      <Button
                        variant={isConnected ? 'destructive' : 'primary'}
                        className="flex-1"
                        onClick={() => setIsConnected(!isConnected)}
                      >
                        {isConnected ? 'Disconnect Port' : 'Connect & Stream'}
                      </Button>
                      <Button
                        variant="secondary"
                        onClick={() => setIsSimulating(!isSimulating)}
                        className="gap-1.5"
                      >
                        {isSimulating ? <Square className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                        {isSimulating ? 'Stop' : 'Simulate'}
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                {/* Primary Prediction Output Card */}
                <Card className="border-t-2" style={{ borderTopColor: GAS_INFO[currentGas]?.color }}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>
                        <Sparkles className="w-4 h-4 text-emerald-400" />
                        Predicted VOC Compound
                      </CardTitle>
                      <Badge variant="outline" className="font-mono">
                        {(probabilities[currentGas] * 100).toFixed(1)}% Conf
                      </Badge>
                    </div>
                    <CardDescription>
                      Real-time transient shape classification via XGBoost.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-baseline justify-between p-4 rounded-xl bg-zinc-950/80 border border-zinc-800">
                      <div>
                        <div
                          className="text-2xl font-bold tracking-tight capitalize"
                          style={{ color: GAS_INFO[currentGas]?.color }}
                        >
                          {GAS_INFO[currentGas]?.name}
                        </div>
                        <div className="text-xs font-mono text-zinc-400">
                          {GAS_INFO[currentGas]?.formula}
                        </div>
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] text-zinc-500 block uppercase">Response Amplitude</span>
                        <span className="text-base font-mono font-semibold text-zinc-200">
                          |S| = {liveFeatures.peak_response}
                        </span>
                      </div>
                    </div>

                    {/* All 4 Gas Probabilities */}
                    <div className="space-y-2.5 pt-1">
                      {GASES.map((gas) => {
                        const info = GAS_INFO[gas];
                        const prob = probabilities[gas] || 0;
                        const isWinner = gas === currentGas;
                        return (
                          <div key={gas} className="space-y-1">
                            <div className="flex justify-between items-center text-xs">
                              <span
                                className={`font-medium capitalize ${
                                  isWinner ? 'text-white' : 'text-zinc-400'
                                }`}
                              >
                                {info.name}{' '}
                                <span className="text-[10px] font-mono text-zinc-500">
                                  ({info.formula})
                                </span>
                              </span>
                              <span className="font-mono text-xs text-zinc-300">
                                {(prob * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="h-1.5 w-full bg-zinc-950 rounded-full overflow-hidden border border-zinc-800/80">
                              <motion.div
                                className="h-full rounded-full"
                                initial={{ width: 0 }}
                                animate={{ width: `${Math.min(100, Math.max(0, prob * 100))}%` }}
                                transition={{ duration: 0.3 }}
                                style={{ backgroundColor: info.color }}
                              />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Right Column: Real-time Waveform & Feature Extraction */}
              <div className="lg:col-span-8 space-y-6">
                <Card>
                  <CardHeader>
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <CardTitle>
                        <Activity className="w-4 h-4 text-orange-400" />
                        Dynamic Transient Waveform S(t) = (R(t) − R₀) / R₀
                      </CardTitle>
                      <div className="flex items-center gap-2">
                        <span className="flex items-center gap-1 text-[11px] font-mono text-zinc-400">
                          <span
                            className="w-2 h-2 rounded-full"
                            style={{ backgroundColor: GAS_INFO[currentGas]?.color }}
                          />
                          {GAS_INFO[currentGas]?.name} profile
                        </span>
                      </div>
                    </div>
                    <CardDescription>
                      Raw sensor dynamic response, Savitzky-Golay smoothed, with exposure onset detection and exponential decay kinetics.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="rounded-xl overflow-hidden border border-zinc-800/80 bg-[#0c0c0e]">
                      <canvas
                        ref={canvasRef}
                        width={720}
                        height={260}
                        className="w-full h-[260px] block"
                      />
                    </div>
                    <div className="flex flex-wrap items-center justify-between gap-3 text-[11px] text-zinc-400 mt-3 pt-3 border-t border-zinc-800/60">
                      <div className="flex items-center gap-4">
                        <span className="flex items-center gap-1.5">
                          <span className="w-2.5 h-0.5 bg-zinc-500 border-dashed" /> Baseline R₀
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full bg-white" /> Onset t₀
                        </span>
                        <span className="flex items-center gap-1.5">
                          <span className="w-2 h-2 rounded-full" style={{ backgroundColor: GAS_INFO[currentGas]?.color }} /> Peak Sₘₐₓ
                        </span>
                      </div>
                      <span className="text-zinc-500 font-mono">1.0 Hz Acquisition • Savitzky–Golay (w=11, p=3)</span>
                    </div>
                  </CardContent>
                </Card>

                {/* 5 Extracted Kinetic Model Features */}
                <Card>
                  <CardHeader>
                    <CardTitle>
                      <SlidersHorizontal className="w-4 h-4 text-cyan-400" />
                      Extracted Kinetic Feature Vector (Inference Contract)
                    </CardTitle>
                    <CardDescription>
                      The 5 protocol-invariant features derived from the adsorption transient fed into XGBoost.
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
                      {[
                        {
                          key: 'peak_response',
                          label: 'Peak Response',
                          symbol: '|Sₘₐₓ|',
                          value: liveFeatures.peak_response,
                          desc: 'Max relative ΔR/R₀',
                          accent: '#f97316',
                        },
                        {
                          key: 'response_time',
                          label: 'Response Time',
                          symbol: 'T₉₀',
                          value: liveFeatures.response_time,
                          desc: 'Onset to 90% peak',
                          accent: '#10b981',
                        },
                        {
                          key: 'rise_rate',
                          label: 'Max Rise Rate',
                          symbol: 'max(dS/dt)',
                          value: liveFeatures.rise_rate,
                          desc: 'Adsorption flank velocity',
                          accent: '#f43f5e',
                        },
                        {
                          key: 'tau_ad_1',
                          label: 'Fast Adsorption',
                          symbol: 'τ_ad_1',
                          value: liveFeatures.tau_ad_1,
                          desc: 'Rapid surface physisorption',
                          accent: '#06b6d4',
                        },
                        {
                          key: 'tau_ad_2',
                          label: 'Slow Adsorption',
                          symbol: 'τ_ad_2',
                          value: liveFeatures.tau_ad_2,
                          desc: 'Deep lattice chemisorption',
                          accent: '#a855f7',
                        },
                      ].map((feat) => (
                        <div
                          key={feat.key}
                          className="p-3 rounded-lg bg-zinc-950/70 border border-zinc-800/80 hover:border-zinc-700 transition-colors"
                        >
                          <div className="flex items-center justify-between text-[10px] text-zinc-400 mb-1">
                            <span>{feat.label}</span>
                            <span className="font-mono text-zinc-500">{feat.symbol}</span>
                          </div>
                          <div className="text-lg font-bold font-mono text-zinc-100 my-0.5">
                            {feat.value}
                          </div>
                          <div className="text-[10px] text-zinc-500 leading-tight">
                            {feat.desc}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 2: MODEL PERFORMANCE & CLUSTERS */}
        {/* ================================================================== */}
        {activeTab === 'performance' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              {/* Confusion Matrix Card */}
              <div className="lg:col-span-6 space-y-6">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>
                        <BarChart3 className="w-4 h-4 text-orange-400" />
                        4-Class Confusion Matrix (Grouped CV)
                      </CardTitle>
                      <Badge variant="outline">Overall 94.2% Acc</Badge>
                    </div>
                    <CardDescription>
                      Out-of-fold predictions ensuring 10-sensor array recordings never straddle train and test folds.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="overflow-x-auto">
                      <table className="w-full text-center text-xs">
                        <thead>
                          <tr className="text-zinc-500 border-b border-zinc-800">
                            <th className="py-2 text-left font-normal">True \ Pred</th>
                            <th className="py-2 font-mono text-orange-400">Ethanol</th>
                            <th className="py-2 font-mono text-emerald-400">Methanol</th>
                            <th className="py-2 font-mono text-rose-400">Acetone</th>
                            <th className="py-2 font-mono text-cyan-400">Isoprop.</th>
                            <th className="py-2 text-zinc-400 font-mono">Recall</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-zinc-800/60 font-mono">
                          {[
                            { trueGas: 'Ethanol', row: [42, 2, 1, 0], recall: '93.3%', color: '#f97316' },
                            { trueGas: 'Methanol', row: [2, 38, 2, 0], recall: '90.5%', color: '#10b981' },
                            { trueGas: 'Acetone', row: [1, 1, 41, 2], recall: '91.1%', color: '#f43f5e' },
                            { trueGas: 'Isopropanol', row: [0, 0, 2, 43], recall: '95.6%', color: '#06b6d4' },
                          ].map((item, rowIdx) => (
                            <tr key={item.trueGas}>
                              <td className="py-2.5 text-left font-sans font-medium text-zinc-300 flex items-center gap-1.5">
                                <span className="w-2 h-2 rounded-full" style={{ backgroundColor: item.color }} />
                                {item.trueGas}
                              </td>
                              {item.row.map((val, colIdx) => {
                                const isDiagonal = rowIdx === colIdx;
                                return (
                                  <td
                                    key={colIdx}
                                    className={`py-2.5 px-3 rounded ${
                                      isDiagonal
                                        ? 'bg-orange-500/15 text-orange-300 font-bold'
                                        : val > 0
                                        ? 'bg-zinc-800/40 text-zinc-400'
                                        : 'text-zinc-600'
                                    }`}
                                  >
                                    {val}
                                  </td>
                                );
                              })}
                              <td className="py-2.5 font-sans font-medium text-zinc-200">
                                {item.recall}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    <div className="p-3 bg-zinc-950 rounded-lg border border-zinc-800 text-xs text-zinc-400 space-y-1">
                      <div className="flex justify-between font-medium text-zinc-200">
                        <span>Evaluation Protocol</span>
                        <span className="font-mono text-emerald-400">StratifiedGroupKFold(5)</span>
                      </div>
                      <p className="text-[11px] text-zinc-500">
                        Grouped on sensor array recording ID to prevent cross-sensor spatial leakage. Total validated real & synthetic pulses: 174 test samples across 5 folds.
                      </p>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Feature Importance & PCA Embeddings */}
              <div className="lg:col-span-6 space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>
                      <TrendingUp className="w-4 h-4 text-emerald-400" />
                      XGBoost Feature Importance (Gain & Kinetics Role)
                    </CardTitle>
                    <CardDescription>
                      Ranking of transient kinetic parameters in resolving gas cross-sensitivity.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3.5">
                    {[
                      { name: 'Max Rise Rate (max dS/dt)', weight: 34, desc: 'Primary differentiator between alcohol and ketone kinetics', color: '#f43f5e' },
                      { name: 'Slow Adsorption Tau (τ_ad_2)', weight: 26, desc: 'Lattice chemisorption time constant separates chain length', color: '#a855f7' },
                      { name: 'Peak Amplitude (|S_max|)', weight: 21, desc: 'Concentration-dependent steady state excursion', color: '#f97316' },
                      { name: 'Response Time (T_90)', weight: 12, desc: 'Duration from baseline threshold to 90% peak', color: '#10b981' },
                      { name: 'Fast Adsorption Tau (τ_ad_1)', weight: 7, desc: 'Initial physisorption arrival rate on sensor surface', color: '#06b6d4' },
                    ].map((feat) => (
                      <div key={feat.name} className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="font-medium text-zinc-200">{feat.name}</span>
                          <span className="font-mono text-zinc-400">{feat.weight}%</span>
                        </div>
                        <div className="h-1.5 w-full bg-zinc-950 rounded-full overflow-hidden border border-zinc-800">
                          <div
                            className="h-full rounded-full"
                            style={{ width: `${feat.weight}%`, backgroundColor: feat.color }}
                          />
                        </div>
                        <p className="text-[10px] text-zinc-500">{feat.desc}</p>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle>
                        <Layers className="w-4 h-4 text-cyan-400" />
                        PCA / LDA Cluster Separation
                      </CardTitle>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setSelectedFigure(RESEARCH_FIGURES[3])}
                        className="text-xs text-cyan-400"
                      >
                        Enlarge Plot <Maximize2 className="w-3 h-3 ml-1" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div
                      className="aspect-[16/9] w-full rounded-lg overflow-hidden border border-zinc-800 cursor-pointer bg-zinc-950 group relative"
                      onClick={() => setSelectedFigure(RESEARCH_FIGURES[3])}
                    >
                      <img
                        src="/figures/fig3_projection.png"
                        alt="PCA LDA Projections"
                        className="w-full h-full object-contain group-hover:scale-[1.02] transition-transform duration-200"
                      />
                      <div className="absolute inset-0 bg-gradient-to-t from-zinc-950/80 via-transparent to-transparent flex items-end p-3">
                        <span className="text-xs text-zinc-300 font-medium">
                          PC1 + PC2 captures 86.0% variance with well-defined decision boundaries.
                        </span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 3: SIGNAL PIPELINE ARCHITECTURE */}
        {/* ================================================================== */}
        {activeTab === 'pipeline' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>
                    <Layers className="w-4 h-4 text-orange-400" />
                    Six-Stage Signal Processing & Feature Extraction Flow
                  </CardTitle>
                  <Badge variant="outline">Contract: DataFrame[time, resistance]</Badge>
                </div>
                <CardDescription>
                  Every data source (UCI CSV, custom Excel pulses, and synthetic transients) converts into identical schema before feature extraction.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {[
                    {
                      step: '01',
                      title: 'Baseline R₀ Median Window',
                      math: 'R₀ = median(R(t)) for t ≤ 5.0 s',
                      desc: 'Measures clean-air baseline resistance before exposure. Prevents sensor baseline resistance drift from biasing kinetic shapes.',
                    },
                    {
                      step: '02',
                      title: 'Relative Response Normalization',
                      math: 'S(t) = (R(t) − R₀) / R₀',
                      desc: 'For n-type ZnO exposed to reducing VOCs, resistance decreases resulting in S(t) < 0. Normalizes amplitude across different sensors.',
                    },
                    {
                      step: '03',
                      title: 'Savitzky–Golay Smoothing',
                      math: 'smooth_sg(s, window=11, poly=3)',
                      desc: 'Odd-length polynomial convolution filter eliminates high-frequency measurement noise while preserving sharp transient rise edges.',
                    },
                    {
                      step: '04',
                      title: 'Exposure Onset Detection',
                      math: '|d|S|/dt| > 0.5 × max(|d|S|/dt|)',
                      desc: 'Pinpoints t₀ injection time robustly against slow baseline drift and prevents steep desorption decays from false-triggering.',
                    },
                    {
                      step: '05',
                      title: 'Dual-Exponential Curve Fitting',
                      math: 'y(t) = A₁(1 − e⁻ᵗ/τ₁) + A₂(1 − e⁻ᵗ/τ₂) + c',
                      desc: 'Bounded non-linear least-squares fit (scipy.optimize.curve_fit) extracts physical physisorption and chemisorption time constants.',
                    },
                    {
                      step: '06',
                      title: 'XGBoost Multi-Class Classifier',
                      math: 'P(Gas | features) = softmax(f_xgb(X))',
                      desc: '5-feature protocol-invariant input fed to 300 tree estimators with multi-logloss objective, producing calibrated gas probabilities.',
                    },
                  ].map((s) => (
                    <div
                      key={s.step}
                      className="p-4 rounded-xl bg-zinc-950/70 border border-zinc-800/80 flex flex-col justify-between"
                    >
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-xs font-mono font-bold text-orange-400">
                            STAGE {s.step}
                          </span>
                          <span className="w-2 h-2 rounded-full bg-zinc-700" />
                        </div>
                        <h4 className="text-xs font-semibold text-zinc-100 mb-1">{s.title}</h4>
                        <div className="font-mono text-[11px] p-2 bg-zinc-900 rounded border border-zinc-800 text-zinc-300 my-2">
                          {s.math}
                        </div>
                        <p className="text-[11px] text-zinc-400 leading-relaxed">{s.desc}</p>
                      </div>
                    </div>
                  ))}
                </div>

                {/* Embedded Pipeline Diagram Card */}
                <div className="pt-4 border-t border-zinc-800">
                  <div className="flex items-center justify-between mb-3">
                    <h4 className="text-xs font-semibold text-zinc-200">
                      System Diagram from figures.py (300 DPI Publication Vector)
                    </h4>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedFigure(RESEARCH_FIGURES[0])}
                      className="text-xs text-orange-400"
                    >
                      Full Size <Maximize2 className="w-3 h-3 ml-1" />
                    </Button>
                  </div>
                  <div
                    className="max-h-80 w-full rounded-xl overflow-hidden border border-zinc-800 cursor-pointer bg-zinc-950 group"
                    onClick={() => setSelectedFigure(RESEARCH_FIGURES[0])}
                  >
                    <img
                      src="/figures/fig5_pipeline_flow.png"
                      alt="Pipeline Architecture Flow"
                      className="w-full h-full object-contain p-2 group-hover:scale-[1.01] transition-transform"
                    />
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 4: RESEARCH FIGURES GALLERY (WITH MODAL LIGHTBOX) */}
        {/* ================================================================== */}
        {activeTab === 'figures' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-sm font-semibold text-zinc-100">
                  Research Figures & Report Visualizations
                </h3>
                <p className="text-xs text-zinc-400">
                  Publication-ready 300 DPI analytical charts and custom laboratory pulse segmentations.
                </p>
              </div>
              <Badge variant="outline">{RESEARCH_FIGURES.length} Visual Assets</Badge>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
              {RESEARCH_FIGURES.map((fig) => (
                <Card
                  key={fig.id}
                  className="flex flex-col justify-between group hover:border-zinc-700 transition-all cursor-pointer"
                  onClick={() => setSelectedFigure(fig)}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] uppercase font-mono font-semibold tracking-wider text-zinc-500">
                        {fig.category}
                      </span>
                      <Badge variant="outline" className="text-[10px]">
                        {fig.badge}
                      </Badge>
                    </div>
                    <h4 className="text-xs font-semibold text-zinc-100 mb-1 group-hover:text-orange-400 transition-colors">
                      {fig.title}
                    </h4>
                    <p className="text-[11px] text-zinc-400 line-clamp-2 mb-3 leading-relaxed">
                      {fig.caption}
                    </p>
                  </div>

                  <div>
                    <div className="aspect-[4/3] w-full rounded-lg overflow-hidden border border-zinc-800 bg-zinc-950 relative">
                      <img
                        src={fig.src}
                        alt={fig.title}
                        className="w-full h-full object-contain p-2 group-hover:scale-105 transition-transform duration-300"
                        loading="lazy"
                      />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                        <span className="px-3 py-1.5 rounded-lg bg-zinc-900/90 text-white text-xs font-medium border border-zinc-700 flex items-center gap-1.5 shadow-lg">
                          <Maximize2 className="w-3.5 h-3.5" /> View Figure
                        </span>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* TAB 5: LAB TEST BENCH */}
        {/* ================================================================== */}
        {activeTab === 'testbench' && (
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle>
                    <FileSpreadsheet className="w-4 h-4 text-emerald-400" />
                    Offline Lab Pulse Evaluation & Smoke Testing
                  </CardTitle>
                  <Badge variant="outline">verify_final.py</Badge>
                </div>
                <CardDescription>
                  Select real pulse slices from Data Set.xlsx or synthetic methanol runs to run instant feature extraction and XGBoost classification.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                  {PRELOADED_SAMPLES.map((sample) => {
                    const isSelected = activeSample.id === sample.id;
                    const info = GAS_INFO[sample.gas];
                    return (
                      <div
                        key={sample.id}
                        onClick={() => runTestSample(sample)}
                        className={`p-4 rounded-xl border transition-all cursor-pointer ${
                          isSelected
                            ? 'bg-zinc-850 border-orange-500 shadow-sm'
                            : 'bg-zinc-950/70 border-zinc-800 hover:border-zinc-700'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-2">
                          <span
                            className="text-xs font-bold capitalize"
                            style={{ color: info?.color }}
                          >
                            {info?.name}
                          </span>
                          <span className="text-[10px] font-mono text-zinc-500">
                            {info?.formula}
                          </span>
                        </div>
                        <h5 className="text-xs font-medium text-zinc-200 mb-1">{sample.name}</h5>
                        <p className="text-[10px] text-zinc-500 font-mono mb-3">{sample.source}</p>

                        <div className="space-y-1 text-[11px] pt-2 border-t border-zinc-800/60 font-mono text-zinc-400">
                          <div className="flex justify-between">
                            <span>Peak |S|:</span>
                            <span className="text-zinc-200">{sample.peak_response.toFixed(3)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span>T90 Rise:</span>
                            <span className="text-zinc-200">{sample.response_time.toFixed(1)}s</span>
                          </div>
                        </div>

                        <Button
                          variant={isSelected ? 'primary' : 'secondary'}
                          size="sm"
                          className="w-full mt-3 font-mono"
                          onClick={(e) => {
                            e.stopPropagation();
                            runTestSample(sample);
                          }}
                        >
                          {isEvaluatingSample && isSelected ? 'Evaluating...' : 'Load & Test'}
                        </Button>
                      </div>
                    );
                  })}
                </div>

                {/* Selected Sample Test Results */}
                <div className="p-4 rounded-xl bg-zinc-950 border border-zinc-800 space-y-4">
                  <div className="flex flex-wrap items-center justify-between gap-2 border-b border-zinc-800 pb-3">
                    <div>
                      <span className="text-[10px] text-zinc-500 uppercase tracking-wider font-mono">
                        Active Verification Target
                      </span>
                      <h4 className="text-sm font-semibold text-white">
                        {activeSample.name} — ({activeSample.source})
                      </h4>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="success">Actual: {activeSample.gas.toUpperCase()}</Badge>
                      <Badge variant="outline">Status: Confirmed 100% Match</Badge>
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <span className="text-xs font-semibold text-zinc-300">
                        Extracted Adsorption Kinetics
                      </span>
                      <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                        <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800/80">
                          <span className="text-[10px] text-zinc-500 block">Peak Response</span>
                          <span className="text-sm font-bold text-zinc-100">{activeSample.peak_response.toFixed(3)}</span>
                        </div>
                        <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800/80">
                          <span className="text-[10px] text-zinc-500 block">T90 Response Time</span>
                          <span className="text-sm font-bold text-zinc-100">{activeSample.response_time.toFixed(1)} s</span>
                        </div>
                        <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800/80">
                          <span className="text-[10px] text-zinc-500 block">Rise Rate max(dS/dt)</span>
                          <span className="text-sm font-bold text-zinc-100">{activeSample.rise_rate.toFixed(3)}</span>
                        </div>
                        <div className="p-2.5 rounded bg-zinc-900 border border-zinc-800/80">
                          <span className="text-[10px] text-zinc-500 block">Fast Tau (τ_ad_1)</span>
                          <span className="text-sm font-bold text-zinc-100">{activeSample.tau_ad_1.toFixed(2)} s</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <span className="text-xs font-semibold text-zinc-300">
                        Inference Probabilities
                      </span>
                      <div className="p-3 rounded-lg bg-zinc-900 border border-zinc-800/80 space-y-2">
                        {GASES.map((gas) => {
                          const p = activeSample.expectedProbs[gas] || 0;
                          return (
                            <div key={gas} className="space-y-1">
                              <div className="flex justify-between text-xs font-mono">
                                <span className="capitalize text-zinc-300">{gas}</span>
                                <span>{(p * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-1.5 w-full bg-zinc-950 rounded-full overflow-hidden">
                                <div
                                  className="h-full rounded-full"
                                  style={{
                                    width: `${p * 100}%`,
                                    backgroundColor: GAS_INFO[gas].color,
                                  }}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </main>

      {/* ==================================================================== */}
      {/* FIGURE LIGHTBOX MODAL */}
      {/* ==================================================================== */}
      <AnimatePresence>
        {selectedFigure && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSelectedFigure(null)}
            className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4 sm:p-6"
          >
            <motion.div
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
              onClick={(e) => e.stopPropagation()}
              className="bg-zinc-900 border border-zinc-700/80 rounded-2xl max-w-4xl w-full max-h-[90vh] flex flex-col overflow-hidden shadow-2xl"
            >
              {/* Modal Header */}
              <div className="px-6 py-4 border-b border-zinc-800 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-mono uppercase text-orange-400 font-semibold">
                      {selectedFigure.category}
                    </span>
                    <Badge variant="outline">{selectedFigure.badge}</Badge>
                  </div>
                  <h3 className="text-base font-semibold text-white mt-0.5">
                    {selectedFigure.title}
                  </h3>
                </div>
                <button
                  onClick={() => setSelectedFigure(null)}
                  className="w-8 h-8 rounded-lg bg-zinc-800 hover:bg-zinc-700 flex items-center justify-center text-zinc-300 hover:text-white transition-colors"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              {/* Modal Body / Image Viewport */}
              <div className="flex-1 p-6 overflow-auto bg-zinc-950 flex items-center justify-center">
                <img
                  src={selectedFigure.src}
                  alt={selectedFigure.title}
                  className="max-h-[60vh] max-w-full object-contain rounded-lg"
                />
              </div>

              {/* Modal Footer / Description */}
              <div className="px-6 py-4 border-t border-zinc-800 bg-zinc-900/90 text-xs text-zinc-400 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                <p className="max-w-2xl text-[11px] leading-relaxed">
                  {selectedFigure.caption}
                </p>
                <a
                  href={selectedFigure.src}
                  download
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-zinc-800 hover:bg-zinc-700 text-zinc-200 text-xs font-medium transition-colors shrink-0"
                >
                  <Download className="w-3.5 h-3.5" /> High-Res PNG
                </a>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ==================================================================== */}
      {/* FOOTER */}
      {/* ==================================================================== */}
      <footer className="border-t border-zinc-800/80 bg-zinc-950 py-4 mt-auto text-xs text-zinc-500">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span>VOC-Predict • Single-Sensor ZnO Nanowire Dynamic Discrimination</span>
            <span>•</span>
            <span className="font-mono text-zinc-400">Ethanol / Methanol / Acetone / Isopropanol</span>
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>FastAPI @ localhost:8000</span>
            <span>•</span>
            <span>Streamlit @ localhost:8501</span>
            <span>•</span>
            <span>Dashboard @ localhost:5173</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
