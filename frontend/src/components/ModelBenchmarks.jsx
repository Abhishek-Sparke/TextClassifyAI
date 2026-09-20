import React, { useState } from 'react';
import { 
  Trophy, 
  Zap, 
  BarChart3, 
  Timer, 
  TrendingUp, 
  CheckCircle2, 
  Layers, 
  HelpCircle,
  Award
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Legend, 
  Cell 
} from 'recharts';
import { MODEL_PERFORMANCE, DATASET_STATS } from '../data/benchmarkData';

export default function ModelBenchmarks() {
  const [selectedMetric, setSelectedMetric] = useState('accuracy');

  // Format data for comparative bar chart
  const chartData = MODEL_PERFORMANCE.map(m => ({
    name: m.shortName,
    accuracy: m.accuracy,
    precision: m.precision,
    recall: m.recall,
    f1Score: m.f1Score,
    trainingTime: parseFloat(m.trainingTime),
    isBest: m.isBest
  }));

  const metricsInfo = {
    accuracy: { label: 'Accuracy (%)', min: 80, max: 95, color: '#6366f1' },
    precision: { label: 'Precision (%)', min: 80, max: 95, color: '#06b6d4' },
    recall: { label: 'Recall (%)', min: 80, max: 95, color: '#10b981' },
    f1Score: { label: 'F1-Score (%)', min: 80, max: 95, color: '#8b5cf6' },
  };

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Header Section */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/70 border border-indigo-200 dark:border-indigo-800 text-indigo-600 dark:text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <Award className="w-3.5 h-3.5" />
            Empirical Model Evaluation
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            Classifier Performance Benchmarking
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
            Rigorous evaluation across 721 held-out test documents using Stratified 80/20 train-test splits and 5,000 sublinear TF-IDF features.
          </p>
        </div>

        {/* Champion Badge */}
        <div className="flex items-center gap-3 p-3.5 rounded-2xl bg-gradient-to-r from-amber-500/10 via-amber-500/5 to-transparent border border-amber-500/30 text-amber-900 dark:text-amber-200">
          <div className="w-10 h-10 rounded-xl bg-amber-500 flex items-center justify-center text-white shadow-md shadow-amber-500/20">
            <Trophy className="w-5 h-5" />
          </div>
          <div>
            <div className="text-[11px] font-bold uppercase tracking-wider text-amber-600 dark:text-amber-400">
              Benchmark Champion
            </div>
            <div className="text-sm font-bold text-slate-900 dark:text-white">
              Linear SVM (89.74% Acc)
            </div>
          </div>
        </div>
      </div>

      {/* 4 Summary Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Top Test Accuracy</span>
            <TrendingUp className="w-4 h-4 text-emerald-500" />
          </div>
          <div className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
            89.74%
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
            Achieved by Support Vector Machine
          </p>
        </div>

        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Fastest Training</span>
            <Zap className="w-4 h-4 text-amber-500" />
          </div>
          <div className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
            0.004 s
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
            Multinomial Naive Bayes (Laplace)
          </p>
        </div>

        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Test Sample Size</span>
            <CheckCircle2 className="w-4 h-4 text-indigo-500" />
          </div>
          <div className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
            721 docs
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
            20% Stratified hold-out split
          </p>
        </div>

        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="flex items-center justify-between text-slate-500 dark:text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Feature Dimensionality</span>
            <Layers className="w-4 h-4 text-cyan-500" />
          </div>
          <div className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
            5,000
          </div>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1">
            Unigrams & Bigrams with sublinear TF
          </p>
        </div>
      </div>

      {/* Main Charts & Visualization Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left: Interactive Metric Comparison Chart (8 cols) */}
        <div className="lg:col-span-8 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 sm:p-6 shadow-sm space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider">
                Model Performance Comparison
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                Multi-metric benchmarking across all 4 classification algorithms
              </p>
            </div>

            {/* Metric Filter Tabs */}
            <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800 p-1 rounded-xl">
              {['accuracy', 'precision', 'recall', 'f1Score'].map((key) => (
                <button
                  key={key}
                  onClick={() => setSelectedMetric(key)}
                  className={`px-2.5 py-1 rounded-lg text-xs font-semibold capitalize transition-all ${
                    selectedMetric === key
                      ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-xs'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white'
                  }`}
                >
                  {key.replace('Score', '')}
                </button>
              ))}
            </div>
          </div>

          <div className="h-72 w-full pt-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.15} vertical={false} />
                <XAxis 
                  dataKey="name" 
                  tick={{ fontSize: 12, fill: '#888888' }} 
                  axisLine={{ stroke: '#888888', opacity: 0.2 }}
                />
                <YAxis 
                  domain={[80, 92]} 
                  tick={{ fontSize: 11, fill: '#888888' }} 
                  axisLine={{ stroke: '#888888', opacity: 0.2 }}
                  unit="%"
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'rgba(15, 23, 42, 0.9)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                    color: '#fff',
                    fontSize: '12px',
                    boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)'
                  }}
                  formatter={(val) => [`${val}%`, metricsInfo[selectedMetric].label]}
                />
                <Bar 
                  dataKey={selectedMetric} 
                  radius={[8, 8, 0, 0]}
                  barSize={48}
                >
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.isBest ? '#6366f1' : '#94a3b8'} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-800">
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-xs bg-indigo-600 inline-block" />
              <span>Champion (Best Overall)</span>
              <span className="w-3 h-3 rounded-xs bg-slate-400 inline-block ml-3" />
              <span>Competing Algorithms</span>
            </div>
            <span>Values expressed in percentage</span>
          </div>
        </div>

        {/* Right: Training Speed & Latency Comparison (4 cols) */}
        <div className="lg:col-span-4 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 sm:p-6 shadow-sm space-y-4 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-1.5">
                <Timer className="w-4 h-4 text-amber-500" />
                Training Runtime
              </h3>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Fitting time across 2,883 documents on Intel Core / Apple Silicon:
            </p>

            <div className="mt-4 space-y-3">
              {MODEL_PERFORMANCE.map((model) => (
                <div key={model.id} className="space-y-1">
                  <div className="flex items-center justify-between text-xs font-medium">
                    <span className="text-slate-700 dark:text-slate-300">{model.shortName}</span>
                    <span className="font-mono text-slate-500 dark:text-slate-400">{model.trainingTime}</span>
                  </div>
                  <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        model.id === 'naive_bayes' 
                          ? 'bg-amber-500' 
                          : model.id === 'svm' 
                          ? 'bg-indigo-600' 
                          : 'bg-slate-400 dark:bg-slate-600'
                      }`}
                      style={{ 
                        width: `${Math.max((parseFloat(model.trainingTime) / 0.38) * 100, 3)}%` 
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Theoretical Insight Box */}
          <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
            <span className="font-bold text-slate-900 dark:text-white block mb-1">
              Why SVM Wins in Text Categorization:
            </span>
            TF-IDF matrices are ultra-high dimensional (5,000 features) and sparse. Linear SVMs find the maximum geometric margin hyperplane without being susceptible to the curse of dimensionality.
          </div>
        </div>

      </div>

      {/* Comprehensive Metric Comparison Table */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm overflow-hidden">
        <div className="p-5 border-b border-slate-200 dark:border-slate-800">
          <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider">
            Detailed Benchmark Scorecard
          </h3>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Full breakdown of evaluation metrics, algorithmic paradigms, and characteristics
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 dark:bg-slate-800/50 text-slate-500 dark:text-slate-400 font-bold uppercase tracking-wider border-b border-slate-200 dark:border-slate-800">
              <tr>
                <th className="px-5 py-3.5">Model Algorithm</th>
                <th className="px-4 py-3.5">Paradigm</th>
                <th className="px-4 py-3.5 text-right">Accuracy</th>
                <th className="px-4 py-3.5 text-right">Precision</th>
                <th className="px-4 py-3.5 text-right">Recall</th>
                <th className="px-4 py-3.5 text-right">F1-Score</th>
                <th className="px-4 py-3.5 text-right">Train Speed</th>
                <th className="px-5 py-3.5">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-slate-800">
              {MODEL_PERFORMANCE.map((model) => (
                <tr 
                  key={model.id}
                  className={`hover:bg-slate-50/80 dark:hover:bg-slate-800/40 transition-colors ${
                    model.isBest ? 'bg-indigo-50/30 dark:bg-indigo-950/20 font-medium' : ''
                  }`}
                >
                  <td className="px-5 py-4 font-bold text-slate-900 dark:text-white">
                    <div className="flex items-center gap-2">
                      {model.isBest && <Trophy className="w-4 h-4 text-amber-500" />}
                      <span>{model.name}</span>
                    </div>
                  </td>
                  <td className="px-4 py-4 text-slate-500 dark:text-slate-400">
                    {model.type}
                  </td>
                  <td className="px-4 py-4 text-right font-mono font-bold text-slate-900 dark:text-white">
                    {model.accuracy.toFixed(2)}%
                  </td>
                  <td className="px-4 py-4 text-right font-mono text-slate-700 dark:text-slate-300">
                    {model.precision.toFixed(2)}%
                  </td>
                  <td className="px-4 py-4 text-right font-mono text-slate-700 dark:text-slate-300">
                    {model.recall.toFixed(2)}%
                  </td>
                  <td className="px-4 py-4 text-right font-mono text-slate-700 dark:text-slate-300">
                    {model.f1Score.toFixed(2)}%
                  </td>
                  <td className="px-4 py-4 text-right font-mono text-slate-500 dark:text-slate-400">
                    {model.trainingTime}
                  </td>
                  <td className="px-5 py-4">
                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-semibold ${
                      model.isBest
                        ? 'bg-amber-100 dark:bg-amber-950/80 text-amber-800 dark:text-amber-300 border border-amber-300 dark:border-amber-700'
                        : 'bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-400'
                    }`}>
                      {model.badge}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  );
}
