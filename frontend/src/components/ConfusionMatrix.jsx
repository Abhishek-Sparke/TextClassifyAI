import React, { useState } from 'react';
import { 
  Grid3X3, 
  Info, 
  HelpCircle, 
  CheckCircle2, 
  AlertTriangle, 
  Target, 
  PieChart 
} from 'lucide-react';
import { CONFUSION_MATRIX, CATEGORIES } from '../data/benchmarkData';

export default function ConfusionMatrix() {
  const [hoveredCell, setHoveredCell] = useState(null);
  const matrix = CONFUSION_MATRIX.matrix;
  const labels = CONFUSION_MATRIX.labels;

  // Calculate per-class metrics
  const classStats = labels.map((label, idx) => {
    const tp = matrix[idx][idx];
    const totalActual = matrix[idx].reduce((a, b) => a + b, 0);
    const totalPredicted = matrix.reduce((acc, row) => acc + row[idx], 0);
    const fn = totalActual - tp;
    const fp = totalPredicted - tp;
    
    const recall = (tp / totalActual) * 100;
    const precision = (tp / totalPredicted) * 100;
    const f1 = (2 * precision * recall) / (precision + recall);

    return {
      name: label,
      fullName: CATEGORIES[idx]?.name || label,
      icon: CATEGORIES[idx]?.icon || '📄',
      tp,
      totalActual,
      totalPredicted,
      fp,
      fn,
      recall: recall.toFixed(1),
      precision: precision.toFixed(1),
      f1: f1.toFixed(1),
    };
  });

  const totalTestDocs = matrix.flat().reduce((a, b) => a + b, 0);
  const totalCorrect = matrix.reduce((acc, row, i) => acc + row[i], 0);
  const overallAccuracy = ((totalCorrect / totalTestDocs) * 100).toFixed(2);

  // Helper function to get background shading
  const getCellBg = (val, isDiagonal) => {
    if (isDiagonal) {
      if (val > 120) return 'bg-indigo-600 text-white font-bold';
      if (val > 80) return 'bg-indigo-500 text-white font-bold';
      return 'bg-indigo-400 text-white font-bold';
    }
    if (val === 0) return 'bg-slate-50 dark:bg-slate-800/40 text-slate-300 dark:text-slate-600';
    if (val <= 4) return 'bg-amber-100 dark:bg-amber-950/40 text-amber-800 dark:text-amber-300 font-medium';
    if (val <= 10) return 'bg-amber-200 dark:bg-amber-900/60 text-amber-900 dark:text-amber-200 font-semibold';
    return 'bg-rose-200 dark:bg-rose-950/70 text-rose-900 dark:text-rose-200 font-bold';
  };

  const isLargeMatrix = labels.length > 8;

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/70 border border-indigo-200 dark:border-indigo-800 text-indigo-600 dark:text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
            <Grid3X3 className="w-3.5 h-3.5" />
            Error & Misclassification Analysis
          </div>
          <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
            {labels.length}x{labels.length} Confusion Matrix
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
            Detailed {labels.length}x{labels.length} matrix representation showing true ground truth labels vs. model predictions across all {totalTestDocs} evaluated test documents.
          </p>
        </div>

        <div className="flex items-center gap-4 bg-white dark:bg-slate-900 p-3 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-xs">
          <div>
            <div className="text-[10px] uppercase font-bold text-slate-400">Total Classified</div>
            <div className="text-xl font-bold text-slate-900 dark:text-white">{totalCorrect} / {totalTestDocs}</div>
          </div>
          <div className="h-8 w-px bg-slate-200 dark:bg-slate-800" />
          <div>
            <div className="text-[10px] uppercase font-bold text-slate-400">Test Accuracy</div>
            <div className="text-xl font-bold text-indigo-600 dark:text-indigo-400">{overallAccuracy}%</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left: Heatmap Matrix Card (7 cols) */}
        <div className="lg:col-span-7 bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 sm:p-6 shadow-sm space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider">
              Heatmap Matrix Grid
            </h3>
            <span className="text-[11px] text-slate-400">
              Columns = Predicted, Rows = Actual
            </span>
          </div>

          <div className="overflow-x-auto py-2">
            <div className={isLargeMatrix ? "min-w-[820px]" : "min-w-[420px]"}>
              
              {/* Predicted Label Header */}
              <div className="text-center text-xs font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider mb-2">
                Predicted Class →
              </div>

              <div 
                className={`text-center items-center ${isLargeMatrix ? 'gap-1 text-[11px]' : 'gap-2 text-xs'}`}
                style={{ display: 'grid', gridTemplateColumns: `repeat(${labels.length + 1}, minmax(0, 1fr))` }}
              >
                {/* Top left corner empty */}
                <div className="p-1 text-[10px] font-bold text-slate-400 uppercase">
                  Actual ↓
                </div>
                {labels.map((lbl, idx) => (
                  <div key={idx} className="p-1 font-bold text-slate-700 dark:text-slate-300 truncate" title={lbl}>
                    {lbl}
                  </div>
                ))}

                {/* Matrix Rows */}
                {matrix.map((row, rowIdx) => (
                  <React.Fragment key={rowIdx}>
                    <div className="p-1 text-right font-bold text-slate-700 dark:text-slate-300 truncate" title={labels[rowIdx]}>
                      {labels[rowIdx]}
                    </div>
                    {row.map((val, colIdx) => {
                      const isDiagonal = rowIdx === colIdx;
                      return (
                        <div
                          key={colIdx}
                          onMouseEnter={() => setHoveredCell({
                            actual: labels[rowIdx],
                            predicted: labels[colIdx],
                            count: val,
                            isCorrect: isDiagonal
                          })}
                          onMouseLeave={() => setHoveredCell(null)}
                          className={`${isLargeMatrix ? 'p-1.5 text-xs' : 'p-3.5 text-sm'} rounded-lg text-center transition-all duration-150 cursor-pointer shadow-xs ${getCellBg(val, isDiagonal)} hover:scale-110 hover:ring-2 hover:ring-indigo-400`}
                        >
                          {val}
                        </div>
                      );
                    })}
                  </React.Fragment>
                ))}
              </div>
            </div>
          </div>

          {/* Hover Inspector Tooltip Bar */}
          <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-700/60 text-xs min-h-[42px] flex items-center justify-between">
            {hoveredCell ? (
              <div className="flex items-center gap-2">
                <span className={hoveredCell.isCorrect ? 'text-emerald-600 dark:text-emerald-400 font-bold' : 'text-amber-600 dark:text-amber-400 font-bold'}>
                  {hoveredCell.count} documents
                </span>
                <span className="text-slate-600 dark:text-slate-300">
                  Actual: <strong>{hoveredCell.actual}</strong>, Predicted: <strong>{hoveredCell.predicted}</strong>
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-200 dark:bg-slate-700 font-semibold">
                  {hoveredCell.isCorrect ? 'Correct Classification (TP)' : 'Misclassification'}
                </span>
              </div>
            ) : (
              <span className="text-slate-400 text-[11px] italic">
                Hover over any cell in the heatmap to view exact misclassification attribution.
              </span>
            )}
          </div>

          {/* Color Legend */}
          <div className="flex flex-wrap items-center gap-4 text-xs text-slate-500 dark:text-slate-400 pt-2 border-t border-slate-100 dark:border-slate-800">
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-3.5 rounded-xs bg-indigo-600 inline-block" />
              <span>True Positive (&gt;100)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-3.5 rounded-xs bg-amber-200 dark:bg-amber-900 inline-block" />
              <span>Minor Error (1–10)</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-3.5 rounded-xs bg-rose-200 dark:bg-rose-950 inline-block" />
              <span>Higher Error (&gt;10)</span>
            </div>
          </div>
        </div>

        {/* Right: Per-Class Granular Performance (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4">
            <h3 className="text-sm font-bold text-slate-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
              <Target className="w-4 h-4 text-indigo-500" />
              Per-Class Metrics Breakdown ({classStats.length} Classes)
            </h3>

            <div className="space-y-3 max-h-[580px] overflow-y-auto pr-1">
              {classStats.map((stat, idx) => (
                <div 
                  key={idx}
                  className="p-3.5 rounded-xl border border-slate-100 dark:border-slate-800/80 bg-slate-50/50 dark:bg-slate-800/30 space-y-2"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2 font-bold text-xs text-slate-900 dark:text-white">
                      <span>{stat.icon}</span>
                      <span>{stat.fullName}</span>
                    </div>
                    <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                      {stat.tp} / {stat.totalActual} correct
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 text-center text-xs">
                    <div className="p-1.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Precision</div>
                      <div className="font-bold text-slate-800 dark:text-slate-200 mt-0.5">{stat.precision}%</div>
                    </div>
                    <div className="p-1.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">Recall</div>
                      <div className="font-bold text-indigo-600 dark:text-indigo-400 mt-0.5">{stat.recall}%</div>
                    </div>
                    <div className="p-1.5 rounded-lg bg-white dark:bg-slate-900 border border-slate-200/60 dark:border-slate-800">
                      <div className="text-[10px] text-slate-400 uppercase">F1-Score</div>
                      <div className="font-bold text-emerald-600 dark:text-emerald-400 mt-0.5">{stat.f1}%</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Qualitative Error Analysis Insight */}
          <div className="p-4 rounded-2xl bg-indigo-50/50 dark:bg-indigo-950/30 border border-indigo-200/70 dark:border-indigo-800/70 space-y-2 text-xs text-slate-600 dark:text-slate-300">
            <h4 className="font-bold text-slate-900 dark:text-white flex items-center gap-1.5">
              <AlertTriangle className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
              Empirical Error Observations
            </h4>
            <p className="leading-relaxed">
              <strong>Baseball (rec.sport.baseball)</strong> achieved the highest recall (95.8%) due to distinct specialized vocabulary (e.g., <em>pitcher</em>, <em>homerun</em>, <em>inning</em>).
            </p>
            <p className="leading-relaxed">
              <strong>Politics (talk.politics.misc)</strong> experienced the most cross-talk (119 TP out of 149), as political articles frequently debate NASA funding (Space) or computer privacy rights (Graphics/Tech).
            </p>
          </div>

        </div>

      </div>

    </div>
  );
}
