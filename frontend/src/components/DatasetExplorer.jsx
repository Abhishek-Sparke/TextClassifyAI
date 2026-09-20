import React from 'react';
import { 
  Database, 
  FileText, 
  PieChart, 
  Hash, 
  BookOpen, 
  Layers, 
  Tag, 
  ExternalLink 
} from 'lucide-react';
import { CATEGORIES, DATASET_STATS } from '../data/benchmarkData';

export default function DatasetExplorer() {
  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/70 border border-indigo-200 dark:border-indigo-800 text-indigo-600 dark:text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <Database className="w-3.5 h-3.5" />
          Corpus & Taxonomy
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          20 Newsgroups Dataset & Class Distribution
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
          Exploration of the multi-class document corpus, topic domain representations, class distribution statistics, and vocabulary characteristics.
        </p>
      </div>

      {/* Overview Stat Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
            Total Corpus Size
          </div>
          <div className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
            {DATASET_STATS.totalDocuments.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Cleaned text documents
          </div>
        </div>

        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
            Training Partition (80%)
          </div>
          <div className="text-2xl sm:text-3xl font-black text-indigo-600 dark:text-indigo-400">
            {DATASET_STATS.trainingSamples.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Stratified train samples
          </div>
        </div>

        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
            Test Partition (20%)
          </div>
          <div className="text-2xl sm:text-3xl font-black text-cyan-600 dark:text-cyan-400">
            {DATASET_STATS.testingSamples.toLocaleString()}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Held-out benchmark samples
          </div>
        </div>

        <div className="p-4 sm:p-5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-xs">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400 mb-1">
            Avg Word Count
          </div>
          <div className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white">
            {DATASET_STATS.averageWordCount}
          </div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mt-1">
            Median: {DATASET_STATS.medianWordCount} words/doc
          </div>
        </div>
      </div>

      {/* 4 Category Detailed Taxonomy Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {CATEGORIES.map((cat) => (
          <div 
            key={cat.id}
            className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm space-y-4 hover:border-indigo-300 dark:hover:border-indigo-700 transition-all"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <span className="text-3xl p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800 border border-slate-200/60 dark:border-slate-700/60">
                  {cat.icon}
                </span>
                <div>
                  <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                    {cat.name}
                  </h3>
                  <div className="font-mono text-xs text-slate-400 mt-0.5">
                    {cat.id}
                  </div>
                </div>
              </div>

              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/70 text-indigo-600 dark:text-indigo-400 border border-indigo-200/60 dark:border-indigo-800/60">
                {cat.badge}
              </span>
            </div>

            <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
              {cat.description}
            </p>

            {/* Distribution Bar */}
            <div className="space-y-1.5 pt-2">
              <div className="flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
                <span>Class Volume: <strong>{cat.count} docs</strong></span>
                <span className="font-mono">{cat.percentage}% of corpus</span>
              </div>
              <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-indigo-600 rounded-full"
                  style={{ width: `${cat.percentage * 3.5}%` }}
                />
              </div>
            </div>

            {/* Top Distinguishing Keywords */}
            <div className="pt-2 border-t border-slate-100 dark:border-slate-800">
              <div className="text-[11px] font-bold text-slate-500 dark:text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                <Tag className="w-3.5 h-3.5 text-indigo-500" />
                Dominant Characteristic Vocabulary:
              </div>
              <div className="flex flex-wrap gap-1.5">
                {cat.topKeywords.map((kw, idx) => (
                  <span
                    key={idx}
                    className="px-2 py-0.5 rounded-md text-xs font-mono bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200/60 dark:border-slate-700/60"
                  >
                    {kw}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>

    </div>
  );
}
