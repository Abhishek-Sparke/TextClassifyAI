import React from 'react';
import { FileText, Github, Heart, Layers, Cpu, Code2 } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="mt-20 border-t border-slate-200 dark:border-slate-800 bg-white/50 dark:bg-slate-900/50 backdrop-blur-md py-10 transition-colors">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-indigo-600 flex items-center justify-center text-white shadow-xs">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <div className="font-bold text-sm text-slate-900 dark:text-white">
                Classifying Text Documents Using Machine Learning
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                End-to-End NLP Pipeline & Classifier Benchmarking System
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-2 text-xs text-slate-500 dark:text-slate-400">
            <span className="px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 font-mono text-[11px]">
              Scikit-Learn 1.3+
            </span>
            <span className="px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 font-mono text-[11px]">
              TF-IDF (5k Features)
            </span>
            <span className="px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 font-mono text-[11px]">
              Linear SVM
            </span>
            <span className="px-2.5 py-1 rounded-full bg-slate-100 dark:bg-slate-800 font-mono text-[11px]">
              React 18 + Vite
            </span>
          </div>

        </div>

        <div className="mt-6 pt-6 border-t border-slate-100 dark:border-slate-800/80 text-center text-xs text-slate-400 dark:text-slate-500">
          Machine Learning College Term Project • 20 Newsgroups Multi-Class Document Categorization
        </div>
      </div>
    </footer>
  );
}
