import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Send, 
  RotateCcw, 
  CheckCircle2, 
  Clock, 
  Cpu, 
  ChevronRight, 
  Hash, 
  Layers, 
  AlertCircle,
  Copy,
  BookOpen,
  Filter
} from 'lucide-react';
import { SAMPLE_DOCUMENTS, MODEL_PERFORMANCE, CATEGORIES } from '../data/benchmarkData';
import { classifyDocument } from '../services/api';

export default function Playground({ apiStatus }) {
  const [inputText, setInputText] = useState(SAMPLE_DOCUMENTS[0].fullText);
  const [selectedModel, setSelectedModel] = useState('svm');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showPreprocessed, setShowPreprocessed] = useState(false);
  const [copied, setCopied] = useState(false);

  // Compute word & char counts
  const charCount = inputText.length;
  const wordCount = inputText.trim() ? inputText.trim().split(/\s+/).length : 0;

  // Run initial classification on mount
  useEffect(() => {
    handleClassify();
  }, []);

  const handleClassify = async (overrideText = null, overrideModel = null) => {
    const textToClassify = overrideText !== null ? overrideText : inputText;
    const modelToUse = overrideModel !== null ? overrideModel : selectedModel;

    if (!textToClassify.trim()) return;

    setIsLoading(true);
    try {
      const prediction = await classifyDocument(textToClassify, modelToUse);
      setResult(prediction);
    } catch (err) {
      console.error("Classification error:", err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSampleClick = (sample) => {
    setInputText(sample.fullText);
    handleClassify(sample.fullText, selectedModel);
  };

  const handleClear = () => {
    setInputText('');
    setResult(null);
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(inputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Preprocessed text preview generator for client preview
  const getClientCleanTokens = (text) => {
    if (!text) return '';
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .replace(/\d+/g, ' ')
      .split(/\s+/)
      .filter(w => w.length > 2)
      .slice(0, 45)
      .join(' ') + (text.length > 250 ? ' ...' : '');
  };

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Hero Welcome Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-indigo-900 via-slate-900 to-slate-900 p-6 md:p-8 text-white shadow-xl">
        <div className="absolute top-0 right-0 -mt-10 -mr-10 w-80 h-80 bg-indigo-500/20 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/20 border border-indigo-400/30 text-indigo-300 text-xs font-semibold uppercase tracking-wider mb-3">
            <Sparkles className="w-3.5 h-3.5" />
            Live Machine Learning Inference
          </div>
          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold tracking-tight">
            Classify Raw Text Documents in Real-Time
          </h1>
          <p className="mt-2 text-sm sm:text-base text-slate-300 leading-relaxed">
            Test any document against trained classifiers (<span className="text-indigo-300 font-medium">Linear SVM</span>, <span className="text-indigo-300 font-medium">Logistic Regression</span>, <span className="text-indigo-300 font-medium">Naive Bayes</span>, or <span className="text-indigo-300 font-medium">Random Forest</span>). Inspect confidence distributions and top TF-IDF keywords.
          </p>
        </div>
      </div>

      {/* Quick Sample Selector Bar */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
            <BookOpen className="w-3.5 h-3.5 text-indigo-500" />
            Load Realistic Benchmark Samples:
          </span>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {SAMPLE_DOCUMENTS.map((sample, idx) => (
            <button
              key={idx}
              onClick={() => handleSampleClick(sample)}
              className="text-left p-3.5 rounded-xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 hover:border-indigo-400 dark:hover:border-indigo-600 hover:shadow-md transition-all group flex flex-col justify-between"
            >
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-lg">{sample.icon}</span>
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 group-hover:bg-indigo-50 dark:group-hover:bg-indigo-950/60 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 transition-colors">
                    {sample.category}
                  </span>
                </div>
                <h4 className="text-xs font-semibold text-slate-800 dark:text-slate-200 group-hover:text-indigo-600 dark:group-hover:text-indigo-400 line-clamp-1">
                  {sample.title}
                </h4>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 line-clamp-2 leading-relaxed">
                  {sample.preview}
                </p>
              </div>
              <div className="mt-2.5 flex items-center text-[10px] font-medium text-indigo-600 dark:text-indigo-400">
                <span>Load document</span>
                <ChevronRight className="w-3 h-3 ml-0.5 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>
          ))}
        </div>
      </div>

      {/* Main Two-Column Playground Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Input and Model Selection (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          
          {/* Model Selector Card */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
            <div className="flex items-center justify-between mb-3">
              <label className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <Cpu className="w-4 h-4 text-indigo-600 dark:text-indigo-400" />
                Select ML Classifier Algorithm:
              </label>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
              {MODEL_PERFORMANCE.map((model) => {
                const isSelected = selectedModel === model.id;
                return (
                  <button
                    key={model.id}
                    onClick={() => {
                      setSelectedModel(model.id);
                      handleClassify(inputText, model.id);
                    }}
                    className={`relative p-2.5 rounded-xl text-left border transition-all ${
                      isSelected
                        ? 'border-indigo-600 dark:border-indigo-500 bg-indigo-50/70 dark:bg-indigo-950/50 shadow-sm ring-1 ring-indigo-500'
                        : 'border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 bg-slate-50/50 dark:bg-slate-800/40'
                    }`}
                  >
                    {model.isBest && (
                      <span className="absolute -top-2 right-2 text-[9px] font-bold px-1.5 py-0.2 rounded-md bg-indigo-600 text-white shadow-xs">
                        BEST
                      </span>
                    )}
                    <div className="font-semibold text-xs text-slate-900 dark:text-white truncate">
                      {model.shortName}
                    </div>
                    <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                      Acc: <span className="font-semibold text-slate-700 dark:text-slate-200">{model.accuracy}%</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Text Input Card */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                  Raw Document Text
                </span>
                <span className="text-xs text-slate-400 dark:text-slate-500">
                  (Paste or type content)
                </span>
              </div>
              <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
                <button
                  onClick={handleCopyText}
                  className="hover:text-slate-800 dark:hover:text-slate-200 transition-colors flex items-center gap-1"
                  title="Copy text"
                >
                  <Copy className="w-3.5 h-3.5" />
                  {copied ? 'Copied' : 'Copy'}
                </button>
                <button
                  onClick={handleClear}
                  className="hover:text-red-600 dark:hover:text-red-400 transition-colors flex items-center gap-1"
                  title="Clear text"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Clear
                </button>
              </div>
            </div>

            <div className="relative">
              <textarea
                rows={9}
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                    handleClassify();
                  }
                }}
                placeholder="Paste news articles, technical document abstracts, science reports, or political debates here..."
                className="w-full p-4 rounded-xl text-sm leading-relaxed bg-slate-50 dark:bg-slate-950/70 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all font-sans resize-y"
              />
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
              <div className="flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400 font-mono">
                <span>{wordCount.toLocaleString()} words</span>
                <span>•</span>
                <span>{charCount.toLocaleString()} characters</span>
              </div>

              <div className="flex items-center gap-3">
                <button
                  onClick={() => handleClassify()}
                  disabled={isLoading || !inputText.trim()}
                  className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-sm font-semibold text-white shadow-md transition-all duration-200 ${
                    isLoading || !inputText.trim()
                      ? 'bg-indigo-400/60 dark:bg-indigo-600/40 cursor-not-allowed'
                      : 'bg-indigo-600 hover:bg-indigo-500 active:scale-[0.98] shadow-indigo-600/25 hover:shadow-indigo-600/40'
                  }`}
                >
                  {isLoading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      <span>Classifying...</span>
                    </>
                  ) : (
                    <>
                      <Send className="w-4 h-4" />
                      <span>Run Classification</span>
                    </>
                  )}
                </button>
              </div>
            </div>
            
            <p className="text-[11px] text-slate-400 dark:text-slate-500 italic text-right">
              Tip: Press <kbd className="px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-800 border border-slate-300 dark:border-slate-700 text-[10px] font-mono">Ctrl + Enter</kbd> to classify quickly
            </p>
          </div>

        </div>

        {/* Right Column: Prediction Results & Insights (5 cols) */}
        <div className="lg:col-span-5 space-y-4">
          
          {result ? (
            <div className="space-y-4 animate-fade-in">
              
              {/* Hero Predicted Category Card */}
              <div className="p-6 rounded-2xl bg-gradient-to-br from-indigo-600 via-indigo-700 to-slate-900 text-white shadow-xl relative overflow-hidden">
                <div className="absolute top-0 right-0 p-4 opacity-15 text-8xl pointer-events-none select-none">
                  {result.icon}
                </div>

                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-bold uppercase tracking-wider text-indigo-200 flex items-center gap-1.5">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                      Predicted Category
                    </span>
                    <span className="text-xs font-medium px-2.5 py-0.5 rounded-full bg-white/15 backdrop-blur-xs text-white">
                      {result.modelUsed.split('(')[0].trim()}
                    </span>
                  </div>

                  <div className="flex items-baseline gap-3">
                    <span className="text-3xl">{result.icon}</span>
                    <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
                      {result.category}
                    </h2>
                  </div>

                  <div className="mt-4 pt-4 border-t border-white/15 flex items-center justify-between">
                    <div>
                      <div className="text-xs text-indigo-200">Confidence Score</div>
                      <div className="text-2xl font-black tracking-tight text-white mt-0.5">
                        {result.confidence}%
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-xs text-indigo-200">Execution Time</div>
                      <div className="text-sm font-mono font-semibold text-emerald-300 mt-1 flex items-center justify-end gap-1">
                        <Clock className="w-3.5 h-3.5" />
                        {result.processingTime}s
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Class Probabilities Distribution */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-4">
                <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center justify-between">
                  <span>Class Probability Distribution</span>
                  <span className="text-[11px] text-slate-400 font-normal">Softmax Calibrated</span>
                </h3>

                <div className="space-y-3">
                  {result.probabilities.map((item, idx) => {
                    const isTop = idx === 0;
                    return (
                      <div key={item.categoryId} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2 font-medium text-slate-700 dark:text-slate-200">
                            <span>{item.icon}</span>
                            <span className={isTop ? 'font-bold text-indigo-600 dark:text-indigo-400' : ''}>
                              {item.name}
                            </span>
                          </div>
                          <span className={`font-mono text-xs ${isTop ? 'font-bold text-indigo-600 dark:text-indigo-400' : 'text-slate-500 dark:text-slate-400'}`}>
                            {item.probability}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 dark:bg-slate-800 h-2.5 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all duration-500 ease-out ${
                              isTop 
                                ? 'bg-gradient-to-r from-indigo-500 to-indigo-600' 
                                : 'bg-slate-300 dark:bg-slate-700'
                            }`}
                            style={{ width: `${item.probability}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Top Influential TF-IDF Keywords */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Hash className="w-4 h-4 text-indigo-500" />
                    Top TF-IDF Influential Keywords
                  </h3>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Feature Weights
                  </span>
                </div>

                {result.topKeywords && result.topKeywords.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {result.topKeywords.map((kw, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200/80 dark:border-slate-700/80"
                      >
                        <span className="text-indigo-600 dark:text-indigo-400 font-semibold">#{kw.term}</span>
                        <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">({kw.weight})</span>
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 dark:text-slate-400 italic">
                    General vocabulary terms detected. Add more domain-specific terms to see detailed TF-IDF feature attribution.
                  </p>
                )}
              </div>

              {/* Preprocessed Tokens Inspection */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm">
                <button
                  onClick={() => setShowPreprocessed(!showPreprocessed)}
                  className="w-full flex items-center justify-between text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider hover:text-indigo-600 transition-colors"
                >
                  <span className="flex items-center gap-1.5">
                    <Filter className="w-4 h-4 text-indigo-500" />
                    NLP Preprocessing Inspection
                  </span>
                  <span className="text-[11px] font-medium text-indigo-600 dark:text-indigo-400">
                    {showPreprocessed ? 'Hide Tokens' : 'Inspect Tokens'}
                  </span>
                </button>

                {showPreprocessed && (
                  <div className="mt-3 pt-3 border-t border-slate-100 dark:border-slate-800 space-y-2 animate-fade-in">
                    <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-relaxed">
                      Cleaned representation after lowercasing, stripping punctuation, filtering stop words, and applying WordNet lemmatization:
                    </p>
                    <div className="p-3 rounded-xl bg-slate-50 dark:bg-slate-950 font-mono text-xs text-slate-700 dark:text-slate-300 leading-relaxed border border-slate-200 dark:border-slate-800 max-h-32 overflow-y-auto">
                      {result.cleanText || getClientCleanTokens(inputText)}
                    </div>
                  </div>
                )}
              </div>

            </div>
          ) : (
            <div className="h-full min-h-[350px] flex flex-col items-center justify-center p-8 text-center bg-white dark:bg-slate-900 rounded-2xl border border-dashed border-slate-200 dark:border-slate-800 text-slate-400">
              <Sparkles className="w-10 h-10 mb-3 text-slate-300 dark:text-slate-700" />
              <h4 className="text-sm font-semibold text-slate-700 dark:text-slate-300">
                Awaiting Document Input
              </h4>
              <p className="text-xs text-slate-500 mt-1 max-w-xs">
                Enter text or choose a benchmark sample from above, then click &quot;Run Classification&quot; to inspect ML predictions.
              </p>
            </div>
          )}

        </div>

      </div>

    </div>
  );
}
