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
  Filter,
  UploadCloud,
  PlusCircle,
  FileText,
  Trash2,
  FolderOpen
} from 'lucide-react';
import { SAMPLE_DOCUMENTS, MODEL_PERFORMANCE, CATEGORIES } from '../data/benchmarkData';
import { classifyDocument, readDocumentFile } from '../services/api';
import DocumentUploadModal from './DocumentUploadModal';

export default function Playground({ apiStatus }) {
  const [inputText, setInputText] = useState(SAMPLE_DOCUMENTS[0].fullText);
  const [selectedModel, setSelectedModel] = useState('svm');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [showPreprocessed, setShowPreprocessed] = useState(false);
  const [copied, setCopied] = useState(false);

  // User Documents Library state (persisted to localStorage)
  const [userDocuments, setUserDocuments] = useState(() => {
    try {
      const saved = localStorage.getItem('textclassify_user_docs');
      return saved ? JSON.parse(saved) : [];
    } catch {
      return [];
    }
  });
  const [exampleTab, setExampleTab] = useState('presets'); // 'presets' | 'custom'
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [activeFileMeta, setActiveFileMeta] = useState(null);
  const [isDragOver, setIsDragOver] = useState(false);


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
    setActiveFileMeta(null);
  };

  const handleCopyText = () => {
    navigator.clipboard.writeText(inputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Add Custom Document to Library
  const handleAddDocument = (newDoc) => {
    const updated = [newDoc, ...userDocuments];
    setUserDocuments(updated);
    try {
      localStorage.setItem('textclassify_user_docs', JSON.stringify(updated));
    } catch (err) {
      console.error("Failed to save to localStorage:", err);
    }
    setExampleTab('custom');
    setInputText(newDoc.fullText);
    handleClassify(newDoc.fullText, selectedModel);
  };

  // Delete Custom Document from Library
  const handleDeleteDocument = (docId, e) => {
    e.stopPropagation();
    const updated = userDocuments.filter(d => d.id !== docId);
    setUserDocuments(updated);
    try {
      localStorage.setItem('textclassify_user_docs', JSON.stringify(updated));
    } catch (err) {
      console.error("Failed to save to localStorage:", err);
    }
  };

  // Direct File Upload into Playground
  const handleDirectFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const parsed = await readDocumentFile(file);
      setActiveFileMeta({
        name: parsed.filename,
        size: (parsed.size / 1024).toFixed(1) + ' KB',
        words: parsed.wordCount
      });
      setInputText(parsed.text);
      handleClassify(parsed.text, selectedModel);
    } catch (err) {
      console.error("File upload failed:", err);
    }
  };

  // Drag and Drop on Input Area
  const handleInputDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragOver(true);
    } else if (e.type === "dragleave") {
      setIsDragOver(false);
    }
  };

  const handleInputDrop = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      try {
        const parsed = await readDocumentFile(file);
        setActiveFileMeta({
          name: parsed.filename,
          size: (parsed.size / 1024).toFixed(1) + ' KB',
          words: parsed.wordCount
        });
        setInputText(parsed.text);
        handleClassify(parsed.text, selectedModel);
      } catch (err) {
        console.error("Drop failed:", err);
      }
    }
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

      {/* DOCUMENT PRESET & CUSTOM DOCUMENTS TOOLBAR */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-3">
          
          {/* Tab Switcher: Presets vs My Documents */}
          <div className="flex items-center gap-1 bg-slate-100 dark:bg-slate-800/80 p-1 rounded-xl">
            <button
              onClick={() => setExampleTab('presets')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                exampleTab === 'presets'
                  ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-xs'
                  : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
              }`}
            >
              <BookOpen className="w-3.5 h-3.5" />
              Standard Presets
            </button>
            <button
              onClick={() => setExampleTab('custom')}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-bold transition-all ${
                exampleTab === 'custom'
                  ? 'bg-white dark:bg-slate-900 text-indigo-600 dark:text-indigo-400 shadow-xs'
                  : 'text-slate-500 hover:text-slate-800 dark:hover:text-slate-200'
              }`}
            >
              <FolderOpen className="w-3.5 h-3.5" />
              My Documents
              {userDocuments.length > 0 && (
                <span className="px-1.5 py-0.2 rounded-full text-[10px] bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-300">
                  {userDocuments.length}
                </span>
              )}
            </button>
          </div>

          {/* Quick Add / Upload Document Button */}
          <button
            onClick={() => setIsUploadModalOpen(true)}
            className="flex items-center gap-1.5 px-3.5 py-1.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-bold transition-all shadow-xs hover:shadow-indigo-500/25"
          >
            <PlusCircle className="w-3.5 h-3.5" />
            + Add Document
          </button>
        </div>

        {/* Presets List */}
        {exampleTab === 'presets' && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2 pt-1">
            {SAMPLE_DOCUMENTS.map((sample, idx) => (
              <button
                key={idx}
                onClick={() => handleSampleClick(sample)}
                className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800 hover:border-indigo-500 dark:hover:border-indigo-500 hover:bg-indigo-50/50 dark:hover:bg-indigo-950/30 transition-all text-left group"
              >
                <span className="text-base">{sample.icon}</span>
                <span className="text-xs font-semibold text-slate-700 dark:text-slate-300 group-hover:text-indigo-600 dark:group-hover:text-indigo-400">
                  {sample.category}
                </span>
              </button>
            ))}
          </div>
        )}

        {/* User Added Custom Documents */}
        {exampleTab === 'custom' && (
          <div className="pt-1">
            {userDocuments.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-2.5">
                {userDocuments.map((doc) => (
                  <div
                    key={doc.id}
                    onClick={() => {
                      setInputText(doc.fullText);
                      handleClassify(doc.fullText, selectedModel);
                    }}
                    className="flex items-center justify-between p-2.5 rounded-xl bg-slate-50 dark:bg-slate-800/60 border border-slate-200 dark:border-slate-800 hover:border-indigo-500 hover:bg-indigo-50/30 dark:hover:bg-indigo-950/30 transition-all cursor-pointer group"
                  >
                    <div className="flex items-center gap-2.5 min-w-0 pr-2">
                      <span className="text-base shrink-0">{doc.icon || '📄'}</span>
                      <div className="min-w-0">
                        <div className="text-xs font-bold text-slate-800 dark:text-slate-200 truncate group-hover:text-indigo-600 dark:group-hover:text-indigo-400">
                          {doc.title}
                        </div>
                        <div className="text-[10px] text-slate-400 truncate">
                          {doc.wordCount || doc.fullText.split(/\s+/).length} words • {doc.category || 'Custom'}
                        </div>
                      </div>
                    </div>
                    <button
                      onClick={(e) => handleDeleteDocument(doc.id, e)}
                      title="Delete document"
                      className="p-1 rounded-lg text-slate-400 hover:text-red-500 hover:bg-red-50 dark:hover:bg-red-950/40 transition-colors shrink-0"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-xl">
                <FileText className="w-6 h-6 text-slate-400 mx-auto mb-1.5" />
                <div className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                  No custom documents added yet
                </div>
                <div className="text-[11px] text-slate-400 mt-0.5">
                  Click the "+ Add Document" button above to add custom articles or import files.
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Main Two-Column Playground Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Input (7 cols) */}
        <div className="lg:col-span-7 space-y-4">
          
          {/* Production Model Info Banner */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-4 shadow-sm flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-100 dark:border-indigo-800/60 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
                <Cpu className="w-5 h-5" />
              </div>
              <div>
                <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 dark:text-slate-500">
                  MODEL USED
                </div>
                <div className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  Support Vector Machine
                  <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-emerald-100 dark:bg-emerald-950/60 text-emerald-700 dark:text-emerald-300 border border-emerald-300 dark:border-emerald-800">
                    Production
                  </span>
                </div>
              </div>
            </div>
            <div className="text-right hidden sm:block">
              <span className="text-xs text-slate-500 dark:text-slate-400">Status</span>
              <div className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5 justify-end">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
                Online
              </div>
            </div>
          </div>

          {/* Text Input Card */}
          <div 
            onDragEnter={handleInputDrag}
            onDragLeave={handleInputDrag}
            onDragOver={handleInputDrag}
            onDrop={handleInputDrop}
            className={`bg-white dark:bg-slate-900 rounded-2xl border transition-all p-5 shadow-sm space-y-3 ${
              isDragOver 
                ? 'border-indigo-500 ring-2 ring-indigo-500/20 bg-indigo-50/20 dark:bg-indigo-950/20' 
                : 'border-slate-200 dark:border-slate-800'
            }`}
          >
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                  DOCUMENT INPUT
                </span>
                <span className="text-xs text-slate-400 dark:text-slate-500 hidden sm:inline">
                  "Paste text or drop a file"
                </span>
              </div>
              
              <div className="flex items-center gap-2 text-xs text-slate-500 dark:text-slate-400">
                {/* Upload File Button */}
                <label className="cursor-pointer hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors flex items-center gap-1 px-2 py-1 rounded-lg bg-slate-50 dark:bg-slate-800 border border-slate-200 dark:border-slate-700">
                  <UploadCloud className="w-3.5 h-3.5" />
                  <span>Upload File</span>
                  <input
                    type="file"
                    onChange={handleDirectFileUpload}
                    accept=".txt,.md,.json,.csv"
                    className="hidden"
                  />
                </label>

                <button
                  onClick={handleCopyText}
                  className="hover:text-slate-800 dark:hover:text-slate-200 transition-colors flex items-center gap-1 px-1.5 py-1"
                  title="Copy text"
                >
                  <Copy className="w-3.5 h-3.5" />
                  {copied ? 'Copied' : 'Copy'}
                </button>
                <button
                  onClick={handleClear}
                  className="hover:text-red-600 dark:hover:text-red-400 transition-colors flex items-center gap-1 px-1.5 py-1"
                  title="Clear text"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                  Clear
                </button>
              </div>
            </div>

            {/* Active File Loaded Indicator */}
            {activeFileMeta && (
              <div className="flex items-center justify-between px-3 py-1.5 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-200 dark:border-indigo-800/80 text-indigo-700 dark:text-indigo-300 text-xs">
                <div className="flex items-center gap-2">
                  <FileText className="w-3.5 h-3.5" />
                  <span className="font-semibold">{activeFileMeta.name}</span>
                  <span className="text-indigo-400 dark:text-indigo-400">({activeFileMeta.size} • {activeFileMeta.words} words)</span>
                </div>
                <button 
                  onClick={() => setActiveFileMeta(null)}
                  className="text-indigo-400 hover:text-indigo-700 dark:hover:text-indigo-200"
                >
                  ✕
                </button>
              </div>
            )}

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
                placeholder="Paste news articles, technical document abstracts, science reports, political debates, or drag and drop files here..."
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
                  onClick={handleClear}
                  className="px-4 py-2.5 rounded-xl text-sm font-semibold text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white border border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 transition-all"
                >
                  Clear
                </button>
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
                      <span>Analyze Document →</span>
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
                  <span>CLASSIFICATION CONFIDENCE</span>
                  <span className="text-[11px] text-slate-400 font-normal">Model Scores</span>
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

              {/* IMPORTANT TF-IDF FEATURES */}
              <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-5 shadow-sm space-y-3">
                <div className="flex items-center justify-between">
                  <h3 className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                    <Hash className="w-4 h-4 text-indigo-500" />
                    IMPORTANT TF-IDF FEATURES
                  </h3>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Weights
                  </span>
                </div>

                {result.topKeywords && result.topKeywords.length > 0 ? (
                  <div className="flex flex-wrap gap-1.5">
                    {result.topKeywords.map((kw, idx) => (
                      <span
                        key={idx}
                        className="inline-flex items-center gap-1 px-2.5 py-1 rounded-lg text-xs font-medium bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 border border-slate-200/80 dark:border-slate-700/80"
                      >
                        <span className="text-indigo-600 dark:text-indigo-400 font-semibold">{kw.term}</span>
                        <span className="text-[10px] font-mono text-slate-400 dark:text-slate-500">({kw.weight})</span>
                      </span>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-slate-500 dark:text-slate-400 italic">
                    No strong vocabulary matches detected.
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
              <span className="text-4xl mb-3">🤖</span>
              <h4 className="text-base font-bold text-slate-800 dark:text-slate-200">
                Ready to analyze
              </h4>
              <p className="text-xs text-slate-500 dark:text-slate-400 mt-1 max-w-xs leading-relaxed">
                Enter a document and we'll automatically classify it.
              </p>
            </div>
          )}

        </div>

      </div>

      {/* Add / Upload Document Modal */}
      <DocumentUploadModal
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onAddDocument={handleAddDocument}
        selectedModel={selectedModel}
      />

    </div>
  );
}

