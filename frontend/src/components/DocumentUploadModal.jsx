import React, { useState, useRef } from 'react';
import { 
  X, 
  UploadCloud, 
  FileText, 
  PlusCircle, 
  CheckCircle2, 
  AlertCircle, 
  Layers, 
  Download, 
  Trash2,
  Sparkles,
  RefreshCw
} from 'lucide-react';
import { CATEGORIES } from '../data/benchmarkData';
import { readDocumentFile, classifyBatchDocuments } from '../services/api';

export default function DocumentUploadModal({ 
  isOpen, 
  onClose, 
  onAddDocument, 
  selectedModel = 'svm' 
}) {
  const [activeTab, setActiveTab] = useState('single'); // 'single' | 'batch'
  
  // Single Document Form State
  const [docTitle, setDocTitle] = useState('');
  const [docCategory, setDocCategory] = useState('auto');
  const [docContent, setDocContent] = useState('');
  const [uploadedFileMeta, setUploadedFileMeta] = useState(null);
  
  // Batch State
  const [batchFiles, setBatchFiles] = useState([]);
  const [batchParsedDocs, setBatchParsedDocs] = useState([]);
  const [batchResults, setBatchResults] = useState(null);
  const [isProcessingBatch, setIsProcessingBatch] = useState(false);
  const [dragActive, setDragActive] = useState(false);

  const fileInputRef = useRef(null);
  const batchFileInputRef = useRef(null);

  if (!isOpen) return null;

  // Handle Single File Upload
  const handleSingleFileChange = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const parsed = await readDocumentFile(file);
      setUploadedFileMeta({
        name: parsed.filename,
        size: (parsed.size / 1024).toFixed(1) + ' KB',
        words: parsed.wordCount
      });
      setDocContent(parsed.text);
      if (!docTitle) {
        setDocTitle(parsed.filename.replace(/\.[^/.]+$/, ""));
      }
    } catch (err) {
      console.error("Error reading file:", err);
    }
  };

  // Handle Drag and Drop for Single File
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDropSingle = async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      try {
        const parsed = await readDocumentFile(file);
        setUploadedFileMeta({
          name: parsed.filename,
          size: (parsed.size / 1024).toFixed(1) + ' KB',
          words: parsed.wordCount
        });
        setDocContent(parsed.text);
        if (!docTitle) {
          setDocTitle(parsed.filename.replace(/\.[^/.]+$/, ""));
        }
      } catch (err) {
        console.error("Drop error:", err);
      }
    }
  };

  // Save Single Document
  const handleSaveSingle = () => {
    if (!docContent.trim()) return;
    const newDoc = {
      id: `custom_${Date.now()}`,
      title: docTitle.trim() || `Document ${new Date().toLocaleTimeString()}`,
      fullText: docContent.trim(),
      category: docCategory === 'auto' ? 'Custom Added' : (CATEGORIES.find(c => c.id === docCategory)?.name || docCategory),
      categoryId: docCategory === 'auto' ? 'custom' : docCategory,
      icon: docCategory === 'auto' ? '📄' : (CATEGORIES.find(c => c.id === docCategory)?.icon || '📄'),
      snippet: docContent.trim().slice(0, 160) + (docContent.trim().length > 160 ? '...' : ''),
      wordCount: docContent.trim().split(/\s+/).length,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isCustom: true
    };
    onAddDocument(newDoc);
    onClose();
  };

  // Handle Batch Files Selection
  const handleBatchFileChange = async (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;

    let parsedList = [];
    for (const file of files) {
      try {
        const parsed = await readDocumentFile(file);
        if (parsed.batchItems && parsed.batchItems.length > 0) {
          parsedList.push(...parsed.batchItems);
        } else {
          parsedList.push({
            id: `batch_${file.name}_${parsedList.length}`,
            title: file.name,
            text: parsed.text,
            size: (file.size / 1024).toFixed(1) + ' KB'
          });
        }
      } catch (err) {
        console.error("Batch parse error:", err);
      }
    }
    setBatchFiles(files);
    setBatchParsedDocs(parsedList);
    setBatchResults(null);
  };

  // Run Bulk Classification
  const handleRunBatchClassification = async () => {
    if (!batchParsedDocs.length) return;
    setIsProcessingBatch(true);
    try {
      const response = await classifyBatchDocuments(batchParsedDocs, selectedModel);
      setBatchResults(response);
    } catch (err) {
      console.error("Batch classification error:", err);
    } finally {
      setIsProcessingBatch(false);
    }
  };

  // Export Batch Results to CSV
  const handleExportCSV = () => {
    if (!batchResults || !batchResults.results) return;
    const headers = ["Title", "Predicted Category", "Confidence (%)", "Word Count", "Top Keywords", "Text Snippet"];
    const rows = batchResults.results.map(r => [
      `"${(r.title || '').replace(/"/g, '""')}"`,
      `"${r.category}"`,
      r.confidence,
      r.wordCount,
      `"${(r.topKeywords || []).map(k => k.term).join(', ')}"`,
      `"${(r.snippet || '').replace(/"/g, '""')}"`
    ]);

    const csvContent = "data:text/csv;charset=utf-8," + [headers.join(','), ...rows.map(e => e.join(','))].join('\n');
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `textclassify_batch_results_${Date.now()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/70 backdrop-blur-sm animate-fade-in">
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl w-full max-w-2xl shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-slate-200 dark:border-slate-800">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-indigo-50 dark:bg-indigo-950/60 border border-indigo-100 dark:border-indigo-800/60 flex items-center justify-center text-indigo-600 dark:text-indigo-400">
              <PlusCircle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="text-lg font-bold text-slate-900 dark:text-white">
                Add & Upload Documents
              </h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Import files, add custom articles, or run batch classification
              </p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="w-8 h-8 rounded-lg flex items-center justify-center text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Tab Toggle */}
        <div className="flex border-b border-slate-200 dark:border-slate-800 px-5 pt-3 gap-4">
          <button
            onClick={() => setActiveTab('single')}
            className={`pb-3 text-xs font-bold uppercase tracking-wider transition-all border-b-2 ${
              activeTab === 'single'
                ? 'border-indigo-600 text-indigo-600 dark:text-indigo-400'
                : 'border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'
            }`}
          >
            Add Single Document
          </button>
          <button
            onClick={() => setActiveTab('batch')}
            className={`pb-3 text-xs font-bold uppercase tracking-wider transition-all border-b-2 ${
              activeTab === 'batch'
                ? 'border-indigo-600 text-indigo-600 dark:text-indigo-400'
                : 'border-transparent text-slate-400 hover:text-slate-600 dark:hover:text-slate-300'
            }`}
          >
            Batch Upload & CSV
          </button>
        </div>

        {/* Body Content */}
        <div className="p-5 overflow-y-auto flex-1 space-y-4">
          
          {activeTab === 'single' ? (
            <>
              {/* Drag & Drop File Box */}
              <div
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDropSingle}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-xl p-5 text-center cursor-pointer transition-all ${
                  dragActive 
                    ? 'border-indigo-500 bg-indigo-50/50 dark:bg-indigo-950/30' 
                    : 'border-slate-200 dark:border-slate-800 hover:border-indigo-400 bg-slate-50/60 dark:bg-slate-950/40'
                }`}
              >
                <input 
                  type="file" 
                  ref={fileInputRef} 
                  onChange={handleSingleFileChange} 
                  accept=".txt,.md,.json,.csv"
                  className="hidden" 
                />
                <UploadCloud className="w-8 h-8 text-indigo-500 mx-auto mb-2" />
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                  Click to browse or drag & drop document file
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Supports plain text (.txt, .md), JSON (.json), or CSV (.csv)
                </div>
                {uploadedFileMeta && (
                  <div className="mt-3 inline-flex items-center gap-2 px-3 py-1 rounded-lg bg-indigo-100 dark:bg-indigo-950 border border-indigo-200 dark:border-indigo-800 text-indigo-700 dark:text-indigo-300 text-xs font-mono">
                    <FileText className="w-3.5 h-3.5" />
                    <span>{uploadedFileMeta.name}</span>
                    <span>({uploadedFileMeta.size} • {uploadedFileMeta.words} words)</span>
                  </div>
                )}
              </div>

              {/* Title and Category Row */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Document Title
                  </label>
                  <input
                    type="text"
                    value={docTitle}
                    onChange={(e) => setDocTitle(e.target.value)}
                    placeholder="e.g. Hubble Deep Space Observation"
                    className="w-full px-3.5 py-2.5 rounded-xl text-sm bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-1.5">
                    Assigned Category
                  </label>
                  <select
                    value={docCategory}
                    onChange={(e) => setDocCategory(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-xl text-sm bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="auto">✨ Auto-Detect via AI Classifier</option>
                    {CATEGORIES.map(c => (
                      <option key={c.id} value={c.id}>
                        {c.icon} {c.name}
                      </option>
                    ))}
                  </select>
                </div>
              </div>

              {/* Content Textarea */}
              <div>
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-xs font-semibold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                    Document Text Content
                  </label>
                  <span className="text-xs text-slate-400 font-mono">
                    {docContent.trim() ? docContent.trim().split(/\s+/).length : 0} words
                  </span>
                </div>
                <textarea
                  rows={6}
                  value={docContent}
                  onChange={(e) => setDocContent(e.target.value)}
                  placeholder="Paste or write document text here..."
                  className="w-full p-3.5 rounded-xl text-sm bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-slate-800 dark:text-slate-200 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-sans"
                />
              </div>
            </>
          ) : (
            /* Batch Upload Tab */
            <div className="space-y-4">
              <div
                onClick={() => batchFileInputRef.current?.click()}
                className="border-2 border-dashed border-slate-200 dark:border-slate-800 hover:border-indigo-400 rounded-xl p-6 text-center cursor-pointer bg-slate-50/60 dark:bg-slate-950/40 transition-all"
              >
                <input
                  type="file"
                  ref={batchFileInputRef}
                  onChange={handleBatchFileChange}
                  multiple
                  accept=".txt,.md,.json,.csv"
                  className="hidden"
                />
                <Layers className="w-8 h-8 text-indigo-500 mx-auto mb-2" />
                <div className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                  Select Multiple Files or a CSV / JSON dataset
                </div>
                <div className="text-xs text-slate-400 mt-1">
                  Upload multiple .txt files or a CSV containing a 'text' column for batch classification
                </div>
              </div>

              {/* Parsed Items Summary */}
              {batchParsedDocs.length > 0 && (
                <div className="bg-slate-50 dark:bg-slate-950/80 rounded-xl p-4 border border-slate-200 dark:border-slate-800 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider flex items-center gap-2">
                      <FileText className="w-3.5 h-3.5 text-indigo-500" />
                      Loaded {batchParsedDocs.length} Documents
                    </div>
                    <button
                      onClick={handleRunBatchClassification}
                      disabled={isProcessingBatch}
                      className="px-3.5 py-1.5 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs font-semibold flex items-center gap-1.5 transition-all shadow-xs disabled:opacity-50"
                    >
                      {isProcessingBatch ? (
                        <>
                          <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                          Processing...
                        </>
                      ) : (
                        <>
                          <Sparkles className="w-3.5 h-3.5" />
                          Classify All ({batchParsedDocs.length})
                        </>
                      )}
                    </button>
                  </div>

                  {/* Batch Results Table */}
                  {batchResults && batchResults.results && (
                    <div className="space-y-3 pt-2">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-emerald-600 dark:text-emerald-400 font-semibold flex items-center gap-1">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Classified {batchResults.results.length} items in {batchResults.processingTime}s
                        </span>
                        <button
                          onClick={handleExportCSV}
                          className="text-xs font-medium text-indigo-600 dark:text-indigo-400 hover:underline flex items-center gap-1"
                        >
                          <Download className="w-3.5 h-3.5" />
                          Download CSV
                        </button>
                      </div>

                      <div className="max-h-52 overflow-y-auto rounded-lg border border-slate-200 dark:border-slate-800 text-xs">
                        <table className="w-full text-left">
                          <thead className="bg-slate-100 dark:bg-slate-900 sticky top-0 text-slate-600 dark:text-slate-400 border-b border-slate-200 dark:border-slate-800">
                            <tr>
                              <th className="p-2 font-semibold">Title</th>
                              <th className="p-2 font-semibold">Category</th>
                              <th className="p-2 font-semibold">Confidence</th>
                              <th className="p-2 font-semibold">Top Term</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-200 dark:divide-slate-800">
                            {batchResults.results.map((res, i) => (
                              <tr key={i} className="hover:bg-slate-100/50 dark:hover:bg-slate-800/50">
                                <td className="p-2 font-medium text-slate-800 dark:text-slate-200 truncate max-w-[120px]">
                                  {res.title}
                                </td>
                                <td className="p-2 flex items-center gap-1">
                                  <span>{res.icon}</span>
                                  <span className="font-semibold text-slate-700 dark:text-slate-300">{res.category}</span>
                                </td>
                                <td className="p-2 font-mono text-indigo-600 dark:text-indigo-400 font-semibold">
                                  {res.confidence}%
                                </td>
                                <td className="p-2 text-slate-500 font-mono">
                                  {res.topKeywords?.[0]?.term || '-'}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between p-4 bg-slate-50 dark:bg-slate-950/60 border-t border-slate-200 dark:border-slate-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-xl text-xs font-semibold text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white transition-all"
          >
            Cancel
          </button>
          
          {activeTab === 'single' && (
            <button
              onClick={handleSaveSingle}
              disabled={!docContent.trim()}
              className="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-700 disabled:opacity-50 text-white text-xs font-bold flex items-center gap-2 transition-all shadow-md hover:shadow-indigo-500/25"
            >
              <CheckCircle2 className="w-4 h-4" />
              Save to My Documents
            </button>
          )}
        </div>

      </div>
    </div>
  );
}
