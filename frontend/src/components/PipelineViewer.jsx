import React, { useState } from 'react';
import { 
  GitFork, 
  ShieldCheck, 
  CheckCircle2, 
  ArrowRight, 
  Code2, 
  Sparkles, 
  FileCode, 
  Lock, 
  Cpu, 
  Zap, 
  ChevronDown, 
  ChevronUp 
} from 'lucide-react';
import { PIPELINE_STEPS } from '../data/benchmarkData';

export default function PipelineViewer() {
  const [activeStep, setActiveStep] = useState(0);

  const pipelineDeepDive = [
    {
      title: "Raw Text Ingestion",
      tags: ["20 Newsgroups", "Corpus Loading", "UTF-8"],
      details: [
        "Ingests raw unstructured text documents from newsgroups archives.",
        "Removes email header artifacts (From:, Subject:, Organization:) to prevent superficial dataset leakage.",
        "Filters empty and degenerate documents, maintaining balanced class distributions."
      ],
      codeSnippet: `from sklearn.datasets import fetch_20newsgroups

dataset = fetch_20newsgroups(
    subset='all',
    categories=['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.misc'],
    remove=('headers', 'footers', 'quotes')
)`
    },
    {
      title: "Multi-Stage NLP Text Cleaning",
      tags: ["Regex", "NLTK", "Lemmatization", "Stopwords"],
      details: [
        "Lowercase Normalization: Converts all characters to uniform case.",
        "Regex Filtering: Strips URLs (http/https), emails, punctuation marks, and standalone numeric digits.",
        "Stopword Removal: Eliminates common English filler words ('the', 'is', 'at') using standard NLTK stopwords.",
        "WordNet Lemmatization: Reduces inflected word forms to canonical morphological roots (e.g. 'orbiting' -> 'orbit', 'pitchers' -> 'pitcher')."
      ],
      codeSnippet: `import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'https?:\\/\\/\\S+|www\\.\\S+', '', text)
    text = re.sub(r'[^a-zA-Z\\s]', '', text)
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(tokens)`
    },
    {
      title: "TF-IDF Vectorization & Leakage Prevention",
      tags: ["5,000 Dimensions", "Sublinear TF", "Unigram + Bigram", "L2 Norm"],
      details: [
        "Data Leakage Safeguard: The TfidfVectorizer is strictly fit ONLY on the 80% training split, then applied via transform() to test documents.",
        "Sublinear TF Scaling: Employs 1 + log(tf) scaling to dampen the disproportionate influence of high-frequency words.",
        "N-gram Expansion: Extracts both single words (unigrams) and 2-word pairs (bigrams, e.g., 'space shuttle', 'home run').",
        "L2 Normalization: Normalizes vector lengths to unit Euclidean norm to make document representations invariant to document length."
      ],
      codeSnippet: `from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
    norm='l2'
)

# Crucial: Fit ONLY on training data!
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)`
    },
    {
      title: "Supervised Classification & Training",
      tags: ["Linear SVM", "Logistic Regression", "Naive Bayes", "Random Forest"],
      details: [
        "Linear Support Vector Machine (LinearSVC): Solves convex quadratic optimization to locate the maximum geometric margin separating hyperplanes.",
        "Logistic Regression (Multinomial): Optimizes multinomial log-loss with L2 ridge regularization and L-BFGS solver.",
        "Multinomial Naive Bayes: Evaluates class prior and conditional feature likelihoods with Laplace smoothing (alpha=1.0).",
        "Random Forest Classifier: Ensembles 150 randomized decision trees using Gini impurity and sub-sampling."
      ],
      codeSnippet: `from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

base_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
# Calibrate decision values to generate true probabilities
calibrated_svm = CalibratedClassifierCV(base_svm, method='sigmoid', cv='prefit')

base_svm.fit(X_train_tfidf, y_train)
calibrated_svm.fit(X_train_tfidf, y_train)`
    },
    {
      title: "Prediction & Explainability",
      tags: ["Softmax Probabilities", "Feature Attribution", "Latency < 2ms"],
      details: [
        "Outputs calibrated multi-class probability scores across all 4 categories.",
        "Extracts top influential TF-IDF terms by matching non-zero indices back to vectorizer feature names.",
        "Provides transparent, human-auditable explanations for each decision boundary outcome."
      ],
      codeSnippet: `y_pred = calibrated_svm.predict(X_test_tfidf)
probabilities = calibrated_svm.predict_proba(sample_vector)[0]

# Extract influential features
feature_names = vectorizer.get_feature_names_out()
top_indices = sample_vector.toarray()[0].argsort()[::-1][:10]`
    }
  ];

  return (
    <div className="space-y-8 animate-fade-in">
      
      {/* Header */}
      <div>
        <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-50 dark:bg-indigo-950/70 border border-indigo-200 dark:border-indigo-800 text-indigo-600 dark:text-indigo-400 text-xs font-semibold uppercase tracking-wider mb-2">
          <GitFork className="w-3.5 h-3.5" />
          End-to-End NLP Architecture
        </div>
        <h2 className="text-2xl sm:text-3xl font-bold tracking-tight text-slate-900 dark:text-white">
          Data Preprocessing & Machine Learning Pipeline
        </h2>
        <p className="text-sm text-slate-500 dark:text-slate-400 mt-1 max-w-3xl">
          Modular engineering architecture engineered to avoid data snooping, normalize textual variance, and maximize linear separability.
        </p>
      </div>

      {/* Interactive 5-Step Pipeline Flow Diagram */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
        {PIPELINE_STEPS.map((step, idx) => {
          const isSelected = activeStep === idx;
          return (
            <button
              key={idx}
              onClick={() => setActiveStep(idx)}
              className={`p-4 rounded-2xl text-left border transition-all relative overflow-hidden flex flex-col justify-between ${
                isSelected
                  ? 'bg-white dark:bg-slate-900 border-indigo-600 dark:border-indigo-500 shadow-md ring-2 ring-indigo-500/20'
                  : 'bg-white/60 dark:bg-slate-900/60 border-slate-200 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700'
              }`}
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded-md ${
                    isSelected
                      ? 'bg-indigo-600 text-white'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'
                  }`}>
                    STEP {step.number}
                  </span>
                  {idx < PIPELINE_STEPS.length - 1 && (
                    <ArrowRight className="w-3.5 h-3.5 text-slate-300 dark:text-slate-700 hidden md:block" />
                  )}
                </div>
                <h4 className="text-xs font-bold text-slate-900 dark:text-white line-clamp-1">
                  {step.title}
                </h4>
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mt-1 line-clamp-2">
                  {step.subtitle}
                </p>
              </div>

              <div className="mt-3 pt-2 border-t border-slate-100 dark:border-slate-800/80 text-[10px] font-semibold text-indigo-600 dark:text-indigo-400">
                {isSelected ? 'Viewing Step' : 'Click to inspect'}
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Step Deep-Dive Inspector */}
      <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 p-6 shadow-sm">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          
          {/* Left: Step Details (6 cols) */}
          <div className="lg:col-span-6 space-y-4">
            <div>
              <div className="text-xs font-mono font-bold text-indigo-600 dark:text-indigo-400 uppercase tracking-wider mb-1">
                Stage {PIPELINE_STEPS[activeStep].number} Specification
              </div>
              <h3 className="text-xl font-bold text-slate-900 dark:text-white">
                {pipelineDeepDive[activeStep].title}
              </h3>
            </div>

            <div className="flex flex-wrap gap-1.5">
              {pipelineDeepDive[activeStep].tags.map((tag, i) => (
                <span 
                  key={i}
                  className="px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-indigo-50 dark:bg-indigo-950/60 text-indigo-600 dark:text-indigo-400 border border-indigo-200/60 dark:border-indigo-800/60"
                >
                  {tag}
                </span>
              ))}
            </div>

            <ul className="space-y-2.5 pt-2">
              {pipelineDeepDive[activeStep].details.map((item, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                  <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Right: Code Implementation Snippet (6 cols) */}
          <div className="lg:col-span-6 flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between pb-2 text-xs font-mono text-slate-500 dark:text-slate-400">
                <span className="flex items-center gap-1.5 font-semibold">
                  <Code2 className="w-3.5 h-3.5 text-indigo-500" />
                  Scikit-Learn / Python Pipeline Snippet
                </span>
                <span>python</span>
              </div>
              <div className="p-4 rounded-xl bg-slate-950 text-slate-200 font-mono text-[11px] leading-relaxed overflow-x-auto border border-slate-800 shadow-inner">
                <pre>{pipelineDeepDive[activeStep].codeSnippet}</pre>
              </div>
            </div>
          </div>

        </div>
      </div>

      {/* Critical Leakage Prevention Guarantee Card */}
      <div className="p-6 rounded-2xl bg-gradient-to-r from-emerald-950/20 via-slate-900 to-slate-900 border border-emerald-500/30 text-white shadow-md">
        <div className="flex flex-col md:flex-row md:items-center gap-5">
          <div className="w-12 h-12 rounded-2xl bg-emerald-500/20 border border-emerald-400/30 flex items-center justify-center text-emerald-400 shrink-0">
            <ShieldCheck className="w-6 h-6" />
          </div>
          <div>
            <h4 className="text-base font-bold text-white flex items-center gap-2">
              <span>Methodological Rigor: Zero Data Leakage</span>
              <span className="text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-400/30">
                Verified
              </span>
            </h4>
            <p className="text-xs text-slate-300 mt-1 leading-relaxed">
              In accordance with academic machine learning best practices, text preprocessing parameters and the TF-IDF feature vocabulary are <strong>fit strictly on the 80% training partition</strong>. The 20% test partition acts as an entirely unseen simulation of real-world inference, guaranteeing that reported accuracy, precision, and recall metrics are completely unbiased.
            </p>
          </div>
        </div>
      </div>

    </div>
  );
}
