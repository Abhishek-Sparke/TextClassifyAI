import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import Playground from './components/Playground';
import ModelBenchmarks from './components/ModelBenchmarks';
import ConfusionMatrix from './components/ConfusionMatrix';
import PipelineViewer from './components/PipelineViewer';
import DatasetExplorer from './components/DatasetExplorer';
import Footer from './components/Footer';
import { API_BASE_URL } from './services/api';

export default function App() {
  const [activeTab, setActiveTab] = useState('playground');
  const [darkMode, setDarkMode] = useState(() => {
    // Check localStorage or system preference
    const saved = localStorage.getItem('theme');
    if (saved) return saved === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  const [apiStatus, setApiStatus] = useState({
    connected: false,
    checking: true
  });

  // Sync dark mode class with HTML document
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
      localStorage.setItem('theme', 'dark');
    } else {
      document.documentElement.classList.remove('dark');
      localStorage.setItem('theme', 'light');
    }
  }, [darkMode]);

  // Ping backend to check if Python server is available
  useEffect(() => {
    const checkBackend = async () => {
      if (!API_BASE_URL) {
        setApiStatus({ connected: false, checking: false });
        return;
      }
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 1200);
        const res = await fetch(`${API_BASE_URL}/api/health`, { signal: controller.signal });
        clearTimeout(timeoutId);
        if (res.ok) {
          setApiStatus({ connected: true, checking: false });
          return;
        }
      } catch (e) {
        // Backend not running; fallback to client-side engine
      }
      setApiStatus({ connected: false, checking: false });
    };

    checkBackend();
  }, []);

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 text-slate-900 dark:text-slate-100 flex flex-col font-sans transition-colors duration-200">
      
      {/* Top Navigation */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        darkMode={darkMode}
        setDarkMode={setDarkMode}
        apiStatus={apiStatus}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 pt-6 pb-12">
        {activeTab === 'playground' && <Playground apiStatus={apiStatus} />}
        {activeTab === 'benchmarks' && <ModelBenchmarks />}
        {activeTab === 'confusion' && <ConfusionMatrix />}
        {activeTab === 'pipeline' && <PipelineViewer />}
        {activeTab === 'dataset' && <DatasetExplorer />}
      </main>

      {/* Footer */}
      <Footer />

    </div>
  );
}
