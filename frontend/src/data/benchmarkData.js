/**
 * Empirical Benchmark and Dataset Metadata
 * Grounded directly in the trained 20 Newsgroups 6-class pipeline results.
 */

export const DATASET_STATS = {
  totalDocuments: 5489,
  categoriesCount: 6,
  trainingSamples: 4391,
  testingSamples: 1098,
  tfidfFeatures: 5000,
  averageWordCount: 187.7,
  medianWordCount: 83,
  bestModelName: "Support Vector Machine (Linear SVM)",
  bestAccuracy: "87.07%"
};

export const CATEGORIES = [
  {
    id: "comp.graphics",
    name: "Computer Graphics",
    badge: "Technology",
    icon: "💻",
    color: "indigo",
    count: 953,
    percentage: 17.4,
    description: "3D rendering, GPU shaders, polygon meshes, OpenGL/Vulkan, anti-aliasing, and computer animation.",
    topKeywords: ["graphics", "image", "3d", "render", "polygon", "format", "color", "animation"]
  },
  {
    id: "rec.autos",
    name: "Automobiles",
    badge: "Transport",
    icon: "🚗",
    color: "cyan",
    count: 930,
    percentage: 16.9,
    description: "Automotive engineering, powertrains, vehicle mechanics, transmissions, engines, and road handling.",
    topKeywords: ["car", "cars", "engine", "dealer", "speed", "miles", "drive", "oil"]
  },
  {
    id: "rec.sport.baseball",
    name: "Sports",
    badge: "Sports",
    icon: "⚽",
    color: "amber",
    count: 951,
    percentage: 17.3,
    description: "Major league baseball, pitching statistics, home runs, innings, playoffs, and roster strategies.",
    topKeywords: ["baseball", "pitcher", "inning", "hitter", "strikeout", "run", "game", "team"]
  },
  {
    id: "sci.med",
    name: "Medicine",
    badge: "Healthcare",
    icon: "🩺",
    color: "rose",
    count: 957,
    percentage: 17.4,
    description: "Clinical diagnosis, biomedical research, pharmacology, pathology, treatments, and medical science.",
    topKeywords: ["doctor", "disease", "treatment", "pain", "medical", "patients", "clinical", "syndrome"]
  },
  {
    id: "sci.space",
    name: "Space Science",
    badge: "Science",
    icon: "🚀",
    color: "sky",
    count: 953,
    percentage: 17.4,
    description: "Planetary exploration, NASA missions, satellite orbital mechanics, space telescopes, and astrophysics.",
    topKeywords: ["space", "nasa", "orbit", "satellite", "launch", "rocket", "shuttle", "planetary"]
  },
  {
    id: "talk.politics.misc",
    name: "Politics",
    badge: "Governance",
    icon: "🏛️",
    color: "emerald",
    count: 756,
    percentage: 13.8,
    description: "Congressional legislation, foreign policy treaties, civil rights, constitutional law, and governance.",
    topKeywords: ["government", "policy", "congress", "law", "president", "state", "rights", "political"]
  }
];

export const MODEL_PERFORMANCE = [
  {
    id: "svm",
    name: "Support Vector Machine",
    shortName: "Linear SVM",
    isBest: true,
    badge: "Best Overall",
    accuracy: 87.07,
    precision: 87.07,
    recall: 87.07,
    f1Score: 87.06,
    trainingTime: "0.208 s",
    type: "Maximum Margin Linear Hyperplane",
    description: "Excels in high-dimensional sparse TF-IDF spaces by maximizing the geometric margin between decision boundaries."
  },
  {
    id: "logistic_regression",
    name: "Logistic Regression",
    shortName: "Logistic Reg",
    isBest: false,
    badge: "Runner Up",
    accuracy: 86.34,
    precision: 86.39,
    recall: 86.34,
    f1Score: 86.33,
    trainingTime: "0.239 s",
    type: "Multinomial Softmax Classifier",
    description: "Generates calibrated probabilistic predictions using multinomial cross-entropy with L2 regularization."
  },
  {
    id: "naive_bayes",
    name: "Multinomial Naive Bayes",
    shortName: "Naive Bayes",
    isBest: false,
    badge: "Fastest Inference",
    accuracy: 85.70,
    precision: 85.87,
    recall: 85.70,
    f1Score: 85.73,
    trainingTime: "0.006 s",
    type: "Generative Probabilistic Model",
    description: "Ultra-fast probabilistic classifier leveraging conditional word independence and Laplace smoothing."
  },
  {
    id: "random_forest",
    name: "Random Forest",
    shortName: "Random Forest",
    isBest: false,
    badge: "Ensemble",
    accuracy: 78.42,
    precision: 79.66,
    recall: 78.42,
    f1Score: 78.39,
    trainingTime: "0.462 s",
    type: "Ensemble of 150 Decision Trees",
    description: "Bagging ensemble of randomized decision trees. Robust against non-linear patterns."
  }
];

// 6x6 Confusion Matrix corresponding to the Linear SVM test split (N = 1098)
// Order: [Graphics, Autos, Sports, Medicine, Space, Politics]
export const CONFUSION_MATRIX = {
  labels: ["Graphics", "Autos", "Sports", "Medicine", "Space", "Politics"],
  matrix: [
    [167, 5, 4, 5, 6, 3],
    [5, 166, 3, 5, 2, 5],
    [3, 4, 171, 2, 4, 5],
    [6, 1, 5, 164, 10, 5],
    [7, 6, 4, 6, 162, 6],
    [0, 6, 9, 3, 7, 126]
  ]
};

export const SAMPLE_DOCUMENTS = [
  {
    title: "Real-time Raytracing & Shader Rendering",
    category: "Technology",
    icon: "💻",
    preview: "Modern 3D graphics hardware accelerates real-time raytracing shaders and polygon rasterization...",
    fullText: "Modern real-time ray tracing requires hardware-accelerated GPUs with dedicated shader cores. The 3D rendering pipeline transforms polygon meshes and texture maps using Vulkan and DirectX, computing vertex lighting, anti-aliasing, reflections, and shadow maps at high refresh rates."
  },
  {
    title: "High-Performance Automotive Engineering",
    category: "Automobiles",
    icon: "🚗",
    preview: "Turbocharged internal combustion engines optimize thermal efficiency through variable valve timing...",
    fullText: "Turbocharged internal combustion engines optimize thermal efficiency through variable valve timing, dual-clutch transmission gearing, and electronic fuel injection. Aerodynamic chassis design combined with independent suspension improves torque delivery and brake caliper responsiveness."
  },
  {
    title: "Major League Baseball Championship Game",
    category: "Sports",
    icon: "⚽",
    preview: "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings...",
    fullText: "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings. In the bottom of the ninth, the clean-up hitter drove in two runs with a solid line drive over the outfield fence, securing the championship victory as the stadium erupted."
  },
  {
    title: "Clinical Pharmacology & Diagnostic Medicine",
    category: "Medicine",
    icon: "🩺",
    preview: "Patients presenting with acute cardiovascular symptoms received clinical assessment including biomarker enzyme assays...",
    fullText: "Patients presenting with acute cardiovascular symptoms received clinical assessment including biomarker enzyme assays, electrocardiogram monitoring, and targeted antimicrobial therapies. Controlled clinical trials indicate significant efficacy in lowering blood serum cholesterol and mitigating chronic autoimmune inflammatory response."
  },
  {
    title: "Deep Space Planetary Missions & Astrophysics",
    category: "Science",
    icon: "🚀",
    preview: "NASA deep space exploration probes utilize gravitational slingshots and ion thrusters to navigate interplanetary trajectories...",
    fullText: "NASA deep space exploration probes utilize gravitational slingshots and ion thrusters to navigate interplanetary trajectories. High-resolution spectroscopic imaging from orbital telescopes measures cosmic microwave radiation, solar flare flux, and planetary atmospheric composition."
  },
  {
    title: "Congressional Legislative Debate & Reform",
    category: "Politics",
    icon: "🏛️",
    preview: "Congress held an extensive legislative debate on national fiscal reform, international trade treaties...",
    fullText: "Congress held an extensive legislative debate on national fiscal reform, international trade treaties, and civil rights. Leaders presented constitutional arguments regarding government budget allocation, judicial oversight, and executive appointments."
  }
];

export const PIPELINE_STEPS = [
  {
    number: "01",
    title: "Document Input",
    subtitle: "Raw Unstructured Text",
    description: "Accepts raw text documents, articles, emails, or forum posts in any character encoding.",
    details: "Supports variable-length text from brief abstracts to multi-page essays across 6 distinct domains."
  },
  {
    number: "02",
    title: "Text Preprocessing",
    subtitle: "Noise & Artifact Cleaning",
    description: "Lowercases text, strips headers, URLs, emails, numbers, and punctuation. Eliminates stopwords and lemmatizes words to base forms.",
    details: "Uses NLTK WordNetLemmatizer and English stop-word filtering to preserve pure semantic content."
  },
  {
    number: "03",
    title: "TF-IDF Extraction",
    subtitle: "Numerical Feature Representation",
    description: "Converts preprocessed tokens into a 5,000-dimensional TF-IDF vector capturing unigrams and bigrams.",
    details: "Applies sublinear scaling TF = 1 + log(tf) and L2 normalization without data leakage."
  },
  {
    number: "04",
    title: "Machine Learning Model",
    subtitle: "Multi-Class Pattern Classification",
    description: "Passes high-dimensional vectors to trained algorithms (Linear SVM, Logistic Regression, Naive Bayes, or Random Forest).",
    details: "Employs probability calibration to generate confidence estimates and multi-class margins."
  },
  {
    number: "05",
    title: "Predicted Category",
    subtitle: "Confidence & Explainability",
    description: "Outputs predicted category label, calibrated confidence score (%), and top influential TF-IDF key terms.",
    details: "Provides instant probability distribution across all 6 predefined topic categories."
  }
];
