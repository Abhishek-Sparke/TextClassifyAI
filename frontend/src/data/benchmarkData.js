/**
 * Empirical Benchmark and Dataset Metadata
 * Grounded directly in the trained 20 Newsgroups pipeline results.
 */

export const DATASET_STATS = {
  totalDocuments: 3613,
  categoriesCount: 4,
  trainingSamples: 2883,
  testingSamples: 721,
  tfidfFeatures: 5000,
  averageWordCount: 214.5,
  medianWordCount: 142,
  bestModelName: "Support Vector Machine (Linear SVM)",
  bestAccuracy: "89.74%"
};

export const CATEGORIES = [
  {
    id: "comp.graphics",
    name: "Computer Graphics",
    badge: "Technology",
    icon: "💻",
    color: "indigo",
    count: 953,
    percentage: 26.4,
    description: "3D rendering, GPU shaders, polygon meshes, OpenGL/Vulkan, anti-aliasing, and computer animation.",
    topKeywords: ["graphics", "image", "3d", "render", "polygon", "format", "color", "animation"]
  },
  {
    id: "rec.sport.baseball",
    name: "Baseball",
    badge: "Sports",
    icon: "⚾",
    color: "amber",
    count: 951,
    percentage: 26.3,
    description: "Major league baseball, pitching statistics, home runs, innings, playoffs, and roster strategies.",
    topKeywords: ["baseball", "pitcher", "inning", "hitter", "strikeout", "run", "game", "team"]
  },
  {
    id: "sci.space",
    name: "Space Science",
    badge: "Science",
    icon: "🚀",
    color: "sky",
    count: 953,
    percentage: 26.4,
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
    percentage: 20.9,
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
    accuracy: 89.74,
    precision: 89.75,
    recall: 89.74,
    f1Score: 89.74,
    trainingTime: "0.114 s",
    type: "Maximum Margin Linear Hyperplane",
    description: "Excels in high-dimensional sparse TF-IDF spaces by maximizing the geometric margin between decision boundaries."
  },
  {
    id: "logistic_regression",
    name: "Logistic Regression",
    shortName: "Logistic Reg",
    isBest: false,
    badge: "Runner Up",
    accuracy: 89.32,
    precision: 89.35,
    recall: 89.32,
    f1Score: 89.31,
    trainingTime: "0.144 s",
    type: "Multinomial Softmax Classifier",
    description: "Generates calibrated probabilistic predictions using multinomial cross-entropy with L2 regularization."
  },
  {
    id: "naive_bayes",
    name: "Multinomial Naive Bayes",
    shortName: "Naive Bayes",
    isBest: false,
    badge: "Fastest Inference",
    accuracy: 89.04,
    precision: 89.06,
    recall: 89.04,
    f1Score: 89.02,
    trainingTime: "0.004 s",
    type: "Generative Probabilistic Model",
    description: "Ultra-fast probabilistic classifier leveraging conditional word independence and Laplace smoothing."
  },
  {
    id: "random_forest",
    name: "Random Forest",
    shortName: "Random Forest",
    isBest: false,
    badge: "Ensemble",
    accuracy: 83.50,
    precision: 83.74,
    recall: 83.50,
    f1Score: 83.37,
    trainingTime: "0.369 s",
    type: "Ensemble of 150 Decision Trees",
    description: "Bagging ensemble of randomized decision trees. Robust against non-linear patterns."
  }
];

// 4x4 Confusion Matrix corresponding to the SVM test split (N = 721)
// Classes: [Computer Graphics, Baseball, Space Science, Politics]
export const CONFUSION_MATRIX = {
  labels: ["Graphics", "Baseball", "Space", "Politics"],
  matrix: [
    [174, 3, 11, 3],   // Actual Graphics: 174 correctly classified
    [2, 182, 4, 2],    // Actual Baseball: 182 correctly classified
    [9, 5, 172, 5],    // Actual Space: 172 correctly classified
    [12, 6, 12, 119]   // Actual Politics: 119 correctly classified
  ]
};

export const SAMPLE_DOCUMENTS = [
  {
    title: "James Webb Deep Field & Planetary Orbit",
    category: "Space Science",
    icon: "🚀",
    preview: "The space telescope captured deep field infrared spectroscopy of newly discovered nebulae in orbit around distant stars...",
    fullText: "The James Webb Space Telescope and Hubble instruments have captured extraordinary deep-space spectra of distant stellar nebulae. NASA scientists and astrophysicists are analyzing orbital trajectories and gravitational anomalies to calibrate propulsion thrust for deep-space probes exploring the outer solar system."
  },
  {
    title: "Major League Baseball Game Final Inning",
    category: "Baseball",
    icon: "⚾",
    preview: "The starting pitcher tossed a two-hit shutout into the ninth inning before a dramatic walk-off grand slam...",
    fullText: "The starting pitcher dominated the game with eleven strikeouts across eight scoreless innings. In the bottom of the ninth, with two runners on base, the batter crushed a 98-mph fastball over the left field wall for a game-winning home run, thrilling the stadium crowd."
  },
  {
    title: "Real-time Raytracing & Shader Rendering",
    category: "Computer Graphics",
    icon: "💻",
    preview: "Modern 3D graphics hardware accelerates real-time raytracing shaders and polygon rasterization using compute cores...",
    fullText: "Next-generation GPU architectures compute real-time ray-traced reflections and global illumination using dedicated hardware shaders. High-resolution texture buffers and anti-aliasing passes ensure photorealistic rendering of complex 3D polygon meshes at 120 frames per second."
  },
  {
    title: "Congressional Debate on International Treaties",
    category: "Politics",
    icon: "🏛️",
    preview: "The senate held a bipartisan debate on foreign trade treaties, budget expenditures, and executive appointments...",
    fullText: "Congressional lawmakers convened to debate the ratification of international trade pacts, federal fiscal policy, and judicial confirmations. Both majority and minority committee members presented arguments regarding executive authority, public taxation, and constitutional civil rights protections."
  }
];

export const PIPELINE_STEPS = [
  {
    number: "01",
    title: "Document Input",
    subtitle: "Raw Unstructured Text",
    description: "Accepts raw text documents, articles, emails, or forum posts in any character encoding.",
    details: "Supports variable-length text from brief abstracts to multi-page essays."
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
    details: "Provides instant probability distribution across all predefined topic categories."
  }
];
