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
    name: "Sports",
    badge: "Sports",
    icon: "⚽",
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
// Classes: [Computer Graphics, Sports, Space Science, Politics]
export const CONFUSION_MATRIX = {
  labels: ["Graphics", "Sports", "Space", "Politics"],
  matrix: [
    [174, 3, 11, 3],   // Actual Graphics: 174 correctly classified
    [2, 182, 4, 2],    // Actual Sports: 182 correctly classified
    [9, 5, 172, 5],    // Actual Space: 172 correctly classified
    [12, 6, 12, 119]   // Actual Politics: 119 correctly classified
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
    title: "Major League Baseball Championship Game",
    category: "Sports",
    icon: "⚾",
    preview: "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings...",
    fullText: "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings. In the bottom of the ninth, the clean-up hitter drove in two runs with a solid line drive over the outfield fence, securing the championship victory as the stadium erupted."
  },
  {
    title: "Quarterly Financial Growth & Earnings",
    category: "Business",
    icon: "💼",
    preview: "The company reported increased revenue during the financial quarter. Investors are expecting stronger profits...",
    fullText: "The company reported increased revenue during the financial quarter. Investors are expecting stronger profits as the organization expands its products into international markets."
  },
  {
    title: "Congressional Legislative Debate & Reform",
    category: "Politics",
    icon: "🏛️",
    preview: "Congress held an extensive legislative debate on national fiscal reform, international trade treaties...",
    fullText: "Congress held an extensive legislative debate on national fiscal reform, international trade treaties, and civil rights. Leaders presented constitutional arguments regarding government budget allocation, judicial oversight, and executive appointments."
  },
  {
    title: "International Cinema Awards & Directing",
    category: "Entertainment",
    icon: "🎬",
    preview: "The critically acclaimed feature film received multiple nominations at the international cinema awards...",
    fullText: "The critically acclaimed feature film received multiple nominations at the international cinema awards. Critics praised the director's visionary storytelling, the orchestral musical score, and the lead actor's stirring theatrical performance."
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
