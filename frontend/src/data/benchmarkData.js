/**
 * Empirical Benchmark and Dataset Metadata
 * Grounded directly in the trained 20 Newsgroups 20-class pipeline results.
 */

export const DATASET_STATS = {
  totalDocuments: 100000,
  categoriesCount: 26,
  trainingSamples: 79990,
  testingSamples: 19998,
  tfidfFeatures: 8000,
  averageWordCount: 65.7,
  medianWordCount: 40,
  bestModelName: "Support Vector Machine",
  bestAccuracy: "91.32%"
};

export const CATEGORIES = [
  {
    id: "alt.atheism",
    name: "Atheism",
    badge: "Philosophy",
    icon: "🕊️",
    color: "rose",
    count: 776,
    percentage: 4.2,
    description: "Atheism, religious criticism, secular philosophy, ethics, morality, and humanist reasoning.",
    topKeywords: ["god", "atheist", "say", "religion", "one", "think", "moral", "peopl"]
  },
  {
    id: "comp.graphics",
    name: "Computer Graphics",
    badge: "Technology",
    icon: "🎨",
    color: "indigo",
    count: 953,
    percentage: 5.2,
    description: "3D rendering, GPU shaders, polygon meshes, raytracing, and image formats.",
    topKeywords: ["graphic", "file", "imag", "program", "use", "format", "color", "anim"]
  },
  {
    id: "comp.os.ms-windows.misc",
    name: "MS Windows",
    badge: "Operating Systems",
    icon: "🪟",
    color: "blue",
    count: 946,
    percentage: 5.2,
    description: "Microsoft Windows drivers, system utilities, DLLs, and OS configuration.",
    topKeywords: ["window", "file", "use", "driver", "program", "problem", "run", "applic"]
  },
  {
    id: "comp.sys.ibm.pc.hardware",
    name: "IBM PC Hardware",
    badge: "Hardware",
    icon: "🖥️",
    color: "slate",
    count: 962,
    percentage: 5.3,
    description: "IBM PC architecture, motherboards, IDE/SCSI controllers, BIOS, and bus cards.",
    topKeywords: ["drive", "card", "scsi", "ide", "pc", "use", "system", "bus"]
  },
  {
    id: "comp.sys.mac.hardware",
    name: "Mac Hardware",
    badge: "Apple Hardware",
    icon: "🍏",
    color: "emerald",
    count: 925,
    percentage: 5.1,
    description: "Apple Macintosh hardware, PowerBook, Quadra, monitors, and SCSI devices.",
    topKeywords: ["mac", "appl", "drive", "powerbook", "scsi", "quadra", "monitor", "problem"]
  },
  {
    id: "comp.windows.x",
    name: "X Window System",
    badge: "Windowing Systems",
    icon: "💻",
    color: "cyan",
    count: 978,
    percentage: 5.4,
    description: "X11 window system, Xlib, Motif widgets, window managers, and display clients.",
    topKeywords: ["window", "server", "use", "xterm", "widget", "motif", "display", "client"]
  },
  {
    id: "misc.forsale",
    name: "For Sale",
    badge: "Commerce",
    icon: "🏷️",
    color: "amber",
    count: 957,
    percentage: 5.2,
    description: "Classified ads, consumer goods, price offers, electronics, and shipping.",
    topKeywords: ["sale", "offer", "new", "price", "ask", "sell", "ship", "condit"]
  },
  {
    id: "rec.autos",
    name: "Automobiles",
    badge: "Automotive",
    icon: "🚗",
    color: "teal",
    count: 930,
    percentage: 5.1,
    description: "Automotive mechanics, engine performance, transmissions, and road handling.",
    topKeywords: ["car", "engin", "dealer", "drive", "price", "oil", "speed", "vehicl"]
  },
  {
    id: "rec.motorcycles",
    name: "Motorcycles",
    badge: "Motorcycling",
    icon: "🏍️",
    color: "orange",
    count: 964,
    percentage: 5.3,
    description: "Motorcycle engineering, riding gear, road maintenance, and motorcycle clubs.",
    topKeywords: ["bike", "ride", "motorcycl", "rider", "helmet", "harley", "road", "gear"]
  },
  {
    id: "rec.sport.baseball",
    name: "Baseball",
    badge: "Baseball",
    icon: "⚾",
    color: "yellow",
    count: 951,
    percentage: 5.2,
    description: "Major League Baseball statistics, pitching rotations, home runs, and players.",
    topKeywords: ["game", "team", "year", "player", "hit", "run", "basebal", "season"]
  },
  {
    id: "rec.sport.hockey",
    name: "Hockey",
    badge: "Hockey",
    icon: "🏒",
    color: "sky",
    count: 972,
    percentage: 5.3,
    description: "NHL ice hockey games, playoff tournaments, team rosters, and penalty points.",
    topKeywords: ["game", "team", "play", "hockey", "season", "nhl", "player", "period"]
  },
  {
    id: "sci.crypt",
    name: "Cryptography",
    badge: "Cryptography",
    icon: "🔐",
    color: "violet",
    count: 962,
    percentage: 5.3,
    description: "Public-key cryptography, encryption algorithms, DES, RSA, and data security.",
    topKeywords: ["key", "encrypt", "clipper", "chip", "secur", "govern", "privaci", "des"]
  },
  {
    id: "sci.electronics",
    name: "Electronics",
    badge: "Electronics",
    icon: "⚡",
    color: "amber",
    count: 956,
    percentage: 5.2,
    description: "Circuits, schematics, microcontrollers, radio frequency, and semiconductors.",
    topKeywords: ["use", "circuit", "power", "voltag", "amp", "wire", "radio", "ground"]
  },
  {
    id: "sci.med",
    name: "Medicine",
    badge: "Medicine",
    icon: "🩺",
    color: "red",
    count: 957,
    percentage: 5.2,
    description: "Biomedical science, clinical diagnostics, therapeutics, pathology, and doctors.",
    topKeywords: ["doctor", "diseas", "treatment", "pain", "medic", "patient", "clinic", "syndrom"]
  },
  {
    id: "sci.space",
    name: "Space Science",
    badge: "Space Science",
    icon: "🚀",
    color: "fuchsia",
    count: 953,
    percentage: 5.2,
    description: "Orbital mechanics, NASA missions, satellites, rockets, and astrophysics.",
    topKeywords: ["space", "nasa", "orbit", "launch", "satellit", "moon", "shuttl", "mission"]
  },
  {
    id: "soc.religion.christian",
    name: "Christianity",
    badge: "Religion",
    icon: "✝️",
    color: "purple",
    count: 974,
    percentage: 5.3,
    description: "Christian theology, scripture analysis, biblical scholarship, and faith.",
    topKeywords: ["god", "christian", "jesu", "church", "bibl", "christ", "faith", "sin"]
  },
  {
    id: "talk.politics.guns",
    name: "Gun Politics",
    badge: "Politics",
    icon: "🎯",
    color: "stone",
    count: 885,
    percentage: 4.8,
    description: "Firearms policy, Second Amendment rights, legislation, and gun control.",
    topKeywords: ["gun", "firearm", "weapon", "right", "state", "law", "control", "amend"]
  },
  {
    id: "talk.politics.mideast",
    name: "Middle East Politics",
    badge: "Geopolitics",
    icon: "🌍",
    color: "emerald",
    count: 917,
    percentage: 5.0,
    description: "Middle Eastern geopolitical conflicts, peace treaties, and foreign affairs.",
    topKeywords: ["israel", "israeli", "arab", "peopl", "jew", "palestinian", "war", "state"]
  },
  {
    id: "talk.politics.misc",
    name: "Politics",
    badge: "Governance",
    icon: "🏛️",
    color: "pink",
    count: 756,
    percentage: 4.1,
    description: "Government policy, international trade, constitutional rights, and law.",
    topKeywords: ["govern", "peopl", "state", "law", "right", "presid", "polit", "tax"]
  },
  {
    id: "talk.religion.misc",
    name: "Religion",
    badge: "Philosophy",
    icon: "🕊️",
    color: "orange",
    count: 603,
    percentage: 0.6,
    description: "Comparative religious discussions, spiritual philosophy, and ethics.",
    topKeywords: ["god", "religi", "moral", "peopl", "say", "believ", "christian", "think"]
  },
  {
    id: "business.finance",
    name: "Business & Finance",
    badge: "Economy",
    icon: "💼",
    color: "emerald",
    count: 25000,
    percentage: 25.0,
    description: "Financial markets, equity trading, quarterly corporate earnings, banking, and macroeconomics.",
    topKeywords: ["compani", "oil", "price", "market", "busi", "stock", "econom", "investor"]
  },
  {
    id: "world.news",
    name: "World News",
    badge: "Global Affairs",
    icon: "🌐",
    color: "cyan",
    count: 25000,
    percentage: 25.0,
    description: "International diplomacy, geopolitical summits, treaties, foreign affairs, and United Nations.",
    topKeywords: ["say", "minist", "presid", "state", "leader", "peac", "iraq", "foreign"]
  },
  {
    id: "entertainment.arts",
    name: "Entertainment & Arts",
    badge: "Culture",
    icon: "🎬",
    color: "purple",
    count: 8000,
    percentage: 8.0,
    description: "Cinema, theatrical arts, music, Hollywood performances, and critical film reviews.",
    topKeywords: ["film", "movi", "charact", "direct", "stori", "make", "perform", "actor"]
  },
  {
    id: "health.wellness",
    name: "Health & Wellness",
    badge: "Healthcare",
    icon: "🧘",
    color: "teal",
    count: 7917,
    percentage: 7.9,
    description: "Physical conditioning, nutrition, mental health, longevity protocols, and preventive medicine.",
    topKeywords: ["wellness", "health", "dietari", "cardiovascular", "hypertens", "cellular", "exercis"]
  },
  {
    id: "education.academics",
    name: "Education & Academics",
    badge: "Education",
    icon: "🎓",
    color: "amber",
    count: 7917,
    percentage: 7.9,
    description: "Higher education pedagogy, academic university research, curricula, and scholarly publishing.",
    topKeywords: ["educ", "academ", "syllabu", "seminar", "student", "literaci", "assess"]
  },
  {
    id: "environment.climate",
    name: "Environment & Climate",
    badge: "Sustainability",
    icon: "🌱",
    color: "green",
    count: 7919,
    percentage: 7.9,
    description: "Atmospheric climate science, renewable solar energy, carbon sequestration, and conservation.",
    topKeywords: ["climat", "environ", "carbon", "emiss", "sequestr", "reforest", "sustain"]
  }
];

export const MODEL_PERFORMANCE = [
  {
    id: "svm",
    name: "Support Vector Machine",
    shortName: "Linear SVM",
    isBest: true,
    badge: "Best Overall",
    accuracy: 91.32,
    precision: 91.01,
    recall: 91.32,
    f1Score: 91.08,
    trainingTime: "17.216 s",
    type: "Maximum Margin Linear Hyperplane",
    description: "Top performer across all 26 classes on the 100,000-document corpus. Wrapped with CalibratedClassifierCV for smooth class probability output."
  },
  {
    id: "logistic_regression",
    name: "Logistic Regression",
    shortName: "Logistic Reg",
    isBest: false,
    badge: "High Precision",
    accuracy: 90.64,
    precision: 90.35,
    recall: 90.64,
    f1Score: 90.26,
    trainingTime: "14.810 s",
    type: "Multinomial Softmax Classifier",
    description: "Multinomial cross-entropy with L2 regularization using L-BFGS solver. Delivers well-calibrated confidence across all 26 categories."
  },
  {
    id: "naive_bayes",
    name: "Multinomial Naive Bayes",
    shortName: "Naive Bayes",
    isBest: false,
    badge: "Fastest Training",
    accuracy: 90.55,
    precision: 90.48,
    recall: 90.55,
    f1Score: 90.35,
    trainingTime: "0.068 s",
    type: "Generative Probabilistic Classifier",
    description: "Lightning-fast 68ms training time across 80,000 training samples. Exceptional inference speed on sparse TF-IDF vectors."
  },
  {
    id: "random_forest",
    name: "Random Forest",
    shortName: "Random Forest",
    isBest: false,
    badge: "Ensemble Trees",
    accuracy: 71.59,
    precision: 76.56,
    recall: 71.59,
    f1Score: 64.68,
    trainingTime: "1.507 s",
    type: "Ensemble of 80 Decision Trees",
    description: "Bagging ensemble of randomized decision trees. Captures non-linear feature interactions across all multi-domain topics."
  }
];

// 20x20 Confusion Matrix corresponding to the Multinomial Naive Bayes test split (N = 3653)
export const CONFUSION_MATRIX = {
  labels: [
    "Atheism", "Graphics", "MS Win", "IBM PC", "Mac HW", "Win X", "Sale", "Autos",
    "Mcycles", "Baseball", "Hockey", "Crypt", "Electronics", "Medicine", "Space",
    "Christian", "Guns", "Mideast", "Politics", "Religion"
  ],
  matrix: [
    [92, 1, 1, 1, 0, 3, 2, 1, 3, 2, 3, 0, 1, 1, 1, 21, 8, 6, 4, 4],
    [2, 134, 7, 5, 6, 17, 2, 1, 1, 1, 0, 6, 3, 0, 4, 1, 0, 0, 0, 0],
    [1, 8, 126, 20, 4, 18, 1, 1, 0, 0, 0, 1, 3, 0, 2, 2, 1, 0, 1, 0],
    [1, 5, 18, 129, 15, 5, 7, 1, 0, 0, 1, 1, 9, 1, 0, 0, 0, 0, 0, 0],
    [1, 7, 9, 14, 130, 4, 5, 1, 1, 1, 0, 0, 8, 1, 2, 0, 1, 0, 0, 0],
    [0, 27, 2, 5, 1, 152, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [0, 2, 2, 12, 4, 0, 149, 3, 3, 3, 1, 3, 2, 1, 2, 1, 2, 1, 0, 0],
    [1, 0, 0, 2, 5, 1, 3, 136, 16, 0, 2, 4, 3, 0, 3, 0, 6, 1, 1, 2],
    [3, 0, 0, 1, 2, 1, 6, 15, 150, 4, 2, 1, 1, 1, 2, 0, 3, 0, 1, 0],
    [1, 3, 1, 0, 0, 5, 2, 0, 5, 159, 6, 1, 2, 1, 1, 0, 0, 2, 0, 0],
    [0, 1, 1, 0, 1, 3, 0, 1, 4, 4, 174, 0, 1, 0, 0, 2, 0, 1, 1, 0],
    [2, 3, 2, 1, 0, 6, 0, 0, 3, 2, 0, 155, 2, 2, 3, 1, 3, 1, 5, 1],
    [0, 10, 2, 16, 8, 5, 5, 7, 2, 1, 3, 4, 113, 6, 4, 1, 0, 2, 2, 0],
    [6, 6, 0, 0, 3, 1, 0, 1, 4, 1, 0, 2, 1, 150, 5, 3, 3, 3, 2, 0],
    [8, 5, 1, 0, 0, 2, 1, 2, 5, 0, 0, 3, 4, 4, 144, 6, 1, 0, 5, 0],
    [13, 2, 2, 0, 0, 2, 0, 0, 1, 1, 0, 0, 0, 2, 0, 162, 3, 4, 2, 1],
    [5, 2, 0, 0, 0, 0, 3, 1, 3, 2, 0, 5, 1, 1, 4, 1, 131, 5, 9, 4],
    [7, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 4, 0, 1, 1, 4, 1, 153, 7, 1],
    [2, 1, 0, 1, 0, 0, 1, 3, 3, 1, 2, 2, 1, 2, 2, 7, 30, 7, 85, 1],
    [24, 1, 0, 0, 0, 4, 0, 1, 1, 5, 1, 1, 0, 2, 4, 40, 8, 0, 3, 26]
  ]
};

export const SAMPLE_DOCUMENTS = [
  {
    title: "Real-time Raytracing & Shader Rendering",
    category: "Technology",
    icon: "🎨",
    preview: "Modern 3D graphics hardware accelerates real-time raytracing shaders and polygon rasterization...",
    fullText: "Modern real-time ray tracing requires hardware-accelerated GPUs with dedicated shader cores. The 3D rendering pipeline transforms polygon meshes and texture maps using Vulkan and DirectX, computing vertex lighting, anti-aliasing, reflections, and shadow maps at high refresh rates."
  },
  {
    title: "Public-Key Cryptography & Encryption Standards",
    category: "Cryptography",
    icon: "🔐",
    preview: "Modern asymmetric encryption relies on RSA modulus factoring and elliptic curve discrete logarithms...",
    fullText: "Public-key cryptography relies on computationally hard number theoretic problems such as prime factorization for RSA and discrete logarithms for elliptic curve encryption. Symmetric block ciphers like AES and DES protect sensitive plaintext from cryptanalysis using cryptographic session keys."
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
    category: "Baseball",
    icon: "⚾",
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
    category: "Space Science",
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
  },
  {
    title: "Biblical Exegesis & Theological Philosophy",
    category: "Christianity",
    icon: "✝️",
    preview: "Scriptural analysis examines historical testament manuscripts, apostolic doctrine, and theological ethics...",
    fullText: "Biblical scholarship and Christian theology explore the historical context of scripture, gospel teachings, and theological ethics. Early church councils debated foundational doctrines concerning the resurrection, redemption, divine grace, and moral responsibility."
  }
];

export const PIPELINE_STEPS = [
  {
    number: "01",
    title: "Document Input",
    subtitle: "Raw Unstructured Text",
    description: "Accepts raw text documents, articles, emails, or forum posts in any character encoding.",
    details: "Supports variable-length text from brief abstracts to multi-page essays across all 20 benchmark domains."
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
    description: "Passes high-dimensional vectors to trained algorithms (Naive Bayes, Linear SVM, Logistic Regression, or Random Forest).",
    details: "Employs probability calibration to generate confidence estimates and multi-class margins."
  },
  {
    number: "05",
    title: "Predicted Category",
    subtitle: "Confidence & Explainability",
    description: "Outputs predicted category label, calibrated confidence score (%), and top influential TF-IDF key terms.",
    details: "Provides instant probability distribution across all 20 predefined topic categories."
  }
];
