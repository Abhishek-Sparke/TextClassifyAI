/**
 * Empirical Benchmark and Dataset Metadata
 * Grounded directly in the trained 100k Multi-Domain 26-class pipeline results.
 */

export const DATASET_STATS = {
  totalDocuments: 2786,
  categoriesCount: 4,
  trainingSamples: 2218,
  testingSamples: 555,
  tfidfFeatures: 5000,
  averageWordCount: 197.7,
  medianWordCount: 59,
  bestModelName: "Multinomial Naive Bayes",
  bestAccuracy: "87.03%"
};

export const CATEGORIES = [
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
    id: "environment.climate",
    name: "Environment & Climate",
    badge: "Sustainability",
    icon: "🌱",
    color: "green",
    count: 7919,
    percentage: 7.9,
    description: "Atmospheric carbon modeling, renewable solar/wind energy, conservation ecology, and climate policy.",
    topKeywords: ["climat", "environ", "carbon", "emiss", "sequestr", "sustain", "photovolta", "ecolog"]
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
    id: "comp.graphics",
    name: "Computer Graphics",
    badge: "Technology",
    icon: "🎨",
    color: "indigo",
    count: 952,
    percentage: 1.0,
    description: "3D rendering, GPU shaders, polygon meshes, raytracing, and image formats.",
    topKeywords: ["graphic", "file", "imag", "program", "use", "format", "color", "anim"]
  },
  {
    id: "comp.os.ms-windows.misc",
    name: "MS Windows",
    badge: "Operating Systems",
    icon: "🪟",
    color: "blue",
    count: 945,
    percentage: 0.9,
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
    percentage: 1.0,
    description: "IBM PC architecture, motherboards, IDE/SCSI controllers, BIOS, and bus cards.",
    topKeywords: ["drive", "card", "scsi", "ide", "pc", "use", "system", "bus"]
  },
  {
    id: "comp.sys.mac.hardware",
    name: "Mac Hardware",
    badge: "Apple Hardware",
    icon: "🍏",
    color: "emerald",
    count: 923,
    percentage: 0.9,
    description: "Apple Macintosh hardware, PowerBook, Quadra, monitors, and SCSI devices.",
    topKeywords: ["mac", "appl", "drive", "powerbook", "scsi", "quadra", "monitor", "problem"]
  },
  {
    id: "comp.windows.x",
    name: "X Window System",
    badge: "Windowing Systems",
    icon: "💻",
    color: "cyan",
    count: 976,
    percentage: 1.0,
    description: "X11 window system, Xlib, Motif widgets, window managers, and display clients.",
    topKeywords: ["window", "server", "use", "xterm", "widget", "motif", "display", "client"]
  },
  {
    id: "misc.forsale",
    name: "For Sale",
    badge: "Commerce",
    icon: "🏷️",
    color: "amber",
    count: 955,
    percentage: 1.0,
    description: "Classified ads, consumer goods, price offers, electronics, and shipping.",
    topKeywords: ["sale", "offer", "new", "price", "ask", "sell", "ship", "condit"]
  },
  {
    id: "rec.autos",
    name: "Automobiles",
    badge: "Automotive",
    icon: "🚗",
    color: "red",
    count: 928,
    percentage: 0.9,
    description: "Cars, engines, automotive engineering, transmissions, and dealership purchasing.",
    topKeywords: ["car", "engin", "use", "one", "dealer", "drive", "price", "look"]
  },
  {
    id: "rec.motorcycles",
    name: "Motorcycles",
    badge: "Motorcycles",
    icon: "🏍️",
    color: "orange",
    count: 962,
    percentage: 1.0,
    description: "Motorcycles, riding technique, helmets, protective gear, and motorcycle maintenance.",
    topKeywords: ["bike", "ride", "motorcycl", "rider", "use", "one", "get", "helmet"]
  },
  {
    id: "rec.sport.baseball",
    name: "Baseball",
    badge: "Sports",
    icon: "⚾",
    color: "lime",
    count: 944,
    percentage: 0.9,
    description: "Major League Baseball, pitching stats, home runs, strikeouts, and playoff pennants.",
    topKeywords: ["game", "year", "team", "basebal", "run", "hit", "pitch", "player"]
  },
  {
    id: "rec.sport.hockey",
    name: "Hockey",
    badge: "Sports",
    icon: "🏒",
    color: "sky",
    count: 971,
    percentage: 1.0,
    description: "National Hockey League, Stanley Cup, goals, ice rinks, and power play rules.",
    topKeywords: ["game", "team", "play", "player", "hockey", "nhl", "year", "season"]
  },
  {
    id: "sci.crypt",
    name: "Cryptography",
    badge: "Security",
    icon: "🔐",
    color: "violet",
    count: 962,
    percentage: 1.0,
    description: "Public-key cryptography, RSA, Clipper chip, PGP, encryption algorithms, and security.",
    topKeywords: ["key", "encrypt", "use", "clipper", "chip", "secur", "system", "des"]
  },
  {
    id: "sci.electronics",
    name: "Electronics",
    badge: "Engineering",
    icon: "⚡",
    color: "yellow",
    count: 955,
    percentage: 1.0,
    description: "Circuit diagrams, semiconductors, microcontrollers, capacitors, and RF electronics.",
    topKeywords: ["use", "circuit", "chip", "power", "voltag", "ground", "output", "electron"]
  },
  {
    id: "sci.med",
    name: "Medicine",
    badge: "Healthcare",
    icon: "🩺",
    color: "teal",
    count: 956,
    percentage: 1.0,
    description: "Clinical medicine, pharmacology, diseases, patient symptoms, and biomedical research.",
    topKeywords: ["doctor", "medic", "treatment", "diseas", "patient", "use", "symptom", "effect"]
  },
  {
    id: "sci.space",
    name: "Space Science",
    badge: "Aerospace",
    icon: "🚀",
    color: "fuchsia",
    count: 953,
    percentage: 1.0,
    description: "Astronomy, NASA satellite missions, orbital dynamics, planets, and aerospace propulsion.",
    topKeywords: ["space", "nasa", "orbit", "launch", "moon", "satellit", "shuttl", "year"]
  },
  {
    id: "soc.religion.christian",
    name: "Christianity",
    badge: "Religion",
    icon: "✝️",
    color: "purple",
    count: 973,
    percentage: 1.0,
    description: "Christian doctrine, biblical theology, church history, gospel scriptures, and faith.",
    topKeywords: ["god", "christ", "jesu", "church", "christian", "bibl", "faith", "scriptur"]
  },
  {
    id: "talk.politics.guns",
    name: "Gun Politics",
    badge: "Policy",
    icon: "🎯",
    color: "stone",
    count: 884,
    percentage: 0.9,
    description: "Firearms policy, Second Amendment rights, legislation, and gun control.",
    topKeywords: ["gun", "firearm", "weapon", "right", "state", "law", "control", "amend"]
  },
  {
    id: "talk.politics.mideast",
    name: "Middle East Politics",
    badge: "Geopolitics",
    icon: "🌍",
    color: "emerald",
    count: 914,
    percentage: 0.9,
    description: "Middle Eastern geopolitical conflicts, peace treaties, and foreign affairs.",
    topKeywords: ["israel", "israeli", "arab", "peopl", "jew", "palestinian", "war", "state"]
  },
  {
    id: "talk.politics.misc",
    name: "Politics",
    badge: "Governance",
    icon: "🏛️",
    color: "pink",
    count: 754,
    percentage: 0.8,
    description: "Government policy, international trade, constitutional rights, and law.",
    topKeywords: ["govern", "peopl", "state", "law", "right", "presid", "polit", "tax"]
  },
  {
    id: "alt.atheism",
    name: "Atheism",
    badge: "Philosophy",
    icon: "🕊️",
    color: "rose",
    count: 775,
    percentage: 0.8,
    description: "Atheism, religious criticism, secular philosophy, ethics, morality, and humanist reasoning.",
    topKeywords: ["god", "atheist", "say", "religion", "one", "think", "moral", "peopl"]
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
  labels: ["Atheism", "Graphics", "MS Win", "IBM PC", "Mac HW", "Win X", "For Sale", "Autos", "Mcycles", "Baseball", "Hockey", "Crypt", "Electronics", "Medicine", "Space", "Christian", "Guns", "Mideast", "Politics", "Religion", "Business", "World News", "Arts", "Wellness", "Academics", "Climate"],
  matrix: [
    [94, 2, 0, 0, 1, 0, 0, 4, 3, 0, 0, 0, 2, 3, 3, 8, 2, 5, 6, 7, 4, 2, 9, 0, 0, 0],
    [1, 151, 6, 4, 3, 9, 1, 1, 1, 1, 1, 2, 2, 1, 0, 0, 1, 0, 0, 0, 1, 0, 4, 0, 0, 0],
    [0, 14, 117, 10, 4, 13, 2, 3, 1, 1, 0, 2, 4, 0, 0, 0, 1, 1, 0, 0, 7, 1, 8, 0, 0, 0],
    [0, 6, 15, 118, 16, 4, 5, 4, 1, 0, 0, 1, 15, 0, 1, 0, 0, 0, 0, 0, 4, 1, 1, 0, 0, 0],
    [0, 4, 3, 14, 127, 0, 6, 5, 0, 1, 0, 4, 3, 5, 2, 0, 1, 0, 0, 0, 4, 2, 4, 0, 0, 0],
    [0, 18, 5, 1, 1, 147, 3, 0, 1, 0, 1, 1, 2, 1, 2, 0, 2, 1, 1, 0, 1, 0, 7, 0, 0, 0],
    [0, 0, 2, 3, 6, 0, 148, 2, 1, 0, 1, 0, 4, 0, 2, 0, 2, 0, 0, 0, 10, 3, 7, 0, 0, 0],
    [1, 0, 1, 2, 0, 3, 7, 130, 10, 1, 1, 0, 3, 0, 2, 0, 0, 1, 0, 0, 11, 2, 11, 0, 0, 0],
    [1, 0, 0, 3, 0, 0, 1, 10, 134, 2, 2, 0, 4, 1, 1, 0, 1, 1, 2, 0, 6, 3, 20, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 177, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 10, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 174, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 10, 0, 0, 0],
    [0, 2, 2, 0, 1, 0, 1, 2, 0, 0, 0, 175, 3, 0, 0, 0, 0, 0, 2, 0, 3, 1, 0, 0, 0, 0],
    [0, 1, 5, 5, 9, 2, 4, 10, 3, 1, 0, 3, 119, 5, 2, 0, 1, 2, 2, 1, 10, 2, 4, 0, 0, 0],
    [2, 0, 0, 0, 1, 0, 0, 1, 4, 0, 0, 1, 2, 157, 1, 1, 1, 3, 0, 1, 5, 3, 8, 0, 0, 0],
    [1, 0, 1, 1, 0, 1, 1, 2, 2, 4, 0, 3, 1, 3, 134, 1, 0, 1, 5, 0, 11, 4, 15, 0, 0, 0],
    [17, 3, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 6, 2, 127, 1, 6, 4, 7, 0, 4, 14, 0, 0, 0],
    [3, 0, 0, 0, 1, 0, 0, 1, 3, 4, 0, 9, 2, 2, 5, 1, 119, 2, 7, 1, 2, 11, 4, 0, 0, 0],
    [6, 1, 0, 0, 0, 0, 0, 2, 1, 3, 1, 3, 0, 1, 0, 1, 1, 135, 5, 0, 1, 17, 5, 0, 0, 0],
    [2, 1, 0, 1, 0, 0, 0, 0, 1, 2, 0, 2, 0, 3, 4, 1, 7, 3, 96, 2, 8, 9, 9, 0, 0, 0],
    [13, 0, 1, 0, 0, 0, 1, 2, 2, 0, 0, 0, 0, 1, 2, 21, 10, 0, 4, 46, 1, 1, 16, 0, 0, 0],
    [1, 1, 2, 1, 0, 0, 1, 3, 4, 2, 2, 2, 0, 1, 1, 1, 0, 0, 1, 0, 4780, 174, 23, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 2, 0, 1, 5, 4, 1, 0, 1, 0, 0, 226, 4730, 24, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 0, 1, 3, 0, 1, 0, 15, 27, 1544, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1583, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1583, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1584],
  ]
};

export const SAMPLE_DOCUMENTS = [
  {
    title: "⚠️ Multi-Topic Ambiguity Benchmark",
    category: "Ambiguous / Multi-topic",
    icon: "⚠️",
    preview: "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football.",
    fullText: "The Chief Minister bought a Royal Enfield bike and went to Mars to see Jesus and play football."
  },
  {
    title: "🚀 Deep Space Planetary Science",
    category: "Space Science",
    icon: "🚀",
    preview: "NASA launched a spacecraft into orbit using cryogenic rocket propulsion...",
    fullText: "NASA launched a spacecraft into orbit using advanced cryogenic rocket propulsion. Equipped with deep space telemetry, the planetary probe will traverse the solar system to analyze cosmic radiation and planetary atmospheric composition."
  },
  {
    title: "⚾ Major League Baseball Championship",
    category: "Baseball",
    icon: "⚾",
    preview: "The starting pitcher delivered nine strikeouts over seven scoreless innings...",
    fullText: "The starting pitcher delivered a dominant performance with nine strikeouts over seven scoreless innings. In the bottom of the ninth, the clean-up hitter drove in two runs with a solid line drive over the outfield fence, securing the championship victory as the stadium erupted."
  },
  {
    title: "🏛️ Congressional Legislation & Federal Law",
    category: "Politics",
    icon: "🏛️",
    preview: "The government passed a new law regarding federal taxation and civil liberties...",
    fullText: "The government passed a new law regarding federal taxation, national fiscal policy, and civil liberties. Congressional representatives held an extensive legislative debate concerning constitutional rights, judicial oversight, and executive administration."
  },
  {
    title: "🎨 3D Real-time Raytracing & Shaders",
    category: "Computer Graphics",
    icon: "🎨",
    preview: "Modern 3D graphics hardware accelerates real-time raytracing shaders and polygon rasterization...",
    fullText: "Modern real-time ray tracing requires hardware-accelerated GPUs with dedicated shader cores. The 3D rendering pipeline transforms polygon meshes and texture maps using Vulkan and DirectX, computing vertex lighting, anti-aliasing, reflections, and shadow maps at high refresh rates."
  },
  {
    title: "🍕 Out-of-Domain Detection Benchmark",
    category: "Unknown / Out-of-Domain",
    icon: "❓",
    preview: "I ate pizza today and my laptop battery is low...",
    fullText: "I ate pizza today and my laptop battery is low."
  },
  {
    title: "Wall Street Quarterly Earnings & Market Surge",
    category: "Business & Finance",
    icon: "💼",
    preview: "Wall Street equity indexes surged in midday trading following strong quarterly earnings reports...",
    fullText: "Wall Street equity indexes surged in midday trading following strong quarterly earnings reports from leading tech conglomerates. Central bank governors signaled potential interest rate cuts as corporate revenue and consumer expenditure trends remained resilient."
  },
  {
    title: "Film Festival Drama & Cinematic Storytelling",
    category: "Entertainment & Arts",
    icon: "🎬",
    preview: "The new independent drama film captivates audiences with compelling cinematography and nuanced acting...",
    fullText: "The new independent drama film captivates audiences with compelling cinematography, nuanced character arcs, and masterful direction. Critics at the international film festival praised the lead actor's poignant performance and the soundtrack's emotive orchestration."
  },
  {
    title: "Atmospheric Climate Modeling & Solar Transition",
    category: "Environment & Climate",
    icon: "🌱",
    preview: "Atmospheric climate scientists published updated climate modeling simulations indicating accelerated...",
    fullText: "Atmospheric climate scientists published updated climate modeling simulations indicating accelerated polar glacier melt. Environmental conservation organizations advocate expanding solar and wind renewable energy infrastructure to achieve carbon neutrality."
  },
  {
    title: "Cardiovascular Fitness & Preventive Nutrition",
    category: "Health & Wellness",
    icon: "🧘",
    preview: "Clinical research confirms that regular aerobic exercise combined with balanced dietary nutrition...",
    fullText: "Clinical research confirms that regular aerobic exercise combined with balanced dietary nutrition significantly reduces biomarkers of cardiovascular disease. Preventive wellness protocols emphasizing metabolic fitness and circadian sleep cycles improve long-term longevity."
  },
  {
    title: "Higher Education Pedagogy & STEM Curriculum",
    category: "Education & Academics",
    icon: "🎓",
    preview: "University faculties are redesigning undergraduate STEM curricula to integrate interactive pedagogy...",
    fullText: "University faculties are redesigning undergraduate STEM curricula to integrate interactive pedagogy, empirical research seminars, and digital laboratory platforms. Scholarly dissertations highlight improved student learning retention and academic outcomes."
  },
  {
    title: "Real-time Raytracing & Shader Rendering",
    category: "Computer Graphics",
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
    title: "Deep Space Planetary Missions & Astrophysics",
    category: "Space Science",
    icon: "🚀",
    preview: "NASA deep space exploration probes utilize gravitational slingshots and ion thrusters to navigate...",
    fullText: "NASA deep space exploration probes utilize gravitational slingshots and ion thrusters to navigate interplanetary trajectories. High-resolution spectroscopic imaging from orbital telescopes measures cosmic microwave radiation, solar flare flux, and planetary atmospheric composition."
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
    description: "Accepts raw text documents, articles, emails, or news stories in any character encoding.",
    details: "Supports variable-length text from brief abstracts to multi-page essays across all 26 benchmark domains."
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
    description: "Converts preprocessed tokens into an 8,000-dimensional TF-IDF vector capturing unigrams and bigrams.",
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
    details: "Provides instant probability distribution across all 26 predefined topic categories."
  }
];
