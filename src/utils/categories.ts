export const CATEGORY_LABELS: Record<string, string> = {
  introduction: "Introduction to AI",
  "machine-learning": "Machine Learning",
  "deep-learning": "Deep Learning",
  "nlp-llm": "NLP & LLMs",
  "computer-vision": "Computer Vision",
  "generative-ai": "Generative AI",
  "reinforcement-learning": "Reinforcement Learning",
  "ai-ethics-governance": "AI Ethics & Governance",
  "tools-frameworks": "Tools & Frameworks",
  general: "General",
};

export const CATEGORY_DESCRIPTIONS: Record<string, string> = {
  introduction:
    "Start your journey here. Learn what Artificial Intelligence is, its history, and its core concepts.",
  "machine-learning":
    "Dive into the fundamentals of Machine Learning, including supervised and unsupervised learning algorithms, model evaluation, and feature engineering.",
  "deep-learning":
    "Explore neural networks, backpropagation, CNNs for image processing, and RNNs for sequence data.",
  "nlp-llm":
    "Natural language processing and large language models — from tokenization to modern LLM architectures.",
  "computer-vision":
    "Teach machines to see — image segmentation, object detection, pose estimation, and more.",
  "generative-ai":
    "Learn about Large Language Models (LLMs), Transformers, Diffusion models, and how to build applications using them.",
  "reinforcement-learning":
    "Agents that learn by doing — reward signals, policy optimization, and RLHF.",
  "ai-ethics-governance":
    "Bias, fairness, safety, and the policy landscape shaping how AI gets built and deployed.",
  "tools-frameworks":
    "Get hands-on with popular AI tools and frameworks like PyTorch, TensorFlow, Scikit-learn, and Hugging Face.",
  general: "Miscellaneous notes that don't fit neatly into a single category.",
};

export const CATEGORY_ORDER = Object.keys(CATEGORY_LABELS);

export function categoryOf(id: string) {
  return id.includes("/") ? id.split("/")[0] : "general";
}

export function labelFor(category: string) {
  return CATEGORY_LABELS[category] ?? category;
}
