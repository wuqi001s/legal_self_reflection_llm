# Legal Agentic RAG - Project Configuration

# Model paths (local)
MODEL_PATHS = {
    "base_model": "/home/y/project/hgmodels/Qwen3-0.6B",
    "embedding_model": "/home/y/project/hgmodels/bge-small-zh-v1.5",
    "reranker_model": None,  # Not downloaded, use None for now
}

# Data Configuration
DATA_CONFIG = {
    "civil_code_docx": "/home/y/project/llm/project_law/dataset/中华人民共和国民法典_20200528.docx",
    "lora_csv": "/home/y/project/llm/project_law/dataset/lawzhidao_filter.csv",
    
    "output_dir": "./data",
    "civil_code_clauses": "./data/civil_code_clauses.json",
    "lora_raw": "./data/lora_raw.json",
    "lora_distilled": "./data/lora_distilled.jsonl",
    "train_file": "./data/train.jsonl",
    
    "adversarial_ratio": 0.15,  # 15% negative samples
}

# Retrieval Configuration
RETRIEVAL_CONFIG = {
    "faiss_index_path": "./data/legal_faiss.index",
    "documents_path": "./data/documents.json",
    "embedding_dimension": 512,  # bge-small is 512 dims
    "coarse_top_k": 20,
    "rerank_top_k": 5,
    "use_hyde": True,
}

# Training Configuration
TRAINING_CONFIG = {
    "output_dir": "./models/legal_qlora",
    "merged_output_dir": "./models/legal_merged",
    
    # LoRA Configuration
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    
    # Training Configuration
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "num_train_epochs": 3,
    "max_seq_length": 1024,
    
    # Quantization
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    
    # Other
    "logging_steps": 10,
    "save_steps": 100,
    "warmup_steps": 100,
}

# Agent Configuration
AGENT_CONFIG = {
    "max_iterations": 3,
    "reflection_threshold": 0.7,
    "unsafe_keywords": ["杀人", "抢劫", "贩毒", "伪造证据", "恐怖袭击", "如何犯罪"],
}
