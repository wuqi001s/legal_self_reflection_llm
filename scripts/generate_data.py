#!/usr/bin/env python3
"""
Data Generation Pipeline
1. Parse Civil Code for RAG
2. Process lawzhidao for QLoRA training
"""

import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.legal_parser import ASTLegalParser, LawDataLoader, LegalDataGenerator
from src.data.data_distiller import LegalDataDistiller, AdversarialNegativeSampler

from openai import OpenAI


class OpenRouterModel:
    """OpenRouter model wrapper for teacher and judge models"""
    
    def __init__(self, model_name="deepseek/deepseek-r1-0528:free", api_key=None, referer="", title=""):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name
        self.extra_headers = {
            "HTTP-Referer": referer,
            "X-Title": title,
        } if referer or title else {}
    
    def generate(self, prompt, max_tokens=2048):
        messages = [{"role": "user", "content": prompt}]
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[OpenRouter Error] {e}")
            raise


def parse_civil_code(docx_path: str, output: str):
    """Parse Civil Code docx into structured clauses for RAG"""
    parser = ASTLegalParser()
    clauses = parser.parse_docx(docx_path)
    parser.save_clauses(clauses, output)
    print(f"[RAG] Parsed {len(clauses)} clauses from Civil Code to {output}")
    return clauses


def load_lora_data(csv_path: str, output: str, max_samples: int = None):
    """Load lawzhidao data where is_best=1"""
    data = LawDataLoader.load_best_qa(csv_path, max_samples)
    LawDataLoader.save_qa_pairs(data, output)
    print(f"[QLoRA] Loaded {len(data)} best QA pairs to {output}")
    return data


def distill_data(raw_qa_file: str, output: str, num_samples: int = 5000, teacher_model=None, judge_model=None):
    """Distill raw QA into high-quality training data using Teacher LLM"""
    with open(raw_qa_file, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
        
    print(f"[Distill] Distilling {min(len(raw_data), num_samples)} samples...")
    
    distiller = LegalDataDistiller(teacher_model=teacher_model, judge_model=judge_model)
    distilled = distiller.distill(raw_data[:num_samples], num_samples)
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for item in distilled:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"[Distill] Saved {len(distilled)} distilled samples to {output}")
    return distilled


def add_adversarial_samples(train_file: str, output: str, ratio: float = 0.15):
    """Add 15% adversarial negative samples for safe refusal"""
    with open(train_file, 'r', encoding='utf-8') as f:
        positives = [json.loads(line) for line in f]
        
    sampler = AdversarialNegativeSampler()
    negatives = sampler.generate_negative_samples(positives, ratio)
    
    merged = sampler.merge_with_positives(positives, negatives)
    sampler.save_training_data(merged, output)
    
    print(f"[Adversarial] Added {len(negatives)} negative samples, total {len(merged)}")


def main():
    parser = argparse.ArgumentParser(description="Legal Data Generation Pipeline")
    parser.add_argument("--civil-code", default="/home/y/project/llm/project_law/dataset/中华人民共和国民法典_20200528.docx", 
                        help="Civil Code docx path")
    parser.add_argument("--lora-csv", default="/home/y/project/llm/project_law/dataset/lawzhidao_filter.csv",
                        help="Lawzhidao CSV path")
    parser.add_argument("--output-dir", default="./data", help="Output directory")
    parser.add_argument("--max-lora-samples", type=int, default=10000, help="Max LORA samples to process")
    parser.add_argument("--num-distill", type=int, default=1000, help="Number of samples to distill")
    parser.add_argument("--adversarial-ratio", type=float, default=0.15, help="Adversarial negative ratio")
    parser.add_argument("--openrouter-key", default="",
                        help="OpenRouter API key")
    parser.add_argument("--teacher-model", default="deepseek/deepseek-v3.2",
                        help="Teacher model name")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v3.2",
                        help="Judge model name")
    parser.add_argument("--referer", default="https://github.com/anomalyco/opencode",
                        help="HTTP-Referer for OpenRouter")
    parser.add_argument("--title", default="LegalDataGen",
                        help="X-Title for OpenRouter")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Initializing OpenRouter models...")
    print("="*50)
    
    teacher_model = OpenRouterModel(
        model_name=args.teacher_model,
        api_key=args.openrouter_key,
        referer=args.referer,
        title=args.title
    )
    judge_model = OpenRouterModel(
        model_name=args.judge_model,
        api_key=args.openrouter_key,
        referer=args.referer,
        title=args.title
    )
    
    print(f"Teacher model: {args.teacher_model}")
    print(f"Judge model: {args.judge_model}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    clauses_output = output_dir / "civil_code_clauses.json"
    lora_raw_output = output_dir / "lora_raw.json"
    distilled_output = output_dir / "lora_distilled.jsonl"
    train_output = output_dir / "train.jsonl"
    
    print("\n" + "="*50)
    print("Step 1: Parse Civil Code for RAG")
    print("="*50)
    parse_civil_code(args.civil_code, str(clauses_output))
    
    print("\n" + "="*50)
    print("Step 2: Load LORA data (is_best=1)")
    print("="*50)
    load_lora_data(args.lora_csv, str(lora_raw_output), args.max_lora_samples)
    
    print("\n" + "="*50)
    print("Step 3: Distill with Teacher LLM")
    print("="*50)
    distill_data(str(lora_raw_output), str(distilled_output), args.num_distill, 
                 teacher_model=teacher_model, judge_model=judge_model)
    
    print("\n" + "="*50)
    print("Step 4: Add adversarial negative samples")
    print("="*50)
    add_adversarial_samples(str(distilled_output), str(train_output), args.adversarial_ratio)
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)
    print(f"  RAG clauses: {clauses_output}")
    print(f"  Training data: {train_output}")


if __name__ == "__main__":
    main()
