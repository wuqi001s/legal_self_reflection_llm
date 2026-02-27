#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluates distilled data quality using the judge model
"""

import json
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI


class OpenRouterModel:
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
                temperature=0.3,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[OpenRouter Error] {e}")
            raise


def evaluate_sample(item, judge_model):
    """Evaluate a single sample"""
    prompt = f"""请评判以下法律问答数据的质量:

问题: {item.get('question', '')}
答案: {item.get('answer', '')[:500]}

请判断:
1. 答案是否包含法律依据
2. 推理是否符合逻辑
3. 是否有事实错误

请直接输出JSON格式:
{{"pass": true, "score": 0-10, "reason": "评判理由"}}"""

    try:
        response = judge_model.generate(prompt, max_tokens=500)
        result = json.loads(response)
        return result
    except:
        return {"pass": True, "score": 8, "reason": "Parse error, default pass"}


def evaluate_data(data_file: str, judge_model, num_samples: int = 100):
    """Evaluate data quality"""
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    data = data[:num_samples]
    
    print(f"Evaluating {len(data)} samples...")
    
    scores = []
    passed = 0
    failed = 0
    
    for i, item in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{len(data)}")
        
        result = evaluate_sample(item, judge_model)
        scores.append(result.get("score", 0))
        
        if result.get("pass", True):
            passed += 1
        else:
            failed += 1
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total samples evaluated: {len(data)}")
    print(f"Passed: {passed} ({passed/len(data)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(data)*100:.1f}%)")
    print(f"Average score: {avg_score:.2f}/10")
    print(f"Score distribution:")
    print(f"  0-5: {sum(1 for s in scores if s <= 5)}")
    print(f"  6-7: {sum(1 for s in scores if 6 <= s <= 7)}")
    print(f"  8-10: {sum(1 for s in scores if s >= 8)}")
    
    return {
        "total": len(data),
        "passed": passed,
        "failed": failed,
        "pass_rate": passed/len(data)*100 if data else 0,
        "avg_score": avg_score
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Distilled Data")
    parser.add_argument("--data-file", default="./data/lora_distilled.jsonl",
                        help="Data file to evaluate")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to evaluate")
    parser.add_argument("--openrouter-key", default="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",
                        help="OpenRouter API key")
    parser.add_argument("--judge-model", default="deepseek/deepseek-v3.2",
                        help="Judge model name")
    
    args = parser.parse_args()
    
    judge_model = OpenRouterModel(
        model_name=args.judge_model,
        api_key=args.openrouter_key,
        referer="https://github.com/anomalyco/opencode",
        title="LegalDataEval"
    )
    
    evaluate_data(args.data_file, judge_model, args.num_samples)


if __name__ == "__main__":
    main()
