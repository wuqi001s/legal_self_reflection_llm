#!/usr/bin/env python3
"""
Agentic Data Synthesis Pipeline
Implements real LLM + RAG environment interaction for high-quality training data
"""

import json
import random
import re
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI


class OpenRouterModel:
    def __init__(self, model_name="deepseek/deepseek-v3.2", api_key=None, referer="", title=""):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name
        self.extra_headers = {
            "HTTP-Referer": referer,
            "X-Title": title,
        } if referer or title else {}
    
    def generate(self, prompt, max_tokens=2048, temperature=0.7):
        messages = [{"role": "user", "content": prompt}]
        
        try:
            completion = self.client.chat.completions.create(
                extra_headers=self.extra_headers,
                extra_body={},
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return completion.choices[0].message.content
        except Exception as e:
            print(f"[OpenRouter Error] {e}")
            raise


class AgenticDataSynthesizer:
    """
    Synthesizes high-quality agentic trajectory data using:
    - Teacher LLM (brain)
    - Real RAG system (environment)
    - Multi-turn interaction
    """
    
    def __init__(self, teacher_model, retriever, max_iterations=3, negative_ratio=0.15):
        self.teacher = teacher_model
        self.retriever = retriever
        self.max_iterations = max_iterations
        self.negative_ratio = negative_ratio
    
    def synthesize(self, question: str) -> Dict:
        """Synthesize a single training sample with agentic trajectory"""
        
        is_negative = random.random() < self.negative_ratio
        
        if is_negative:
            return self._synthesize_negative(question)
        else:
            return self._synthesize_positive(question)
    
    def _synthesize_positive(self, question: str) -> Dict:
        """Synthesize positive sample with full agentic trajectory"""
        
        trajectory = []
        final_answer = None
        
        for iteration in range(self.max_iterations):
            if iteration == 0:
                prompt = self._build_thought_action_prompt(question)
                output = self.teacher.generate(prompt, max_tokens=500)
                
                thought, action = self._parse_thought_action(output)
                trajectory.append({
                    "thought": thought,
                    "action": action,
                    "observation": ""
                })
            else:
                prev_trajectory = trajectory[-1]
                prompt = self._build_reflection_prompt(question, prev_trajectory)
                output = self.teacher.generate(prompt, max_tokens=500)
                
                reflection, new_action, should_finalize = self._parse_reflection_output(output)
                
                trajectory[-1]["observation"] = self._get_observation(prev_trajectory["action"])
                trajectory[-1]["reflection"] = reflection
                
                if should_finalize:
                    final_prompt = self._build_final_answer_prompt(question, trajectory)
                    final_answer = self.teacher.generate(final_prompt, max_tokens=1500)
                    break
                
                if new_action:
                    trajectory.append({
                        "thought": reflection,
                        "action": new_action,
                        "observation": ""
                    })
        
        if final_answer is None:
            trajectory[-1]["observation"] = self._get_observation(trajectory[-1]["action"])
            final_prompt = self._build_final_answer_prompt(question, trajectory)
            final_answer = self.teacher.generate(final_prompt, max_tokens=1500)
        
        return {
            "instruction": "你是一个专业的AI法律智能体。请运用检索工具查阅相关法条，并在信息充分后，严格按照IRAC法则输出法律意见。",
            "input": question,
            "output": self._format_output(trajectory, final_answer),
            "metadata": {
                "source": "agentic_synthesized",
                "has_negative_sample": False,
                "iterations": len(trajectory)
            }
        }
    
    def _synthesize_negative(self, question: str) -> Dict:
        """Synthesize negative sample (no relevant docs found -> safe refusal)"""
        
        prompt = self._build_thought_action_prompt(question)
        output = self.teacher.generate(prompt, max_tokens=500)
        
        thought, action = self._parse_thought_action(output)
        
        trajectory = [{
            "thought": thought,
            "action": action,
            "observation": "[Observation] 检索无匹配结果（No relevant documents found）"
        }]
        
        refusal_prompt = f"""【注意】外部数据库没有查到任何法条。由于法律场景极度严谨，你绝对不能根据自己的记忆编造法条。

【用户提问】：{question}

【任务】：请直接输出 [Thought] 承认无信息，并输出拒绝回答的 [Final Answer]。格式：
[Thought] 你的思考...
[Final Answer] 抱歉，经过在现行法律数据库中检索，暂未找到与您诉求直接相关的明确法律条款支持。由于法律问题具有极强的专业性与严谨性，建议您携带相关证据材料直接咨询执业律师或前往当地劳动监察大队寻求帮助。"""
        
        final_answer = self.teacher.generate(refusal_prompt, max_tokens=500)
        
        return {
            "instruction": "你是一个专业的AI法律智能体。请运用检索工具查阅相关法条，并在信息充分后，严格按照IRAC法则输出法律意见。",
            "input": question,
            "output": f"[Thought] {thought}\n[Action] {action}\n[Observation] 检索无匹配结果。\n[Thought] 经过检索，知识库中未包含相关法律依据。为保证严谨性，我不能凭空生成法律建议。\n[Final Answer] {final_answer}",
            "metadata": {
                "source": "agentic_negative",
                "has_negative_sample": True,
                "should_refuse": True
            }
        }
    
    def _build_thought_action_prompt(self, question: str) -> str:
        return f"""你是一个严谨的AI法律智能体。现在的任务是分析用户的法律咨询。
由于你没有内部知识，你必须调用外部法律数据库。

【用户提问】：{question}

请你输出第一步的思考（Thought）和你需要执行的搜索动作（Action）。
动作格式必须严格为：Search_DB("你的搜索词")。
输出完 Action 后，请立刻停止生成！不要捏造检索结果！

【输出格式】：
[Thought] 你的思考过程...
[Action] Search_DB("关键词")"""
    
    def _build_reflection_prompt(self, question: str, prev_trajectory: Dict) -> str:
        prev_obs = prev_trajectory.get("observation", "")
        if not prev_obs:
            prev_obs = self._get_observation(prev_trajectory["action"])
        
        return f"""【历史轨迹】：
[Thought] {prev_trajectory['thought']}
[Action] {prev_trajectory['action']}
[Observation] {prev_obs}

【用户提问】：{question}

【任务指令】：
请你评估 [Observation] 中的信息是否足以完全解答用户的提问。
- 如果信息不足，请输出 [Reflection] 进行自我反思，并输出新的 [Action] 进行二次检索。
- 如果信息已经充分，请直接严格按照 IRAC（争议焦点-规则-适用-结论）法则输出 [Final Answer]。

【输出格式】（如果需要继续检索）：
[Reflection] 你的反思...
[Action] Search_DB("新的关键词")

【输出格式】（如果信息充分）：
[Final Answer] 你的完整法律意见（包含IRAC结构）"""
    
    def _build_final_answer_prompt(self, question: str, trajectory: List[Dict]) -> str:
        traj_str = "\n".join([
            f"[Thought] {t['thought']}\n[Action] {t['action']}\n[Observation] {t.get('observation', '')}"
            for t in trajectory
        ])
        
        return f"""【完整历史轨迹】：
{traj_str}

【用户提问】：{question}

【任务指令】：
基于以上检索结果，请严格按照 IRAC 法则输出最终的法律意见。

【IRAC格式】：
【Issue 争议焦点】：列出案件的核心法律争议点
【Rule 法律适用】：列出相关的法律条文和规定
【Application 事实分析】：将法律规则应用于具体案件事实
【Conclusion 法律结论】：给出最终的法律建议"""
    
    def _parse_thought_action(self, output: str) -> Tuple[str, str]:
        thought_match = re.search(r'\[Thought\](.*?)(?:\[Action\]|$)', output, re.DOTALL)
        action_match = re.search(r'\[Action\]\s*Search_DB\("(.*?)"\)', output)
        
        thought = thought_match.group(1).strip() if thought_match else "分析用户问题"
        action = action_match.group(1) if action_match else f'Search_DB("{output[:50]}")'
        
        return thought, f'Search_DB("{action}")'
    
    def _parse_reflection_output(self, output: str) -> Tuple[str, str, bool]:
        reflection_match = re.search(r'\[Reflection\](.*?)(?:\[Action\]|\[Final\]|$)', output, re.DOTALL)
        action_match = re.search(r'\[Action\]\s*Search_DB\("(.*?)"', output)
        final_match = re.search(r'\[Final Answer\](.*)', output, re.DOTALL)
        
        reflection = reflection_match.group(1).strip() if reflection_match else ""
        action = action_match.group(1) if action_match else ""
        
        return reflection, f'Search_DB("{action}")', final_match is not None
    
    def _get_observation(self, action: str) -> str:
        match = re.search(r'Search_DB\("(.*?)"\)', action)
        if not match:
            return "[Observation] 无法解析搜索词"
        
        query = match.group(1)
        
        try:
            docs = self.retriever.retrieve(query, top_k=3)
            if not docs:
                return "[Observation] 检索无匹配结果"
            
            obs_parts = ["[Observation] "]
            for i, doc in enumerate(docs, 1):
                obs_parts.append(f"【文档{i}】{doc.content[:300]}")
            
            return "\n".join(obs_parts)
        except Exception as e:
            return f"[Observation] 检索出错: {str(e)}"
    
    def _format_output(self, trajectory: List[Dict], final_answer: str) -> str:
        parts = []
        for t in trajectory:
            parts.append(f"[Thought] {t['thought']}")
            parts.append(f"[Action] {t['action']}")
            if t.get('observation'):
                parts.append(t['observation'])
            if t.get('reflection'):
                parts.append(f"[Reflection] {t['reflection']}")
        
        parts.append(f"\n{final_answer}")
        return "\n".join(parts)


def load_raw_data(csv_path: str, max_samples: int = None) -> List[Dict]:
    """Load raw QA data"""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    df = df[df['is_best'] == 1]
    df = df[df['question'].notna()]
    df = df[df['question'].str.strip() != '']
    
    if max_samples:
        df = df.head(max_samples)
    
    return df.to_dict('records')


def main():
    parser = argparse.ArgumentParser(description="Agentic Data Synthesis Pipeline")
    parser.add_argument("--lora-csv", default="/home/y/project/llm/project_law/dataset/lawzhidao_filter.csv",
                        help="Lawzhidao CSV path")
    parser.add_argument("--output", default="./data/agentic_train.jsonl",
                        help="Output file")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to synthesize")
    parser.add_argument("--max-iterations", type=int, default=3,
                        help="Max retrieval iterations")
    parser.add_argument("--negative-ratio", type=float, default=0.15,
                        help="Negative sample ratio")
    parser.add_argument("--openrouter-key", default="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",
                        help="OpenRouter API key")
    parser.add_argument("--teacher-model", default="deepseek/deepseek-v3.2",
                        help="Teacher model name")
    parser.add_argument("--index-path", default="./data/legal_faiss.index",
                        help="FAISS index path")
    parser.add_argument("--docs-path", default="./data/documents.json",
                        help="Documents JSON path")
    
    args = parser.parse_args()
    
    print("="*50)
    print("Initializing Agentic Data Synthesizer...")
    print("="*50)
    
    teacher = OpenRouterModel(
        model_name=args.teacher_model,
        api_key=args.openrouter_key,
        referer="https://github.com/anomalyco/opencode",
        title="AgenticDataGen"
    )
    
    print("Loading retriever...")
    from src.retrieval.retriever import TwoStageRetriever, FAISSRetriever
    from sentence_transformers import SentenceTransformer
    
    embedding_model = SentenceTransformer("/home/y/project/hgmodels/bge-small-zh-v1.5")
    
    class EmbedWrapper:
        def __init__(self, model):
            self.model = model
        def encode(self, texts):
            return self.model.encode(texts, convert_to_numpy=True)
    
    retriever = FAISSRetriever(embedding_model=EmbedWrapper(embedding_model))
    
    try:
        retriever.load_index(args.index_path)
        import json
        with open(args.docs_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            retriever.documents = data.get("documents", [])
            retriever.metadata = data.get("metadata", [])
        print(f"Loaded FAISS index with {len(retriever.documents)} docs")
    except Exception as e:
        print(f"Warning: Could not load index: {e}")
        print("Will use template-based observations")
    
    synthesizer = AgenticDataSynthesizer(
        teacher_model=teacher,
        retriever=retriever,
        max_iterations=args.max_iterations,
        negative_ratio=args.negative_ratio
    )
    
    print(f"\nLoading raw data...")
    raw_data = load_raw_data(args.lora_csv, args.num_samples)
    print(f"Loaded {len(raw_data)} raw samples")
    
    print(f"\nSynthesizing {len(raw_data)} samples...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    with open(args.output, 'w', encoding='utf-8') as f:
        for i, item in enumerate(raw_data):
            if (i + 1) % 10 == 0:
                print(f"Progress: {i+1}/{len(raw_data)}")
            
            question = item.get('question') or item.get('title', '')
            if not question:
                continue
            
            try:
                result = synthesizer.synthesize(question)
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"Error processing sample {i}: {e}")
                continue
    
    print(f"\n{'='*50}")
    print(f"Done! Output saved to: {args.output}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
