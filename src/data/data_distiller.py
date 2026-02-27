import json
import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ReActTrajectory:
    """ReAct action trajectory for legal reasoning"""
    thought: str
    action: str
    action_input: str
    observation: str


@dataclass
class IRACReasoning:
    """IRAC legal reasoning structure"""
    issue: str
    rule: str
    application: str
    conclusion: str


class LegalDataDistiller:
    """
    Teacher LLM based data distillation pipeline
    Converts rough Q&A to high-quality instructions with ReAct + IRAC
    """
    
    def __init__(self, teacher_model=None, judge_model=None):
        self.teacher_model = teacher_model
        self.judge_model = judge_model
        
    def distill(self, raw_data: List[Dict], num_samples: int = 1000) -> List[Dict]:
        """Distill raw Q&A into high-quality training data"""
        distilled = []
        
        for item in raw_data[:num_samples]:
            result = self._distill_single(item)
            if result and self._judge_quality(result):
                distilled.append(result)
                
        return distilled
    
    def _distill_single(self, raw_item: Dict) -> Optional[Dict]:
        """Distill a single Q&A pair with ReAct and IRAC"""
        if not self.teacher_model:
            return self._generate_template_distillation(raw_item)
        
        prompt = self._build_distillation_prompt(raw_item)
        response = self.teacher_model.generate(prompt)
        
        try:
            return json.loads(response)
        except:
            return self._generate_template_distillation(raw_item)
    
    def _generate_template_distillation(self, raw_item: Dict) -> Dict:
        """Template-based distillation when model is not available"""
        question = raw_item.get("question") or raw_item.get("title", "")
        answer = raw_item.get("reply") or raw_item.get("answer", "")
        
        react_trajectory = [
            {
                "thought": "我需要分析这个法律问题",
                "action": "search_legal_database",
                "action_input": question,
                "observation": "找到相关法律条文"
            },
            {
                "thought": "我需要应用法律规则",
                "action": "analyze_legal_issue",
                "action_input": "根据IRAC方法分析",
                "observation": "确定法律依据和适用"
            }
        ]
        
        irac = {
            "issue": question,
            "rule": "相关法律条文内容",
            "application": "将法律规则应用于具体案件事实",
            "conclusion": answer
        }
        
        return {
            "question": question,
            "answer": answer,
            "react_trajectory": react_trajectory,
            "irac_reasoning": irac,
            "metadata": {
                "source": "distilled",
                "has_negative_sample": False
            }
        }
    
    def _build_distillation_prompt(self, raw_item: Dict) -> str:
        """Build prompt for Teacher LLM distillation"""
        return f"""请将以下粗糙的法律问答对重写为高质量的指令数据，融合ReAct动作轨迹和IRAC法律推演法则。

原始问答:
问题: {raw_item.get('question', '')}
答案: {raw_item.get('answer', '')}

请按以下JSON格式输出:
{{
    "question": "重写后的法律问题",
    "answer": "重写后的法律答案",
    "react_trajectory": [
        {{"thought": "思考", "action": "动作", "action_input": "输入", "observation": "观察"}}
    ],
    "irac_reasoning": {{
        "issue": "法律问题",
        "rule": "法律规则",
        "application": "法律适用",
        "conclusion": "结论"
    }}
}}"""
    
    def _judge_quality(self, item: Dict) -> bool:
        """Use LLM-as-a-Judge to filter low-quality data"""
        if not self.judge_model:
            return True
            
        prompt = self._build_judge_prompt(item)
        response = self.judge_model.generate(prompt)
        
        try:
            result = json.loads(response)
            return result.get("pass", True)
        except:
            return True
    
    def _build_judge_prompt(self, item: Dict) -> str:
        """Build prompt for judge model"""
        return f"""请评判以下法律问答数据的质量:

问题: {item.get('question', '')}
答案: {item.get('answer', '')}

请判断:
1. 答案是否包含法律依据
2. 推理是否符合逻辑
3. 是否有事实错误

请输出JSON格式:
{{"pass": true/false, "reason": "评判理由"}}"""


class AdversarialNegativeSampler:
    """Generate adversarial negative samples for safe refusal training"""
    
    def __init__(self):
        self.unsafe_patterns = [
            "如何杀人",
            "如何抢劫",
            "如何贩毒",
            "如何逃税",
            "如何伪造证据"
        ]
        
    def generate_negative_samples(self, base_data: List[Dict], ratio: float = 0.15) -> List[Dict]:
        """Generate 15% adversarial negative samples"""
        num_negatives = int(len(base_data) * ratio)
        negatives = []
        
        for _ in range(num_negatives):
            neg = self._generate_single_negative()
            negatives.append(neg)
            
        return negatives
    
    def _generate_single_negative(self) -> Dict:
        """Generate a single negative sample"""
        return {
            "question": random.choice(self.unsafe_patterns),
            "answer": "抱歉，我不能协助这类违法行为的请求。建议您通过合法途径解决问题。",
            "react_trajectory": [],
            "irac_reasoning": {
                "issue": "用户提出违法请求",
                "rule": "拒绝回答原则",
                "application": "识别为不当请求",
                "conclusion": "安全拒答"
            },
            "metadata": {
                "source": "adversarial_negative",
                "has_negative_sample": True,
                "should_refuse": True
            }
        }
    
    def merge_with_positives(self, positives: List[Dict], negatives: List[Dict]) -> List[Dict]:
        """Merge positive and negative samples"""
        merged = positives + negatives
        random.shuffle(merged)
        return merged
    
    def save_training_data(self, data: List[Dict], output_path: str):
        """Save training data to JSONL"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
