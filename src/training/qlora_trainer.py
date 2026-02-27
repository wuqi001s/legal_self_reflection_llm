import os
import torch
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class TrainingConfig:
    """QLoRA training configuration"""
    model_name: str = "Qwen/Qwen2-0.5B"
    output_dir: str = "./legal_qlora_model"
    train_file: str = "./data/train.jsonl"
    val_file: Optional[str] = None
    
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    max_seq_length: int = 1024
    
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    warmup_steps: int = 100
    
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True


class LegalQLoRATrainer:
    """QLoRA fine-tuning for legal domain"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup(self):
        """Initialize model and tokenizer with 4-bit quantization"""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        
        quantization_config = None
        if self.config.use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True
        )
        
        if self.config.model_name.endswith("-8B"):
            self.config.per_device_train_batch_size = 1
            self.config.gradient_accumulation_steps = 8
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if not self.config.use_4bit:
            self.model = self.model.half()
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_dataset(self):
        """Prepare training dataset"""
        from datasets import Dataset
        import pandas as pd
        
        data = []
        train_file = Path(self.config.train_file)
        if train_file.exists():
            import json
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
        
        def format_instruction(item):
            question = item.get("question", "")
            answer = item.get("answer", "")
            
            react = item.get("react_trajectory", [])
            irac = item.get("irac_reasoning", {})
            
            system_prompt = "你是一个专业的法律助手，请根据法律知识回答问题。"
            
            user_prompt = f"问题: {question}\n"
            
            if irac:
                user_prompt += f"\nIRAC分析:\n"
                user_prompt += f"- 问题: {irac.get('issue', '')}\n"
                user_prompt += f"- 规则: {irac.get('rule', '')}\n"
                user_prompt += f"- 适用: {irac.get('application', '')}\n"
                user_prompt += f"- 结论: {irac.get('conclusion', '')}\n"
            
            user_prompt += f"\n请给出回答:"
            
            return {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": answer}
                ]
            }
        
        formatted_data = [format_instruction(item) for item in data]
        return Dataset.from_list(formatted_data)
    
    def train(self):
        """Execute QLoRA training"""
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from trl import SFTTrainer
        
        train_data = self.prepare_dataset()
        
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            num_train_epochs=self.config.num_train_epochs,
            max_steps=-1,
            fp16=self.config.fp16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            warmup_steps=self.config.warmup_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            eval_strategy="no" if not self.config.val_file else "steps",
            save_strategy="steps",
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            data_collator=data_collator,
        )
        
        self.trainer.train()
        
    def save_model(self):
        """Save fine-tuned model"""
        self.model.save_pretrained(self.config.output_dir)
        self.tokenizer.save_pretrained(self.config.output_dir)
        
    def merge_and_save(self, output_path: str = "./legal_model_merged"):
        """Merge LoRA weights and save"""
        from peft import PeftModel
        
        merged_model = self.model.merge_and_unload()
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
