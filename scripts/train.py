#!/usr/bin/env python3
"""
Training Script
Fine-tune legal LLM with QLoRA
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.qlora_trainer import LegalQLoRATrainer, TrainingConfig


def main():
    parser = argparse.ArgumentParser(description="Legal LLM QLoRA Training")
    parser.add_argument("--model-name", default="/home/y/project/hgmodels/Qwen3-0.6B", 
                        help="Base model path")
    parser.add_argument("--train-file", default="./data/train.jsonl", help="Training data file")
    parser.add_argument("--output-dir", default="./models/legal_qlora", help="Output directory")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--batch-size", type=int, default=2, help="Per device batch size")
    parser.add_argument("--max-seq-length", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--merge-only", action="store_true", help="Only merge and save model")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        model_name=args.model_name,
        train_file=args.train_file,
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        lora_r=args.lora_r,
        per_device_train_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
    )
    
    trainer = LegalQLoRATrainer(config)
    
    if not args.merge_only:
        trainer.setup()
        print("Starting training...")
        trainer.train()
        print("Training completed!")
        trainer.save_model()
    else:
        trainer.setup()
        
    print(f"Merging LoRA weights to {args.output_dir}_merged...")
    trainer.merge_and_save(f"{args.output_dir}_merged")
    print("Done!")


if __name__ == "__main__":
    main()
