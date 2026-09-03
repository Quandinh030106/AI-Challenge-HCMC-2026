# ==============================================================================
# AIC 2026 - QLoRA FINE-TUNING SCRIPT FOR MODEL A: NLP QUERY PARSER (Qwen2.5-7B)
# ==============================================================================
import os
import yaml
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class NLPDataset(torch.utils.data.Dataset):
    """Custom Dataset for Qwen2.5-7B Instruction Tuning."""
    def __init__(self, data_path: str, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        messages = item.get("messages", [])
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        enc = self.tokenizer(text, max_length=self.max_length, truncation=True, padding=False, return_tensors="pt")
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def train_model_a(config_path: str = "configs/finetune_config.yaml"):
    """
    Executes 4-bit QLoRA instruction fine-tuning on Qwen2.5-7B-Instruct.
    Teaches the model:
    1. Anti-bias strict JSON schema formatting
    2. Contextual Phrase Translation (CPT) without over-abstraction
    3. Dynamic M-Entity preservation
    """
    print("=" * 80)
    print("[INFO] AIC 2026 - STARTING QLoRA TRAINING FOR MODEL A (NLP PARSER)")
    print("=" * 80)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    nlp_cfg = cfg.get("nlp_finetune", {})
    model_id = nlp_cfg.get("base_model", "Qwen/Qwen2.5-7B-Instruct")
    output_dir = nlp_cfg.get("output_dir", "/kaggle/working/models/nlp_lora_adapter")
    data_path = nlp_cfg.get("data_path", "data/finetune/nlp_train.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(data_path):
        from src.finetune.data_generator import FinetuneDataGenerator
        print(f"[INFO] Dataset '{data_path}' missing. Generating seed dataset...")
        FinetuneDataGenerator.create_sample_nlp_data(data_path)

    # 1. 4-bit Quantization Configuration (NF4 with double quant)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[INFO] Loading base model '{model_id}' in 4-bit NF4 mode...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config if torch.cuda.is_available() else None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 2. LoRA Adapter Configuration
    lora_config = LoraConfig(
        r=nlp_cfg.get("lora_r", 16),
        lora_alpha=nlp_cfg.get("lora_alpha", 32),
        lora_dropout=nlp_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=nlp_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load & Format Dataset
    print(f"[INFO] Loading dataset from '{data_path}'...")
    train_dataset = NLPDataset(data_path, tokenizer, max_length=nlp_cfg.get("max_seq_length", 1024))
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=nlp_cfg.get("batch_size", 2),
        gradient_accumulation_steps=nlp_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=nlp_cfg.get("learning_rate", 2e-4),
        num_train_epochs=nlp_cfg.get("num_epochs", 3),
        warmup_ratio=nlp_cfg.get("warmup_ratio", 0.05),
        lr_scheduler_type=nlp_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=nlp_cfg.get("logging_steps", 10),
        save_strategy=nlp_cfg.get("save_strategy", "epoch"),
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none"
    )

    # 5. Standard HuggingFace Trainer (Compatible with all transformers versions)
    try:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            processing_class=tokenizer,
            args=training_args
        )
    except TypeError:
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            data_collator=data_collator,
            args=training_args
        )

    print("[INFO] Starting LoRA Fine-Tuning...")
    trainer.train()

    print(f"[INFO] Saving trained LoRA Adapter to '{output_dir}'...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Verify saved files
    if os.path.exists(output_dir):
        saved_files = os.listdir(output_dir)
        print(f"[INFO] Verified saved adapter files in '{output_dir}': {saved_files}")

    print("=" * 80)
    print(f"[INFO] MODEL A (NLP PARSER) QLoRA TRAINING COMPLETE! Saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    train_model_a()
