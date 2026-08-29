# ==============================================================================
# AIC 2026 - QLoRA FINE-TUNING SCRIPT FOR MODEL B: VLM VISUAL VERIFIER (Qwen2.5-VL-7B)
# ==============================================================================
import os
import yaml
import json
import torch
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class VLMDataset(torch.utils.data.Dataset):
    """Custom Multi-Modal Dataset for Qwen2.5-VL Visual Grounding & CoT Scoring."""
    def __init__(self, data_path: str, processor, max_pixels: int = 1048576):
        self.processor = processor
        self.max_pixels = max_pixels
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
        img_path = item.get("image_path", "")
        query = item.get("query_text", "")
        cot_reasoning = item.get("cot_reasoning", f"Điểm: {item.get('score', 50.0)}")

        # Create dummy image if path doesn't exist for synthetic dry-runs
        if os.path.exists(img_path):
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (448, 448), color=(random_r := 100, 150, 200))

        image.thumbnail((896, 896))

        prompt_text = (
            f"Nhiệm vụ: Quan sát khung ảnh HD và đánh giá mức độ trùng khớp với mô tả sau:\n"
            f"'{query}'\n"
            f"Yêu cầu: Thực hiện suy luận từng bước (CoT). Nếu thiếu bất kỳ chi tiết quan trọng nào, trừ điểm dưới 30. "
            f"Chỉ chấm trên 80 nếu xuất hiện đủ 100% tất cả thực thể."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": cot_reasoning}]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = self.processor(text=[text], images=[image], padding="max_length", max_length=512, return_tensors="pt")
        
        # Flatten batch dimension
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": inputs["pixel_values"].squeeze(0) if "pixel_values" in inputs else torch.tensor([]),
            "image_grid_thw": inputs["image_grid_thw"].squeeze(0) if "image_grid_thw" in inputs else torch.tensor([]),
            "labels": inputs["input_ids"].squeeze(0)
        }

def train_model_b(config_path: str = "configs/finetune_config.yaml"):
    """
    Executes 4-bit QLoRA multi-modal fine-tuning on Qwen2.5-VL-7B-Instruct.
    Teaches the model:
    1. Chain-of-Thought (CoT) visual inspection
    2. Strict penalty for partial visual matches (<30 pts)
    3. Fine-grained multi-entity co-occurrence scoring (>80 pts)
    """
    print("=" * 80)
    print("[INFO] AIC 2026 - STARTING QLoRA TRAINING FOR MODEL B (VLM VERIFIER)")
    print("=" * 80)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    vlm_cfg = cfg.get("vlm_finetune", {})
    model_id = vlm_cfg.get("base_model", "Qwen/Qwen2.5-VL-7B-Instruct")
    output_dir = vlm_cfg.get("output_dir", "/kaggle/working/models/vlm_lora_adapter")
    data_path = vlm_cfg.get("data_path", "data/finetune/vlm_train.jsonl")

    if not os.path.exists(data_path):
        from src.finetune.data_generator import FinetuneDataGenerator
        print(f"[INFO] Dataset '{data_path}' missing. Generating seed VLM dataset...")
        FinetuneDataGenerator.create_sample_vlm_data(data_path)

    # 1. 4-bit Quantization Configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    print(f"[INFO] Loading base multi-modal model '{model_id}' in 4-bit mode...")
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    model = prepare_model_for_kbit_training(model)

    # 2. LoRA Adapter Configuration
    lora_config = LoraConfig(
        r=vlm_cfg.get("lora_r", 16),
        lora_alpha=vlm_cfg.get("lora_alpha", 32),
        lora_dropout=vlm_cfg.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=vlm_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Multi-modal Dataset
    train_dataset = VLMDataset(data_path, processor, max_pixels=vlm_cfg.get("max_pixels", 1048576))

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=vlm_cfg.get("batch_size", 1),
        gradient_accumulation_steps=vlm_cfg.get("gradient_accumulation_steps", 8),
        learning_rate=vlm_cfg.get("learning_rate", 1e-4),
        num_train_epochs=vlm_cfg.get("num_epochs", 3),
        logging_steps=vlm_cfg.get("logging_steps", 5),
        save_strategy=vlm_cfg.get("save_strategy", "epoch"),
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        report_to="none"
    )

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args
    )

    print("[INFO] Starting Multi-Modal VLM LoRA Fine-Tuning...")
    trainer.train()

    print(f"[INFO] Saving trained VLM LoRA Adapter to '{output_dir}'...")
    trainer.model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print("=" * 80)
    print(f"[INFO] MODEL B (VLM VERIFIER) QLoRA TRAINING COMPLETE! Saved to: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    train_model_b()
