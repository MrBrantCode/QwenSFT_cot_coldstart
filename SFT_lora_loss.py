import os
import torch
import glob
import shutil
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
# 🟢 引入 load_from_disk 用于读取缓存
from datasets import load_dataset, load_from_disk
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm import tqdm

# ================= 环境变量配置 =================
# 禁用 NCCL P2P/IB 防止 3090/4090 报 -11 错误
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ================= 全局配置 =================
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(gradient_accumulation_steps=4, kwargs_handlers=[ddp_kwargs])

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "Models/Qwen/Qwen2.5-Coder-1.5B-Instruct")
DATASET_PATH = os.path.join(BASE_DIR, "Datasets/O1-OPEN/OpenO1-SFT")

# 🟢 数据缓存路径 (处理后的数据将保存在这里)
SAVE_PATH = os.path.join(BASE_DIR, "output/qwen2.5_lora_sft")

MAX_LENGTH = 1024
BATCH_SIZE = 2
LR = 5e-5
NUM_EPOCHS = 1
SAVE_STEPS = 100

# ================= 功能函数 =================

def load_model_and_tokenizer():
    if accelerator.is_main_process:
        print(f"正在从 {MODEL_PATH} 加载模型...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map={"": accelerator.local_process_index},
        trust_remote_code=True
    )
    
    model = prepare_model_for_kbit_training(model)
    # 强制禁用 use_reentrant 防止 DDP 崩溃
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

    model = get_peft_model(model, peft_config)

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    return model, tokenizer


def process_data(samples, tokenizer):
    inputs = []
    labels = []
    for instruction, output in zip(samples['instruction'], samples['output']):
        user_messages = [{"role": "user", "content": instruction}]
        full_messages = [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
        
        prompt_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
        
        encoded = tokenizer(full_text, max_length=MAX_LENGTH, padding="max_length", truncation=True, return_tensors="pt")
        input_ids = encoded.input_ids[0]
        attention_mask = encoded.attention_mask[0]
        
        label = input_ids.clone()
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
        prompt_len = len(prompt_ids)
        
        if prompt_len < MAX_LENGTH:
            label[:prompt_len] = -100
        else:
            label[:] = -100
            
        label[attention_mask == 0] = -100
        inputs.append(input_ids)
        labels.append(label)

    return {
        "input_ids": torch.stack(inputs),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(inputs).ne(tokenizer.pad_token_id)
    }

# ================= 主程序 =================
def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ 模型路径不存在: {MODEL_PATH}")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"❌ 数据集路径不存在: {DATASET_PATH}")

    model, tokenizer = load_model_and_tokenizer()

    # ================= 🟢 [修改] 智能数据加载逻辑 =================
    train_dataset = None
    
    # 1. 检查是否存在缓存
    if os.path.exists(PROCESSED_DATA_CACHE):
        if accelerator.is_main_process:
            print(f"🚀 发现已处理的数据缓存: {PROCESSED_DATA_CACHE}，正在直接加载...")
        # 直接从磁盘加载处理好的数据，速度极快
        train_dataset = load_from_disk(PROCESSED_DATA_CACHE)
    else:
        if accelerator.is_main_process:
            print(f"🐢 未发现缓存，正在读取原始数据并进行处理（这需要一点时间）...")
            
        # 2. 如果没有缓存，则加载原始数据进行处理
        data_files = glob.glob(os.path.join(DATASET_PATH, "**/*.parquet"), recursive=True)
        if not data_files:
            data_files = glob.glob(os.path.join(DATASET_PATH, "**/*.jsonl"), recursive=True)
        
        if not data_files:
            raise FileNotFoundError("❌ 未找到数据文件")

        if data_files[0].endswith(".parquet"):
            raw_dataset = load_dataset("parquet", data_files=data_files, split="train")
        else:
            raw_dataset = load_dataset("json", data_files=data_files, split="train")

        # 只在主进程进行 map 处理
        with accelerator.main_process_first():
            train_dataset = raw_dataset.map(
                lambda x: process_data(x, tokenizer),
                batched=True,
                batch_size=100,
                remove_columns=raw_dataset.column_names,
                desc="正在处理数据并 Tokenize"
            )
            
            # 3. 处理完成后，保存到磁盘！
            if accelerator.is_main_process:
                print(f"💾 数据处理完成，正在保存到缓存: {PROCESSED_DATA_CACHE} ...")
                train_dataset.save_to_disk(PROCESSED_DATA_CACHE)
        
        # 等待主进程保存完毕
        accelerator.wait_for_everyone()
        
        # 为了保证多进程数据一致性，建议所有进程都重新从磁盘 load 一遍
        train_dataset = load_from_disk(PROCESSED_DATA_CACHE)

    # 设定格式
    train_dataset.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])
    
    if accelerator.is_main_process:
        print(f"✅ 数据加载完毕，样本数量: {len(train_dataset)}")
    # ============================================================
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda x: {k: torch.stack([f[k] for f in x]) for k in x[0]},
        num_workers=0,
        pin_memory=True
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    num_training_steps = len(train_dataloader) * NUM_EPOCHS // accelerator.gradient_accumulation_steps
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_training_steps)

    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    
    model = accelerator.prepare(model)

    model.train()

    if accelerator.is_main_process:
        print("\n🚀 开始训练...")

    for epoch in range(NUM_EPOCHS):
        progress_bar = tqdm(train_dataloader, disable=not accelerator.is_main_process, desc=f"Epoch {epoch + 1}")

        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.is_main_process and step % 10 == 0:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            if (step + 1) % SAVE_STEPS == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    checkpoint_dir = os.path.join(SAVE_PATH, f"checkpoint-epoch-{epoch}-step-{step + 1}")
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    print(f"\n💾 保存检查点: {checkpoint_dir}")
                    accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
                    tokenizer.save_pretrained(checkpoint_dir)
                accelerator.wait_for_everyone()
                
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        epoch_dir = os.path.join(SAVE_PATH, f"checkpoint-epoch-{epoch}-end")
        accelerator.unwrap_model(model).save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"\n✅ 训练完成，已保存至 {epoch_dir}")

if __name__ == "__main__":
    main()