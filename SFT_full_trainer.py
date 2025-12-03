import os
import torch
import json
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# ================= 配置路径 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 建议：如果是16G显卡，请将此处改为 1.5B 模型的路径
MODEL_PATH = os.path.join(BASE_DIR, "Models/Qwen/Qwen2.5-Coder-1.5B-Instruct")
DATASET_PATH = os.path.join(BASE_DIR, "Datasets/O1-OPEN/OpenO1-SFT")
OUTPUT_DIR = os.path.join(BASE_DIR, "output/qwen2.5_full_sft")


# ================= 数据处理函数 =================
def process_func(example, tokenizer):
    """
    将数据集格式转换为 Qwen2.5 的 ChatML 格式，并处理掩码（Masking）。
    只计算 Assistant 回复部分的 Loss。
    假设数据集字段为: 'instruction' (或 'prompt') 和 'output' (或 'response')
    """
    MAX_LENGTH = 2048  # 根据显存大小调整，16G建议设为 1024 或 2048

    # 1. 构建符合 Qwen 格式的消息列表
    instruction = example.get('instruction', example.get('prompt', ''))
    response = example.get('output', example.get('response', ''))

    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]

    # 2. 使用 tokenizer 的 chat 模板处理
    # text 格式: <|im_start|>system...<|im_end|><|im_start|>user...<|im_end|><|im_start|>assistant...<|im_end|>
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    # 3. Tokenize 整个对话
    input_ids = tokenizer(text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)["input_ids"][0]
    labels = input_ids.clone()

    # 4. 制作 Labels Mask (将 System 和 User 的部分设为 -100，不计算 Loss)
    # Qwen2.5 的 assistant header 是 "<|im_start|>assistant\n"
    # 我们需要找到 assistant 开始生成回复的位置

    # 这里使用一种简化的 Mask 逻辑：找到最后一个 "<|im_start|>assistant"
    # 注意：具体 token id 需要根据 tokenizer 实际编码结果动态查找，或者再次 tokenize user prompt 计算长度

    # 方法：单独 tokenize (system + user) 的部分来确定长度
    user_messages = messages[:-1]  # 去掉 assistant 的回复
    user_text = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
    user_input_ids = tokenizer(user_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)["input_ids"][0]

    # 将 user 输入部分的 label 设为 -100
    if len(user_input_ids) < len(input_ids):
        labels[:len(user_input_ids)] = -100
    else:
        # 如果 user 输入过长被截断，全设为 -100 (这通常不应该发生，除非 max_length 太小)
        labels[:] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),  # Qwen 默认全 1 即可，padding 在 collator 处理
        "labels": labels
    }


# ================= 主程序 =================
if __name__ == "__main__":
    # 1. 加载 Tokenizer
    print(f"Loading tokenizer from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    # Qwen2.5 默认没有 pad_token，通常使用 eos_token 或者专门指定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 加载模型
    print(f"Loading model from {MODEL_PATH}...")
    # 全量微调需要加载完整模型，建议使用 bf16 节省显存
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # 开启梯度检查点 (关键：节省大量显存)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # 3. 加载和处理数据集
    print(f"Loading dataset from {DATASET_PATH}...")
    # 如果是 json/jsonl 文件
    if os.path.isdir(DATASET_PATH):
        # 假设目录下有 .json 或 .jsonl 文件，或者直接是 huggingface 格式缓存
        try:
            ds = load_dataset(DATASET_PATH, split="train")
        except:
            # 尝试加载 json
            data_files = [os.path.join(DATASET_PATH, f) for f in os.listdir(DATASET_PATH) if
                          f.endswith('.json') or f.endswith('.jsonl')]
            ds = load_dataset('json', data_files=data_files, split='train')
    else:
        # 如果指向的是具体文件
        ds = load_dataset('json', data_files=DATASET_PATH, split='train')

    print("Processing dataset...")
    tokenized_ds = ds.map(
        lambda x: process_func(x, tokenizer),
        remove_columns=ds.column_names,  # 移除原始文本字段
        num_proc=4  # 多进程处理
    )

    # 4. 设置训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,  # 16G 显存全量微调必须设小
        # gradient_checkpointing=True,         # <--- 【新增】在这里开启梯度检查点
        # gradient_checkpointing_kwargs={"use_reentrant": False}, # DDP 常用设置，防止报错
        gradient_accumulation_steps=16,  # 通过累加模拟大 batch size (1*16 = 16)
        logging_steps=10,
        num_train_epochs=1,
        save_steps=100,  # [需求] 每 100 步保存一次 Checkpoint
        save_total_limit=3,  # 最多保留 3 个 Checkpoint，防止硬盘撑爆
        learning_rate=1e-5,  # 全量微调 LR 要小，1e-5 或 5e-6
        weight_decay=0.01,
        warmup_ratio=0.03,
        bf16=True,  # 强烈建议使用 bf16
        dataloader_num_workers=2,
        report_to="none",  # 不上传 wandb，可按需开启
        optim="adamw_bnb_8bit",  # [关键] 使用 8-bit AdamW 优化器节省显存
    )

    # 5. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )

    # 6. 开始训练
    print("Starting training...")
    trainer.train()

    # 7. 保存最终模型
    print(f"Saving final model to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)