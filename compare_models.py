import torch
import gc
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from peft import PeftModel
# ================= ⚙️ 配置区域 =================
# 1. 原始底座模型路径 (1.5B)
BASE_MODEL_PATH = "./Models/Qwen/Qwen2.5-Coder-1.5B-Instruct"

# 2. 你的全量微调 Checkpoint 路径
# 注意：全量微调的 checkpoint 包含完整权重，直接加载即可
CHECKPOINT_PATH_FULL = "./output/qwen2.5_full_sft/checkpoint-2000"
CHECKPOINT_PATH_LORA = "./sft_cot_model/checkpoint-epoch-0-step-3200"
# 3. 测试问题
TEST_PROMPT = "create a java program that takes in 3 integers from the user and outputs the maximum number among them."


# ================= 🛠️ 流式推理函数 =================
def stream_inference(model, tokenizer, prompt, title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

    # 1. 构建输入
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 2. 设置流式输出器
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # 3. 配置生成参数
    generation_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=2048,
        temperature=0.7,
        do_sample=True  # 开启采样，让回复更自然
    )

    # 4. 在新线程中启动生成 (防止阻塞主线程打印)
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # 5. 主线程实时打印输出
    print(f"🤖 回复: ", end="", flush=True)
    generated_text = ""
    for new_text in streamer:
        print(new_text, end="", flush=True)
        generated_text += new_text
    print("\n")

    return generated_text


def clear_gpu():
    """清理显存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()
    print("🧹 显存已清理\n")


# ================= 🚀 主程序 =================
if __name__ == "__main__":
    print(f"❓ 问题: {TEST_PROMPT}\n")

    # 全局使用 bfloat16 (配合 1.5B 模型)
    dtype = torch.bfloat16

    # -------------------------------------------
    # 第一步：原始模型 (Base)
    # -------------------------------------------
    print("⏳ [1/3] 正在加载原始模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    stream_inference(model, tokenizer, TEST_PROMPT, "原始模型 (Base)")

    # 销毁模型释放显存
    del model
    clear_gpu()

    # -------------------------------------------
    # 第二步：全量微调模型 (Checkpoint)
    # -------------------------------------------
    print(f"⏳ [2/3] 正在加载全量微调 Checkpoint: {CHECKPOINT_PATH_FULL}...")

    # 全量微调的 Checkpoint 就是一个独立的完整模型
    # 我们直接从 checkpoint 目录加载
    model = AutoModelForCausalLM.from_pretrained(
        CHECKPOINT_PATH_FULL,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    stream_inference(model, tokenizer, TEST_PROMPT, "全量微调模型 (SFT)")



    # 销毁模型释放显存
    del model
    clear_gpu()
    # -------------------------------------------
    # 第三步：LORA微调模型 (Checkpoint)
    # -------------------------------------------
    print(f"⏳ [3/3] 正在加载LOR微调 Checkpoint: {CHECKPOINT_PATH_LORA}...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        # CHECKPOINT_PATH_LORA,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    finetuned_model=PeftModel.from_pretrained(model, CHECKPOINT_PATH_LORA)
    stream_inference(finetuned_model, tokenizer, TEST_PROMPT, "LORA微调模型 (SFT)")
    print("✅ 对比结束")