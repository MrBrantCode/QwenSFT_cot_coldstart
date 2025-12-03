import os
from modelscope import snapshot_download

# 配置保存路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR,  "Models")
DATASET_DIR = os.path.join(BASE_DIR, "Datasets")

print(f"🚀 准备下载模型到: {MODEL_DIR}")
print(f"🚀 准备下载数据集到: {DATASET_DIR}")

# 1. 下载 Qwen2.5-Coder-7B-Instruct
# snapshot_download 会保持模型原本的目录结构，例如 Models/Qwen/Qwen2.5...
try:
    model_path = snapshot_download(
        'Qwen/Qwen2.5-Coder-1.5B-Instruct',
        cache_dir=MODEL_DIR
    )
    print(f"✅ 模型下载完成，路径: {model_path}")
except Exception as e:
    print(f"❌ 模型下载失败: {e}")

# 2. 下载 OpenO1-SFT 数据集
# 注意：魔塔社区通常会镜像 HuggingFace 的数据集，ID 通常一致
try:
    dataset_path = snapshot_download(
        'O1-OPEN/OpenO1-SFT',
        cache_dir=DATASET_DIR,
        repo_type='dataset' # 显式指定是数据集
    )
    print(f"✅ 数据集下载完成，路径: {dataset_path}")
except Exception as e:
    print(f"❌ 数据集下载失败: {e}")

print("\n🎉 所有资源下载完毕！")