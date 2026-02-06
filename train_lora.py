import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
from modelscope import snapshot_download
import json

# ================= 配置参数 =================
# ModelScope 模型 ID (通常是小写的 qwen/...)
MS_MODEL_ID = "qwen/Qwen2.5-1.5B-Instruct" 
DATA_PATH = "data/train.jsonl"          # 训练数据路径
OUTPUT_DIR = "output_checkpoints"       # 训练输出目录
MAX_LENGTH = 512                        # 最大上下文长度
EPOCHS = 3                              # 训练轮数
BATCH_SIZE = 4                          # 批次大小 (根据显存调整)
LEARNING_RATE = 1e-4                    # 学习率

def get_model_path():
    """
    优先使用 ModelScope 下载模型到本地，解决网络和命名问题
    """
    print(f"正在通过 ModelScope 下载/加载模型: {MS_MODEL_ID} ...")
    try:
        model_dir = snapshot_download(MS_MODEL_ID, cache_dir='./models')
        print(f"模型已准备好，路径: {model_dir}")
        return model_dir
    except Exception as e:
        print(f"ModelScope 下载失败: {e}")
        print("将尝试直接使用 HuggingFace ID (可能会失败)...")
        return "Qwen/Qwen2.5-1.5B-Instruct" # Fallback

def process_func(example, tokenizer):
    """
    将 ChatML 格式的数据转换为模型输入的 input_ids 和 labels
    """
    MAX_LENGTH = 512    
    input_ids, attention_mask, labels = [], [], []
    
    # Qwen 的 Chat 模板处理
    instruction = tokenizer.apply_chat_template(
        example['messages'][:-1], # 输入部分 (System + User)
        tokenize=False, 
        add_generation_prompt=True
    )
    response = example['messages'][-1]['content'] # 输出部分 (Assistant)
    
    # 编码
    instruction_ids = tokenizer.encode(instruction, add_special_tokens=False)
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    
    # 拼接: 输入 + 输出 + EOS
    input_ids = instruction_ids + response_ids + [tokenizer.eos_token_id]
    
    # Labels: 输入部分设为 -100 (不计算 Loss), 输出部分保留
    labels = [-100] * len(instruction_ids) + response_ids + [tokenizer.eos_token_id]
    
    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels
    }

def train():
    # 1. 获取模型路径 (本地绝对路径)
    model_path = get_model_path()
    
    print(f"正在加载 Tokenizer 和 模型 from: {model_path}")
    
    # 2. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token # 确保 pad token 存在

    # 3. 加载模型 (半精度 fp16 加载，节省显存)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # 4. 配置 LoRA (参数微调的核心)
    # 这一步告诉模型：我们不改所有参数，只改其中一小部分 (A/B 矩阵)，实现高效微调
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        inference_mode=False, 
        r=8,            # LoRA 秩，越大参数越多
        lora_alpha=32,  # LoRA 缩放系数
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # 针对所有线性层进行微调效果最好
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # 打印将要训练多少参数

    # 5. 加载并处理数据
    print("正在加载和处理数据...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"找不到训练数据: {DATA_PATH}。请先运行 prepare_data.py")
        
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    
    # 映射数据处理函数
    tokenized_ds = dataset.map(
        lambda x: process_func(x, tokenizer), 
        remove_columns=dataset.column_names
    )

    # 6. 设置训练参数
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4, # 累积梯度，模拟大 batch size
        logging_steps=10,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        save_strategy="epoch",      # 每轮保存一次
        fp16=True,                  # 开启混合精度训练
        optim="adamw_torch",
        report_to="none"            # 不上传 wandb
    )

    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    print("开始训练 (Fine-tuning)... 这可能需要一段时间，取决于您的显卡性能。")
    trainer.train()
    
    # 8. 保存最终模型
    print(f"训练完成！正在保存 LoRA 权重到 {OUTPUT_DIR}/final_model ...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\n训练出错: {e}")
