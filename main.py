import torch
from modelscope import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

def init_model():
    print("正在下载/加载模型 (Qwen2.5-1.5B-Instruct)...")
    # 使用 ModelScope 下载，适合国内网络环境
    # 1.5B 版本适合调试，如果显存足够（>16GB），建议改为 'qwen/Qwen2.5-7B-Instruct' 以获得更好效果
    model_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')
    
    print(f"模型路径: {model_dir}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    return tokenizer, model

def analyze_emotion(text, tokenizer, model):
    # 定义系统提示词 (System Prompt) - 这是捕捉“字里行间”的关键
    system_prompt = """你是一位精通心理学和语言学的情感分析专家。
你的任务是深入分析用户输入的文本，捕捉其中微妙的情感倾向、语气和隐含意图。

请按以下JSON格式输出分析结果：
{
    "label": "情感标签 (如：焦虑、讽刺、欣慰、绝望、平静、愤怒等)",
    "confidence": "置信度 (0-1)",
    "analysis": "简短分析，解释为什么判断为该情感，指出关键词或语境线索"
}

注意：
1. 不要只看表面文字，要理解反语、双关语和潜台词。
2. 输出必须是严格的 JSON 格式，不要包含Markdown代码块或其他废话。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.3,  # 低温度以保证输出稳定
        top_p=0.9
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

if __name__ == "__main__":
    try:
        tokenizer, model = init_model()
        print("\n=== 情感分析大模型 (输入 'quit' 退出) ===")
        
        while True:
            text = input("\n请输入需要分析的文字: ")
            if text.lower() in ['quit', 'exit']:
                break
            
            if not text.strip():
                continue
                
            print("正在分析中...")
            result = analyze_emotion(text, tokenizer, model)
            print("-" * 30)
            print(result)
            print("-" * 30)
            
    except ImportError:
        print("请先安装依赖: pip install -r requirements.txt")
    except Exception as e:
        print(f"发生错误: {e}")
