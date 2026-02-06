import json
import os
from modelscope.msdatasets import MsDataset

def convert_to_qwen_format(dataset, output_file):
    print(f"正在处理数据并写入 {output_file} ...")
    
    # 定义标签映射 (FewCLUE ewepr 的标签通常是英文或数字，这里做个示例映射，具体视下载数据而定)
    # ewepr 标签: happiness, sadness, anger, fear, surprise, disgust
    label_map = {
        "happiness": "快乐",
        "sadness": "悲伤",
        "anger": "愤怒",
        "fear": "恐惧",
        "surprise": "惊讶",
        "disgust": "厌恶",
        "neutral": "平静/中性"
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in dataset:
            # 获取原始文本和标签
            # 不同数据集字段名可能不同，FewCLUE通常是 'sentence' 和 'label'
            text = item.get('sentence', '')
            label_raw = item.get('label', '')
            
            if not text:
                continue

            label_cn = label_map.get(label_raw, label_raw)

            # 构造 Qwen 的微调数据格式 (ChatML)
            data_point = {
                "type": "chatml",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一位情感分析专家。请分析以下文本的情感倾向，并输出最准确的情感标签。"
                    },
                    {
                        "role": "user",
                        "content": text
                    },
                    {
                        "role": "assistant",
                        "content": f"{{\"label\": \"{label_cn}\", \"analysis\": \"基于文本内容的直接情感判断。\"}}" 
                    }
                ],
                "source": "ewepr_emotion"
            }
            
            f.write(json.dumps(data_point, ensure_ascii=False) + '\n')
    
    print(f"完成！共写入 {len(dataset)} 条数据。")

def main():
    # 创建 data 目录
    os.makedirs('data', exist_ok=True)

    import pandas as pd
    import requests
    import io

    print("正在尝试下载真实中文情感数据...")
    
    # 策略 1: 从 GitHub CDN (jsDelivr) 下载 waimai_10k (外卖评论，约 12k 条，二分类)
    # 这是一个非常稳定且高质量的中文情感数据集
    url = "https://cdn.jsdelivr.net/gh/SophonPlus/ChineseNlpCorpus@master/datasets/waimai_10k/waimai_10k.csv"
    print(f"尝试从 CDN 下载 waimai_10k: {url} ...")
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            print("下载成功！正在解析...")
            # waimai_10k.csv 格式: label, review
            df = pd.read_csv(io.StringIO(response.content.decode('utf-8')))
            
            # 转换 DataFrame 为列表字典格式，方便统一处理
            # label: 0 (差评), 1 (好评)
            dataset_list = []
            for _, row in df.iterrows():
                dataset_list.append({
                    'review': row['review'],
                    'label': row['label']
                })
            
            # 划分训练/验证 (前 11000 训练, 后 987 验证)
            ds_train_list = dataset_list[:11000]
            ds_dev_list = dataset_list[11000:]
            
            print(f"获取数据成功: 训练集 {len(ds_train_list)} 条, 验证集 {len(ds_dev_list)} 条")
            
            # 定义标签映射
            label_map_int = {
                0: "失望/愤怒", # 负面
                1: "满意/愉快"  # 正面
            }
            
        else:
            raise Exception(f"HTTP {response.status_code}")

    except Exception as e:
        print(f"CDN 下载失败 ({e})。尝试从 ModelScope 加载 'jd' 数据集...")
        try:
            # 策略 2: ModelScope 'jd' (京东评论)
            ds_train_ms = MsDataset.load('jd', split='train', trust_remote_code=True)
            ds_dev_ms = MsDataset.load('jd', split='validation', trust_remote_code=True) # jd 可能没有 validation
            
            # 如果没有 validation，手动切分
            # 这里简化处理，假设能 load 到
            
            # ModelScope dataset 转 list
            ds_train_list = list(ds_train_ms)
            ds_dev_list = list(ds_dev_ms) if 'ds_dev_ms' in locals() else []
            
            # 京东数据通常也是 0/1 (或者类似)
            label_map_int = {
                0: "失望/愤怒",
                1: "满意/愉快"
            }
             
        except Exception as e2:
            print(f"ModelScope 加载也失败 ({e2})。")
            print("请手动下载 'waimai_10k.csv' 并放置在目录下，或者检查网络连接。")
            return

    # 统一转换逻辑
    print("正在转换为 Qwen ChatML 格式...")
    
    def write_jsonl(data_list, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in data_list:
                text = item.get('review') or item.get('sentence')
                label_code = item.get('label')
                
                if not text: continue
                
                # 映射标签
                try:
                    label_cn = label_map_int.get(int(label_code), str(label_code))
                except:
                    label_cn = str(label_code)

                data_point = {
                    "type": "chatml",
                    "messages": [
                        {
                            "role": "system",
                            "content": "你是一位情感分析专家。请分析以下文本的情感倾向，并输出最准确的情感标签。"
                        },
                        {
                            "role": "user",
                            "content": text
                        },
                        {
                            "role": "assistant",
                            "content": f"{{\"label\": \"{label_cn}\", \"analysis\": \"基于文本的情感倾向判断。\"}}" 
                        }
                    ],
                    "source": "waimai_10k/jd"
                }
                f.write(json.dumps(data_point, ensure_ascii=False) + '\n')

    write_jsonl(ds_train_list, 'data/train.jsonl')
    write_jsonl(ds_dev_list, 'data/dev.jsonl')

    print("\n数据准备完毕！")
    print("训练集: data/train.jsonl")
    print("验证集: data/dev.jsonl")


if __name__ == "__main__":
    main()
