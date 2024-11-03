import random
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlamaForCausalLM, LlamaTokenizer
from optimum.quanto import (
    Optimizer,
    qtype,
    qint4,
    quantize,
    Calibration,
    freeze,
    MaxOptimizer
)
from optimum.quanto.tensor.optimizers import AdaptiveAxisOptimizer, MaxOptimizer

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.random.manual_seed(seed)

def load_and_process_wikitext2(tokenizer, nsamples, seed, seqlen):
    """Load and process WikiText2 dataset"""
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_data_loader(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    """Get appropriate data loader based on dataset name"""
    if 'wikitext2' in name:
        return load_and_process_wikitext2(tokenizer, nsamples, seed, seqlen)
    raise ValueError(f"Dataset {name} not supported")

def evaluate_perplexity(model, tokenizer, device):
    """评估模型在wikitext数据集上的困惑度"""
    print("Evaluating perplexity on wikitext2...")
    
    # 1. 加载和预处理数据
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # 将所有文本连接起来
    text = " ".join(test_dataset["text"])
    
    # 进行tokenization
    encodings = tokenizer(
        text,
        return_tensors="pt",
        max_length=model.config.max_position_embeddings,
        truncation=True,
    )
    
    # 获取sequence length
    seq_len = model.config.max_position_embeddings
    
    # 计算样本数
    testenc = encodings.input_ids
    nsamples = testenc.numel() // seq_len
    
    # 存储negative log likelihoods
    nlls = []
    print(f"Number of samples: {nsamples}")
    
    # 评估
    model.eval()
    with torch.no_grad():
        for i in range(0, nsamples, 1):  # batch_size=1
            if i % 50 == 0:
                print(f"Processing sample {i}")
            
            # 准备输入
            inputs = testenc[:, (i * seq_len):((i + 1) * seq_len)].to(device)
            inputs = inputs.reshape(1, seq_len)  # Add batch dimension
            
            # 前向传播
            outputs = model(inputs)
            logits = outputs.logits
            
            # 准备下一个token预测的logits和labels
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]
            
            # 计算loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1)
            )
            
            # 计算negative log likelihood
            neg_log_likelihood = loss.float() * (seq_len - 1)
            nlls.append(neg_log_likelihood)
            
            # 每处理50个样本清理一次缓存
            if i % 50 == 0:
                torch.cuda.empty_cache()
    
    # 计算困惑度
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * (seq_len - 1)))
    
    return ppl.item()

def calibrate_model(model, tokenizer, device, batch_size=4, num_samples=100):
    """Calibrate model using WikiText2 dataset"""
    print("Calibrating model...")
    cal_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    
    with Calibration(momentum=0.9):
        model.eval()
        total = 0
        for batch in tqdm(cal_dataset.iter(batch_size=batch_size), desc="Calibrating"):
            inputs = tokenizer(
                batch["text"], 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)
                
            total += input_ids.size(0)
            if total >= num_samples:
                break

def get_quantization_config():
    """Get layer-wise quantization axis configuration"""
    return {
        "k_proj": {"axis": -1, "pattern": r".*\.self_attn\.k_proj$"},
        "v_proj": {"axis": 0, "pattern": r".*\.self_attn\.v_proj$"}, 
        "q_proj": {"axis": 0, "pattern": r".*\.self_attn\.q_proj$"},
        "gate_proj": {"axis": 0, "pattern": r".*\.mlp\.gate_proj$"},
        "up_proj": {"axis": 0, "pattern": r".*\.mlp\.up_proj$"}
    }

def main():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载模型和tokenizer
    print("Loading model and tokenizer...")
    model = LlamaForCausalLM.from_pretrained(
        "/data2/share/llama-2/Llama-2-7b-hf",
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    tokenizer = LlamaTokenizer.from_pretrained("/data2/share/llama-2/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 3. 评估初始困惑度
    print("Evaluating initial perplexity...")
    initial_ppl = evaluate_perplexity(model, tokenizer, device)
    print(f"Initial perplexity: {initial_ppl:.2f}")
    
    # 4. 定义量化配置
    axis_config = {
        "k_proj": {"axis": -1, "pattern": r".*\.self_attn\.k_proj$"},
        "v_proj": {"axis": 0, "pattern": r".*\.self_attn\.v_proj$"},
        "q_proj": {"axis": -1, "pattern": r".*\.self_attn\.q_proj$"},
        "gate_proj": {"axis": -1, "pattern": r".*\.mlp\.gate_proj$"},
        "up_proj": {"axis": -1, "pattern": r".*\.mlp\.up_proj$"}
    }
    
    # 5. 创建adaptive optimizer
    adaptive_optimizer = AdaptiveAxisOptimizer(
        base_optimizer=MaxOptimizer(),
        axis_mapping={},
        pattern_mapping={
            config["pattern"]: config["axis"]
            for config in axis_config.values()
        }
    )
    # max_optimizer = MaxOptimizer()
    # 6. 量化模型
    print("Quantizing model...")
    quantize(
        model,
        weights=qint4,
        optimizer=adaptive_optimizer
    )
    
    # 7. 校准模型
    print("Calibrating model...")
    calibrate_model(model, tokenizer, device)
    
    # 8. 冻结模型
    print("Freezing model...")
    freeze(model)
    
    # 9. 评估量化后的困惑度
    print("Evaluating quantized perplexity...")
    quantized_ppl = evaluate_perplexity(model, tokenizer, device)
    print(f"Quantized perplexity: {quantized_ppl:.2f}")
    
    # 10. 打印结果对比
    print("\nResults:")
    print(f"Original PPL: {initial_ppl:.2f}")
    print(f"Quantized PPL: {quantized_ppl:.2f}")
    print(f"PPL Degradation: {((quantized_ppl - initial_ppl) / initial_ppl * 100):.2f}%")

if __name__ == "__main__":
    main()