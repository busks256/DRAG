from vllm import LLM, SamplingParams
from vllm.sampling_params import BeamSearchParams
from datasets import load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer
import json
import os
from typing import Dict
from tqdm import tqdm
from huggingface_hub import login

login(token="")

data_dir = "../data/NQ_output_middle_validation.json"
output_dir = "../data/NQ_output_middle_validation_pred.json"
base_model = "Qwen/Qwen3-4B-Instruct-2507"
# Use or not?
def remove_repeated_ngrams(text, n: int = 3):
    tokens = text.split()
    seen = set()
    new_tokens = []

    for i in range(len(tokens)):
        ngram = tuple(tokens[i:i+n])
        if len(ngram) == n and ngram in seen:
            break
        seen.add(ngram)
        new_tokens.append(tokens[i])
    return " ".join(new_tokens)



def generate_kcdata():
    llm = LLM(model=base_model, tensor_parallel_size=1)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    sampling_params = SamplingParams(
        n=1,                     # 최종 출력은 1개 선택 (원하면 n=10도 가능)
        temperature=0.0,
        max_tokens=256         
    )    
    with open(data_dir, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sample in data:
        question = sample["question"]
        text = " ".join([doc["summary"] for doc in sample['ctxs']])
        # 메시지 포맷 → context 제외
        messages = [
                {"role": "system", "content": (
                    "You are an extraction QA assistant. "
                    "Output minimal answer span (1–3 words)."
                )},
                {
                    "role": "user",
                    "content": (
                        f"Context: {text}\n"
                        f"Question: {question}\n"
                    )
                },{
                    "role": "assistant",
                    "content": (
                        f"Answer: "
                    )
                }
            ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sampling_params)
        answers = [out.text for out in outputs[0].outputs]
        sample["pred_short_answer"]= answers
        messages = [
            {
                "role": "system",
                "content": "You are a knowledgeable QA assistant."
            },
            {
                "role": "user",
                "content": (
                    f"Context: {text}\n"
                    f"Question: {question}\n"
                )
            },{
                "role": "assistant",
                "content": (
                    "Answer: "
                )
            }
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = llm.generate([prompt], sampling_params)
        answers = [out.text for out in outputs[0].outputs]
        sample["pred_long_answer"]= answers
    with open(output_dir, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
if __name__ == "__main__":
    generate_kcdata()