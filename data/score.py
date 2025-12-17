from datasets import load_dataset
import re

def extract_answer(text):
    if text is None:
        return ""

    # 1) "Answer:" 이후 텍스트만 추출
    #    여러 Answer:가 있더라도 첫 번째만 사용
    if "Answer:" in text:
        text = text.split("Answer:", 1)[1]
    else:
        return text.strip()

    # 2) "Question:"이 다시 나오면 그 앞까지만 사용
    if "Question:" in text:
        text = text.split("Question:", 1)[0]

    # 3) special tokens 제거
    text = re.sub(r"<\|.*?\|>", "", text)

    # 4) 공백/개행/따옴표 정리
    text = text.strip()
    text = text.strip("\"'")   # 양쪽 인용부호 제거

    return text

def load_gold_list():
    dataset = load_dataset("google-research-datasets/nq_open")
    gold_answers = []

    for item in dataset["validation"]:  
        gold_answers.append(item["answer"])
    return gold_answers

import string
from collections import Counter

def normalize(text):
    if text is None:
        return ""
    try:
        text = text.lower().strip()
    except:
        import pdb; pdb.set_trace()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def f1_score(prediction, ground_truth):
    pred_tokens = normalize(prediction).split()
    gt_tokens = normalize(ground_truth).split()

    if len(pred_tokens) == 0 and len(gt_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gt_tokens) == 0:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)

def exact_match(prediction, ground_truth):
    return normalize(prediction) in normalize(ground_truth)

import json
import glob
import os

def evaluate_by_order(folder_path, gold_list):
    json_files = glob.glob(os.path.join(folder_path, "*_pred.json"))
    if len(json_files) == 0:
        print("No pred files found.")
        return

    results = {}

    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            pred_data = json.load(f)
        print(file_path)
        if file_path=="./ours_pred.json" or file_path=="./lora_pred.json"or file_path=="./long_pred.json":
            preds = [d["pred_short_answer"] for d in pred_data]
        else:   
            preds = [d["pred_short_answer"][0] for d in pred_data]

        total = min(len(preds), len(gold_list))
        em_sum = 0
        f1_sum = 0
        # index 기반 매칭
        for i in range(100):
            pred = preds[i]
            golds = gold_list[i]

            em_sum += max([exact_match(pred, gold) for gold in golds])
            f1_sum += max([f1_score(pred, gold) for gold in golds])

        model_name = os.path.basename(file_path)
        results[model_name] = {
            "EM": em_sum / total,
            "F1": f1_sum / total,
            "N": total
        }

    return results


gold_list = load_gold_list()

folder = "./"
scores = evaluate_by_order(folder, gold_list)

for model, score in scores.items():
    print(f"Model: {model}")
    print(f"  EM: {score['EM']:.4f}")
    print(f"  F1: {score['F1']:.4f}")
    print(f"  Samples: {score['N']}")