import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -------------------------------
# 1. 데이터 로드
# -------------------------------
data = load_dataset("keunha/nq_top20")

# -------------------------------
# 2. 모델 GPU 로딩
# -------------------------------
model = SentenceTransformer(
    "Qwen/Qwen3-Embedding-4B",
    device="cuda"
)

# -------------------------------
# 3. 모든 ctx 수집 (중복 제거)
# -------------------------------
all_embeddings = {}   # {ctx_id: ctx_text}

for split in ["train", "validation"]:
    print(f"Collecting texts from split: {split}")
    for line in tqdm(data[split]):
        for ctx in line["ctxs"]:
            cid = ctx["id"]
            if cid not in all_embeddings:
                all_embeddings[cid] = ctx["text"]

print(f"총 수집된 ctx 개수 = {len(all_embeddings)}")

# -------------------------------
# 4. 한 번에 encode — GPU batch 처리
# -------------------------------
ids = list(all_embeddings.keys())
texts = [all_embeddings[i] for i in ids]

ctx_embs = model.encode(
    texts,
    batch_size=128,              # GPU 효율 좋게 크게 설정
    convert_to_numpy=True,
    show_progress_bar=True
)

# -------------------------------
# 5. 최종 매핑 (id → embedding)
# -------------------------------
final_embedding = {}

for cid, emb in zip(ids, ctx_embs):
    final_embedding[cid] = emb

print("임베딩 완료:", len(final_embedding))

# -------------------------------
# 6. 저장 (numpy npz 권장)
# -------------------------------
np.savez("nq_ctx_embeddings.npz", 
         ids=np.array(ids), 
         vectors=ctx_embs)

print("저장 완료 → nq_ctx_embeddings.npz")