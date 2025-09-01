# -*- coding: utf-8 -*-
# 실행: streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
import os, re, json
from typing import List, Dict, Any
import streamlit as st
import plotly.graph_objects as go

# --- 성능: 로컬 CPU 배치 추론 ---
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# (선택) 스레드 수 튜닝: 과하면 오히려 느려질 수 있음
try:
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))
except Exception:
    pass

# =========================
# 기본 UI 설정
# =========================
st.set_page_config(page_title="소설 감정 곡선 (로컬 ML, 배치 추론)", layout="wide")
st.title("📖 소설 텍스트 감정 곡선 · 로컬 ML(배치) 전용")
st.caption("Inference API를 사용하지 않습니다. 로컬 CPU에서 배치 추론으로 빠르게 처리합니다.")

st.markdown(
    "- 기본 모델: `dlckdfuf141/korean-emotion-kluebert-v2` (KLUE-BERT 기반, 7라벨)\n"
    "- 파이프라인 대신 **직접 배치 추론**(tokenizer+model)으로 속도 최적화\n"
    "- BERT 512 토큰 한계를 고려해 `max_length`를 줄여 CPU 속도를 끌어올립니다."
)

# =========================
# 라벨 ↔ 가중치(Valence) 매핑
# =========================
DEFAULT_VALENCE = {
    "행복": +1.0,
    "중립": 0.0,
    "놀람": 0.0,   # 맥락 따라 상하 가능 → 중립 취급
    "공포": -0.7,
    "분노": -0.9,
    "슬픔": -0.8,
    "혐오": -0.85,
}

# =========================
# 문장/청크 유틸
# =========================
SENT_SPLIT_RE = re.compile(r"(?<=[\.!?…\n])\s+|(?<=다\.)\s+|(?<=요\.)\s+", re.M)

def split_sentences_kr(text: str) -> List[str]:
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    return sents

def make_chunks(text: str, max_chars=380, overlap=80) -> List[str]:
    """
    문장 단위로 max_chars 내외 청크 생성(+겹침).
    BERT 512 토큰 제한 고려 → 320~384 토큰 목표로 문자 기준 350~450 추천.
    """
    sents = split_sentences_kr(text)
    chunks, cur, cur_len = [], [], 0
    for s in sents:
        L = len(s)
        if cur_len + L + 1 <= max_chars:
            cur.append(s); cur_len += L + 1
        else:
            if cur:
                chunks.append(" ".join(cur))
            tail = chunks[-1][-overlap:] if chunks else ""
            cur = [tail + (" " if tail else "") + s]
            cur_len = len(cur[0])
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# =========================
# 로컬 분류기 (배치 추론)
# =========================
@st.cache_resource
def load_cls_model(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    mdl.eval()
    return tok, mdl

def classify_batch(
    texts: List[str],
    model_id: str,
    max_length: int = 320,
    batch_size: int = 16
) -> List[List[Dict[str, float]]]:
    """
    로컬 CPU 배치 분류. 반환은 pipeline(text-classification)과 호환되는 형태:
    각 입력마다 [{"label":..., "score":...}, ...]
    """
    tok, mdl = load_cls_model(model_id)
    results: List[List[Dict[str, float]]] = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            logits = mdl(**enc).logits  # [B, C]
            probs = F.softmax(logits, dim=-1).cpu().numpy()  # [B, C]
            labels = [mdl.config.id2label[j] for j in range(len(mdl.config.id2label))]
            for row in probs:
                results.append([
                    {"label": lab, "score": float(sc)} for lab, sc in zip(labels, row)
                ])
    return results

# =========================
# 점수화/스무딩/시각화 유틸
# =========================
def to_valence_score(api_output: List[Dict[str, Any]], valence_map: Dict[str, float]) -> Dict[str, Any]:
    """
    [{'label':'행복','score':0.87}, ...] → 가중 합(연속값), top1 등 계산
    라벨이 매핑에 없으면 0 가중치.
    """
    if not api_output:
        return {"weighted": 0.0, "top": "중립", "probs": {}}

    weighted = 0.0
    per_label = {}
    for item in api_output:
        lbl = str(item.get("label", "")).strip()
        sc = float(item.get("score", 0.0))
        if not lbl:
            continue
        per_label[lbl] = sc
        weighted += sc * float(valence_map.get(lbl, 0.0))

    if not per_label:
        return {"weighted": 0.0, "top": "중립", "probs": {}}

    top = max(per_label, key=per_label.get)
    return {"weighted": float(weighted), "top": top, "probs": per_label}

def smooth_moving_avg(vals: List[float], k: int = 5) -> List[float]:
    if k <= 1:
        return vals
    out = []
    for i in range(len(vals)):
        s = vals[max(0, i-k+1): i+1]
        out.append(sum(s)/len(s))
    return out

def plot_timeline(x, raw_vals, smoothed=None, k=1):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=raw_vals, mode="lines+markers", name="원시 점수"))
    if smoothed is not None and k > 1:
        fig.add_trace(go.Scatter(x=x, y=smoothed, mode="lines", name=f"이동평균(k={k})"))
    fig.add_hline(y=0, line_dash="dash")
    fig.update_layout(
        title="작품 흐름에 따른 감정 곡선 (0=중립, +긍정/−부정)",
        xaxis_title="작품 흐름 (청크 번호)",
        yaxis_title="감정 점수 (−1 ~ +1)",
        height=520
    )
    st.plotly_chart(fig, use_container_width=True)

# =========================
# 사이드바 옵션
# =========================
st.sidebar.header("옵션")
MODEL_ID = st.sidebar.text_input(
    "모델 ID (text-classification 호환)",
    value="dlckdfuf141/korean-emotion-kluebert-v2",
    help="예: dlckdfuf141/korean-emotion-kluebert-v2 (7라벨)"
)
max_chars = st.sidebar.slider("청크 길이(문자)", 250, 600, 380, 10)
overlap = st.sidebar.slider("겹침(문자)", 0, 200, 80, 10)
max_length = st.sidebar.slider("토큰 최대 길이(max_length)", 128, 512, 320, 32)
batch_size = st.sidebar.slider("배치 크기", 4, 64, 16, 4)
smooth_k = st.sidebar.slider("이동평균 윈도", 1, 21, 5, 2)

st.sidebar.markdown("**라벨-가중치(Valence) 매핑(JSON)**")
valence_text = st.sidebar.text_area(
    "예: {\"행복\":1.0, \"중립\":0.0, \"슬픔\":-0.8, ...}",
    json.dumps(DEFAULT_VALENCE, ensure_ascii=False, indent=2),
    height=200
)
try:
    USER_VALENCE = json.loads(valence_text)
    if isinstance(USER_VALENCE, dict) and USER_VALENCE:
        VALENCE = {k: float(v) for k, v in USER_VALENCE.items()}
    else:
        VALENCE = DEFAULT_VALENCE.copy()
except Exception:
    st.sidebar.warning("Valence JSON 파싱 실패 → 기본 매핑 사용")
    VALENCE = DEFAULT_VALENCE.copy()

# =========================
# 입력
# =========================
txt = st.text_area(
    "분석할 소설 텍스트(전문 붙여넣기 가능, 자동 분할):",
    height=280,
    placeholder="소설 전문 또는 긴 문단을 붙여넣으세요."
)

colA, colB = st.columns([1,1])
with colA:
    show_table = st.checkbox("청크별 라벨 표 보기", value=False)
with colB:
    show_probs = st.checkbox("라벨 확률 보기", value=False)

run = st.button("🧠 감정 분석 & 시각화 실행")

# =========================
# 실행
# =========================
if run:
    if not txt.strip():
        st.warning("텍스트를 입력하세요.")
        st.stop()

    # 1) 분할
    chunks = make_chunks(txt, max_chars=max_chars, overlap=overlap)
    if not chunks:
        st.error("분할 결과가 비어 있습니다. 더 긴 텍스트를 입력해 주세요.")
        st.stop()
    st.success(f"자동 분할 완료: 총 {len(chunks)}개 청크")

    # 2) 배치 분류 (로컬 CPU)
    with st.spinner("로컬 배치 추론 중..."):
        all_outputs = classify_batch(
            chunks,
            model_id=MODEL_ID,
            max_length=max_length,
            batch_size=batch_size
        )

    # 3) 점수화 및 표 준비
    timeline_vals, top_labels, rows = [], [], []
    seen_labels = set()
    for i, (ch, out) in enumerate(zip(chunks, all_outputs), 1):
        sc = to_valence_score(out, VALENCE)
        timeline_vals.append(sc["weighted"])
        top_labels.append(sc["top"])
        seen_labels.update(sc["probs"].keys())

        row = {
            "idx": i,
            "text": ch[:120] + "..." if len(ch) > 120 else ch,
            "score": round(sc["weighted"], 4),
            "top": sc["top"],
        }
        if show_probs:
            pl = sorted(sc["probs"].items(), key=lambda x: -x[1])[:3]
            row["probs_top3"] = ", ".join([f"{k}:{v:.2f}" for k, v in pl])
        rows.append(row)

    # 4) 시각화
    smoothed = smooth_moving_avg(timeline_vals, k=smooth_k)
    x = list(range(1, len(chunks)+1))
    plot_timeline(x, timeline_vals, smoothed if smooth_k > 1 else None, k=smooth_k)

    # 5) 표(선택)
    if show_table:
        st.subheader("청크별 주요 라벨/점수")
        st.dataframe(rows, use_container_width=True)

    # 6) 라벨 진단
    with st.expander("라벨 진단"):
        st.write("모델이 실제로 반환한 라벨:", sorted(seen_labels))
        missing = [l for l in seen_labels if l not in VALENCE]
        if missing:
            st.warning(f"VALENCE 매핑에 없는 라벨: {missing} → 매핑에 추가하지 않으면 0으로 처리됩니다.")

    # 7) 결과 다운로드
    result = {
        "model": MODEL_ID,
        "valence_mapping": VALENCE,
        "chunks": chunks,
        "timeline_scores": timeline_vals,
        "smoothed": smoothed,
        "top_labels": top_labels,
        "rows": rows
    }
    st.download_button(
        "📥 결과 JSON 다운로드",
        data=json.dumps(result, ensure_ascii=False, indent=2),
        file_name="novel_emotion_timeline_local.json",
        mime="application/json"
    )
