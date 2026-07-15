"""Estimate final score theo công thức chấm thật:
    final_score = 0.3·text_score + 0.3·assertions_score + 0.4·candidates_score

Metrics:
- text_score: trung bình (1 - WER) trên trường text, ghép predicted vs gold theo position
- assertions_score: Jaccard trên assertions, average theo sample
- candidates_score: Jaccard có trọng số, weighted by (len(ground_truth_candidates) + 1)

Quan trọng:
- Nếu predicted có 1 entity mà gold không có (hallucination) → J=0 toàn sample
- Nếu gold có entity mà predicted không có (miss) → không tính vào, không penalty trực tiếp
  nhưng giảm precision → giảm Jaccard
- Type mismatch: text khớp nhưng type khác → "khái niệm mới" → 0 điểm cả 3 metric, NHÂN ĐÔI

Usage:
    # Nếu có gold data:
    python scripts/estimate_score.py --gold data/gold/ --pred output/

    # Nếu chưa có gold → chỉ estimate WER / structural sanity:
    python scripts/estimate_score.py --pred output/ --structural-only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


# ════════════════════════════════════════════════════════════════════════════════
# WER computation (Word Error Rate, theo Levenshtein trên word-level)
# ════════════════════════════════════════════════════════════════════════════════

def _tokenize_words(s: str) -> list[str]:
    """Tách từ đơn giản (whitespace + punctuation). Cho cả VN + EN."""
    s = s.strip().lower()
    # Tách ký tự alphanumeric cụm
    tokens = re.findall(r"[\w]+|[^\s\w]", s)
    return tokens


def _wer(reference: str, hypothesis: str) -> float:
    """WER = (S + D + I) / N, dùng DP.

    Returns 0.0 nếu cả 2 rỗng; 1.0 nếu 1 trong 2 rỗng.
    """
    ref_tokens = _tokenize_words(reference)
    hyp_tokens = _tokenize_words(hypothesis)

    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    # DP
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # deletion
                    dp[i][j - 1],      # insertion
                    dp[i - 1][j - 1],  # substitution
                )
    return dp[n][m] / n


# ════════════════════════════════════════════════════════════════════════════════
# Entity alignment: match predicted vs gold entities theo position + type
# ════════════════════════════════════════════════════════════════════════════════

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def _span_overlap(a: list[int], b: list[int]) -> float:
    """Tính overlap coefficient (intersection / min length of 2 spans)."""
    if not a or not b or len(a) != 2 or len(b) != 2:
        return 0.0
    s1, e1 = a[0], a[1]
    s2, e2 = b[0], b[1]
    if e1 <= s1 or e2 <= s2:
        return 0.0
    inter = max(0, min(e1, e2) - max(s1, s2))
    len_a = e1 - s1
    len_b = e2 - s2
    min_len = min(len_a, len_b)
    return inter / min_len if min_len > 0 else 0.0


def _align_entities(
    gold_ents: list[dict],
    pred_ents: list[dict],
) -> tuple[list[tuple[dict, dict]], list[dict], list[dict]]:
    """Greedy align: với mỗi gold, tìm pred có type khớp + overlap tốt nhất.

    Returns:
        matched: [(gold, pred), ...]
        missed_gold: gold không có pred nào match
        extra_pred: pred không match gold nào
    """
    used_pred = set()
    matched: list[tuple[dict, dict]] = []
    missed_gold: list[dict] = []

    # Sắp xếp gold theo position để deterministic
    gold_sorted = sorted(
        [e for e in gold_ents if isinstance(e.get("position"), list) and len(e["position"]) == 2],
        key=lambda e: e["position"][0],
    )

    for g in gold_sorted:
        best_pred = None
        best_overlap = 0.0
        for i, p in enumerate(pred_ents):
            if i in used_pred:
                continue
            if p.get("type") != g.get("type"):
                continue
            ov = _span_overlap(g.get("position"), p.get("position"))
            if ov > best_overlap:
                best_overlap = ov
                best_pred = (i, p)
        if best_pred is not None and best_overlap > 0:
            used_pred.add(best_pred[0])
            matched.append((g, best_pred[1]))
        else:
            missed_gold.append(g)

    extra_pred = [p for i, p in enumerate(pred_ents) if i not in used_pred]
    return matched, missed_gold, extra_pred


# ════════════════════════════════════════════════════════════════════════════════
# Metric theo công thức chấm
# ════════════════════════════════════════════════════════════════════════════════

def _jaccard(set_a: set, set_b: set) -> float:
    if not set_a and not set_b:
        return 1.0  # Both empty: perfect
    if not set_a and set_b:
        return 0.0  # Gold empty but pred non-empty: hallucination penalty
    if not set_b and set_a:
        return 0.0  # Gold non-empty but pred empty: miss → 0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def _compute_text_score_sample(gold_ents: list[dict], pred_ents: list[dict]) -> float:
    """text_score = 1 - WER trên concatenated text của từng concept.

    Vì gold chỉ chứa text, không có position phức tạp:
    - Match predicted vs gold theo (text normalized + type) sau khi align by position
    - Tính WER cho mỗi matched pair
    - Trả về trung bình (1 - WER), hoặc 1.0 nếu cả 2 rỗng, 0.0 nếu pred hallucinate
    """
    matched, missed_gold, extra_pred = _align_entities(gold_ents, pred_ents)

    if not matched and not missed_gold and not extra_pred:
        return 1.0
    if extra_pred and not matched:
        # Pure hallucination (chỉ có pred không có gold) → 0
        return 0.0

    if not matched:
        return 0.0

    wers = []
    for g, p in matched:
        g_text = g.get("text", "")
        p_text = p.get("text", "")
        wers.append(_wer(g_text, p_text))

    return 1.0 - (sum(wers) / len(wers))


def _compute_assertions_score_sample(gold_ents: list[dict], pred_ents: list[dict]) -> float:
    """Jaccard trên assertions, áp dụng cho TỪNG cặp matched theo position+type."""
    matched, _, extra_pred = _align_entities(gold_ents, pred_ents)

    if extra_pred and not matched:
        # Hallucination toàn sample → 0
        return 0.0

    if not matched:
        # Gold empty (giả định matched empty + no extra) → 1.0
        # Nhưng nếu có extra thì 0 (đã check ở trên)
        return 1.0

    jaccards = []
    for g, p in matched:
        g_assert = set(g.get("assertions") or [])
        p_assert = set(p.get("assertions") or [])
        jaccards.append(_jaccard(g_assert, p_assert))

    return sum(jaccards) / len(jaccards)


def _compute_candidates_score_sample(gold_ents: list[dict], pred_ents: list[dict]) -> tuple[float, int]:
    """Jaccard trên candidates. Trả về (J, weight).

    Weight = Σ_k (len(ground_truth(k)) + 1) cho sample này.
    """
    matched, missed_gold, extra_pred = _align_entities(gold_ents, pred_ents)

    # Tính total candidates trong gold (cho weight)
    gold_total = sum(len(g.get("candidates") or []) for g in gold_ents)
    weight = gold_total + 1  # +1 theo formula

    if extra_pred and not matched:
        # Hallucination toàn sample → 0
        return 0.0, weight

    if not matched:
        # Gold empty (or only missed) → J=1 nếu pred cũng empty, ngược lại 0
        if not extra_pred:
            return 1.0, weight
        return 0.0, weight

    jaccards = []
    for g, p in matched:
        g_cand = set(str(x) for x in (g.get("candidates") or []))
        p_cand = set(str(x) for x in (p.get("candidates") or []))
        jaccards.append(_jaccard(g_cand, p_cand))

    return sum(jaccards) / len(jaccards), weight


# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gold", type=Path, default=None,
                   help="Folder chứa gold JSON (cùng tên file với pred)")
    p.add_argument("--pred", type=Path, default=Path("output"))
    p.add_argument("--limit", type=int, default=0, help="0 = all")
    p.add_argument("--structural-only", action="store_true",
                   help="Không có gold → chỉ in structural sanity check")
    args = p.parse_args()

    pred_files = sorted(
        [f for f in args.pred.glob("*.json") if f.stem.isdigit()],
        key=lambda x: int(x.stem),
    )
    if args.limit:
        pred_files = pred_files[: args.limit]

    if args.structural_only or args.gold is None:
        # Sanity check
        print(f"\n── STRUCTURAL SANITY (no gold) ──")
        print(f"  Pred files: {len(pred_files)}")
        total = 0
        type_counter: dict[str, int] = defaultdict(int)
        no_text = no_type = no_pos = 0
        empty_cand = empty_assert = 0
        for f in pred_files:
            try:
                data = _load_json(f)
            except Exception:
                continue
            for e in data:
                total += 1
                t = e.get("type", "")
                if t in ("THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
                    type_counter[t] += 1
                if not e.get("text"):
                    no_text += 1
                if not t:
                    no_type += 1
                if not e.get("position"):
                    no_pos += 1
                if e.get("candidates") is None or len(e.get("candidates", [])) == 0:
                    empty_cand += 1
                if e.get("assertions") is None or len(e.get("assertions", [])) == 0:
                    empty_assert += 1
        print(f"  Total entities: {total}")
        print(f"  Type distribution:")
        for t, c in sorted(type_counter.items(), key=lambda x: -x[1]):
            print(f"    {t:<25} {c:>5}")
        print(f"  No text: {no_text}, no type: {no_type}, no position: {no_pos}")
        print(f"  Empty candidates: {empty_cand} ({100*empty_cand/max(1,total):.1f}%)")
        print(f"  Empty assertions: {empty_assert} ({100*empty_assert/max(1,total):.1f}%)")
        return 0

    # Full score estimation
    gold_dir = args.gold
    common = []
    for pf in pred_files:
        rec_id = int(pf.stem)
        gf = gold_dir / f"{rec_id}.json"
        if gf.exists():
            common.append((rec_id, gf, pf))

    print(f"\n═══ SCORE ESTIMATION ═══")
    print(f"  Found {len(common)} file pairs (gold × pred)")
    print(f"  Formula: 0.3·text + 0.3·assertions + 0.4·candidates (weighted)")

    text_total = 0.0
    assert_total = 0.0
    cand_weighted_sum = 0.0
    weight_sum = 0
    n_samples = 0
    type_mismatch_count = 0
    hallucination_count = 0

    for rec_id, gf, pf in common:
        try:
            gold_ents = _load_json(gf)
            pred_ents = _load_json(pf)
        except Exception as exc:
            print(f"  [skip {rec_id}] parse fail: {exc}")
            continue

        if not isinstance(gold_ents, list) or not isinstance(pred_ents, list):
            continue

        # Type mismatch detection (text khớp nhưng type khác)
        for g in gold_ents:
            g_text = _normalize_text(g.get("text", ""))
            g_type = g.get("type", "")
            for p in pred_ents:
                if _normalize_text(p.get("text", "")) == g_text and p.get("type", "") != g_type:
                    type_mismatch_count += 1
                    break

        # Per-sample metrics
        t_i = _compute_text_score_sample(gold_ents, pred_ents)
        a_i = _compute_assertions_score_sample(gold_ents, pred_ents)
        c_i, w_i = _compute_candidates_score_sample(gold_ents, pred_ents)

        # Detect hallucination (extra pred không match gold)
        _, _, extra = _align_entities(gold_ents, pred_ents)
        if extra:
            hallucination_count += 1

        text_total += t_i
        assert_total += a_i
        cand_weighted_sum += c_i * w_i
        weight_sum += w_i
        n_samples += 1

    text_score = text_total / max(1, n_samples)
    assert_score = assert_total / max(1, n_samples)
    cand_score = cand_weighted_sum / max(1, weight_sum)

    final = 0.3 * text_score + 0.3 * assert_score + 0.4 * cand_score

    print(f"\n── KẾT QUẢ ──")
    print(f"  Samples:                  {n_samples}")
    print(f"  Type mismatch penalty:    {type_mismatch_count}")
    print(f"  Hallucination samples:    {hallucination_count}")
    print(f"  ────────────────────────────────────────")
    print(f"  text_score      = {text_score:.4f}  (weight 0.3)")
    print(f"  assertions_score = {assert_score:.4f}  (weight 0.3)")
    print(f"  candidates_score = {cand_score:.4f}  (weight 0.4, weighted)")
    print(f"  ────────────────────────────────────────")
    print(f"  ★ FINAL SCORE   = {final:.4f}")
    print(f"  ────────────────────────────────────────")

    # Per-type breakdown
    print(f"\n── BREAKDOWN THEO TYPE (Jaccard nếu có gold) ──")
    type_text: dict[str, list[float]] = defaultdict(list)
    for rec_id, gf, pf in common:
        try:
            gold_ents = _load_json(gf)
            pred_ents = _load_json(pf)
        except Exception:
            continue
        for tname in ("THUỐC", "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"):
            g_sub = [e for e in gold_ents if e.get("type") == tname]
            p_sub = [e for e in pred_ents if e.get("type") == tname]
            if g_sub or p_sub:
                j = _compute_assertions_score_sample(g_sub, p_sub)
                type_text[tname].append(j)
    for tname, vals in sorted(type_text.items()):
        avg = sum(vals) / max(1, len(vals))
        print(f"  {tname:<25} {avg:.4f}  (over {len(vals)} samples)")

    return 0


if __name__ == "__main__":
    sys.exit(main())