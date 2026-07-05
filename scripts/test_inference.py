"""Smoke test: chạy end-to-end trên 1 ví dụ BTC.

Dùng để kiểm tra nhanh sau khi:
- Cài model trong LM Studio
- Có data/rxnorm_index.json (chạy build_vn_dict.py)

Kết quả in ra stdout để so với ground truth trong đề bài.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.llm_client import LLMClient  # noqa: E402
from src.icd_rag import ICDRetriever, Translator  # noqa: E402
from src.postprocess import assemble_record, validate_output, write_output  # noqa: E402
from src.prompts import (  # noqa: E402
    SYSTEM_PROMPT,
    build_user_prompt,
    format_few_shot_messages,
    load_few_shot,
)
from src.rxnorm_rag import RxNormRetriever  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("smoke_test")

EXAMPLE_INPUT = (
    "Danh sách thuốc trước nhập viện chính xác và đầy đủ.\n"
    "1. amlodipine 10 mg po daily\n"
    "2. aspirin 81 mg po daily\n"
    "3. metoprolol succinate xl 50 mg po daily\n"
    "4. guaifenesin ml po q6h:prn điều trị ho\n"
    "5. nystatin oral suspension 5 ml po qid:prn điều trị đau nhức\n"
    "6. acetaminophen 325-650 mg po q6h:prn điều trị sốt đau\n"
    "7. pravastatin 40 mg po daily\n"
    "8. docusate sodium 100 mg po bid điều trị táo bón\n"
    "9. senna 8.6 mg po bid:prn điều trị táo bón\n"
    "10. clonazepam 0.5 mg po qam:prn điều trị lo âu\n"
    "11. clonazepam 1.5 mg po qhs điều trị lo âu mất ngủ\n"
)

# Ground truth từ đề bài (rút gọn để dễ đối chiếu)
EXPECTED_GT = [
    ("amlodipine 10 mg po daily", "THUỐC", ["308135"]),
    ("aspirin 81 mg po daily", "THUỐC", ["243670"]),
    ("metoprolol succinate xl 50 mg po daily", "THUỐC", ["866436"]),
    ("guaifenesin ml po q6h:prn", "THUỐC", ["392085"]),
    ("ho", "TRIỆU_CHỨNG", None),
    ("nystatin oral suspension 5 ml po qid:prn", "THUỐC", ["7597"]),
    ("đau nhức", "TRIỆU_CHỨNG", None),
    ("acetaminophen 325-650 mg po q6h:prn", "THUỐC", ["313782"]),
    ("sốt đau", "TRIỆU_CHỨNG", None),
    ("pravastatin 40 mg po daily", "THUỐC", ["904475"]),
    ("docusate sodium 100 mg po bid", "THUỐC", ["1099279"]),
    ("táo bón", "TRIỆU_CHỨNG", None),
    ("senna 8.6 mg po bid:prn", "THUỐC", ["312935"]),
    ("táo bón", "TRIỆU_CHỨNG", None),
    ("clonazepam 0.5 mg po qam:prn", "THUỐC", ["197527"]),
    ("lo âu", "TRIỆU_CHỨNG", None),
    ("clonazepam 1.5 mg po qhs", "THUỐC", ["197528"]),
    ("lo âu", "TRIỆU_CHỨNG", None),
    ("mất ngủ", "TRIỆU_CHỨNG", None),
]


def _parse_with_retry(llm: LLMClient, system_prompt: str, user_prompt: str,
                       history: list[dict[str, str]]) -> object:
    msgs = [{"role": "system", "content": system_prompt}] + history + [
        {"role": "user", "content": user_prompt}
    ]
    last_raw = ""
    last_err: Exception | None = None
    for attempt in range(llm.config.max_retries + 1):
        try:
            resp = llm._client.chat.completions.create(  # noqa: SLF001
                model=llm.config.model,
                messages=msgs,
                temperature=llm.config.temperature,
                top_p=llm.config.top_p,
                max_tokens=llm.config.max_tokens,
                timeout=llm.config.timeout,
            )
            content = resp.choices[0].message.content or ""
            last_raw = content
            return llm._extract_json(content)  # noqa: SLF001
        except Exception as exc:
            last_err = exc
            logger.warning("Parse lỗi (lần %d): %s", attempt + 1, exc)
            if attempt >= llm.config.max_retries:
                break
            time.sleep(2)
    raise RuntimeError(f"Smoke test fail: {last_err}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("output/smoke_test.json"))
    args = p.parse_args()

    llm = LLMClient()
    retriever = RxNormRetriever()
    # ICD retriever — không dùng remote trong smoke test để tránh network
    translator = Translator()
    from src.inference import DIAGNOSIS_VN_TO_EN_PRESET
    translator.preset(DIAGNOSIS_VN_TO_EN_PRESET)
    icd = ICDRetriever(translator=translator, use_remote=False)
    few_shot = format_few_shot_messages(load_few_shot())

    user_prompt = build_user_prompt(EXAMPLE_INPUT)
    raw = _parse_with_retry(llm, SYSTEM_PROMPT, user_prompt, few_shot)
    if not isinstance(raw, list):
        if isinstance(raw, dict):
            for k in ("entities", "results", "data"):
                if k in raw and isinstance(raw[k], list):
                    raw = raw[k]
                    break
        if not isinstance(raw, list):
            raw = []

    final = assemble_record(EXAMPLE_INPUT, raw, retriever, icd_retriever=icd)
    write_output(args.out, final)

    ok = validate_output(final)
    print(f"Output count: {len(final)} (expected ~{len(EXPECTED_GT)})")
    print("Valid schema:", ok)
    print("\nPredicted:")
    print(json.dumps(final, ensure_ascii=False, indent=2))

    # Quick candidates sanity check
    print("\nCandidates sanity:")
    pred_dict = {(e["text"], e["type"]): set(e.get("candidates", [])) for e in final}
    for txt, etype, expected_codes in EXPECTED_GT:
        if etype != "THUỐC":
            continue
        got = pred_dict.get((txt, etype), set())
        if expected_codes:
            if set(expected_codes).issubset(got) or got.issubset(set(expected_codes)):
                status = "✅"
            else:
                status = f"❌ got={got} want={set(expected_codes)}"
            print(f"{status}  {txt}  pred={sorted(got)} expected={expected_codes}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
