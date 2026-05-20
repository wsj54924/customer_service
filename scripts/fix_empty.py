#!/usr/bin/env python3
"""Fix empty answers by retrying with the chat engine, directly patching the CSV."""

import csv
import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from loguru import logger
from src.core.chat_engine import ChatEngine


EMPTY_IDS = [
    "248", "249", "251", "266", "272", "274",
    "276", "277", "284", "313", "369", "392", "430",
]


def main():
    engine = ChatEngine()
    engine.initialize()

    # Read questions
    questions = {}
    with open("question_public.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions[row["id"]] = row["question"]

    # Read current submission as raw lines for precise patching
    with open("submission_fixed.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = {r["id"]: r for r in reader}

    fixed = 0
    for qid in EMPTY_IDS:
        q = questions.get(qid, "")
        if not q:
            logger.warning(f"Q{qid}: no question text")
            continue

        logger.info(f"Retrying Q{qid}: {q[:80]}...")
        answer_text = ""
        image_ids = []

        for attempt in range(3):
            try:
                result = engine.answer(question=q)
                answer_text = result.get("answer", "")
                image_ids = result.get("image_ids", [])
                if answer_text and len(answer_text.strip()) >= 10:
                    break
                logger.warning(f"Q{qid} attempt {attempt+1}: got {len(answer_text)} chars")
                time.sleep(3)
            except Exception as e:
                logger.error(f"Q{qid} attempt {attempt+1} error: {e}")
                time.sleep(5 * (attempt + 1))

        if answer_text and len(answer_text.strip()) >= 10:
            if image_ids:
                ret = f"{answer_text}\n{json.dumps(image_ids, ensure_ascii=False)}"
            else:
                ret = answer_text
            rows[qid]["ret"] = ret
            fixed += 1
            logger.info(f"Q{qid} FIXED: {len(ret)} chars")
        else:
            # Use a service-style fallback
            rows[qid]["ret"] = (
                "您好，感谢您的咨询。关于您提到的产品操作问题，"
                "我目前没有找到对应的操作说明。建议您查看产品附带的纸质说明书，"
                "或联系人工客服获取更详细的指导。"
            )
            logger.warning(f"Q{qid}: using fallback answer")

    # Write back
    with open("submission_fixed.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for qid in sorted(rows.keys(), key=lambda x: int(x)):
            writer.writerow(rows[qid])

    logger.info(f"Done. Fixed: {fixed}/{len(EMPTY_IDS)}. Saved to submission_fixed.csv")


if __name__ == "__main__":
    main()
