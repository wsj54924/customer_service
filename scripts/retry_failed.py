#!/usr/bin/env python3
"""Retry failed questions from a submission CSV.

Usage:
    python -m scripts.retry_failed --input submission.csv --output submission_fixed.csv
"""

import csv
import json
import argparse
import time
from pathlib import Path

from loguru import logger

from src.core.chat_engine import ChatEngine


PLACEHOLDER = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"


def is_failed(ret: str) -> bool:
    ret = ret.strip()
    return len(ret) < 50 or ret == PLACEHOLDER or ret == ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", default="question_public.csv", help="Original questions CSV")
    parser.add_argument("--input", default="submission.csv", help="Existing submission CSV")
    parser.add_argument("--output", default="submission_fixed.csv", help="Output fixed CSV")
    args = parser.parse_args()

    # Initialize engine
    engine = ChatEngine()
    engine.initialize()

    # Read original questions
    questions = {}
    with open(args.questions, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            questions[row["id"]] = row["question"]

    # Read existing submission
    with open(args.input, "r", encoding="utf-8") as f:
        results = list(csv.DictReader(f))

    # Find failed
    failed = [(r["id"], questions.get(r["id"], "")) for r in results if is_failed(r["ret"])]
    logger.info(f"Found {len(failed)} failed questions to retry")

    # Retry each
    for qid, question_text in failed:
        if not question_text:
            logger.warning(f"Q{qid}: no question text found, skipping")
            continue

        logger.info(f"Retrying Q{qid}: {question_text[:60]}...")
        start = time.time()

        for attempt in range(3):
            try:
                result = engine.answer(question=question_text)
                answer = result["answer"]
                image_ids = result.get("image_ids", [])

                if image_ids:
                    ret = f"{answer}\n{json.dumps(image_ids, ensure_ascii=False)}"
                else:
                    ret = answer

                # Update in results list
                for r in results:
                    if r["id"] == qid:
                        r["ret"] = ret
                        break

                elapsed = time.time() - start
                logger.info(f"Q{qid} OK ({elapsed:.1f}s, attempt {attempt+1})")
                break

            except Exception as e:
                logger.error(f"Q{qid} attempt {attempt+1} failed: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                else:
                    logger.error(f"Q{qid} FAILED after 3 attempts")

    # Write fixed submission
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(results)

    # Stats
    still_failed = sum(1 for r in results if is_failed(r["ret"]))
    logger.info(f"Saved to {args.output}. Still failed: {still_failed}/{len(results)}")


if __name__ == "__main__":
    main()
