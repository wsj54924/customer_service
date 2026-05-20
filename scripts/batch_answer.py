#!/usr/bin/env python3
"""Batch answer all questions and generate submission CSV.

Usage:
    python -m scripts.batch_answer --input question_public.csv --output submission.csv
"""

import csv
import argparse
import time
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from src.core.chat_engine import ChatEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="question_public.csv", help="Input questions CSV")
    parser.add_argument("--output", default="submission.csv", help="Output submission CSV")
    parser.add_argument("--limit", type=int, default=None, help="Process only N questions (for testing)")
    args = parser.parse_args()

    # Initialize engine
    engine = ChatEngine()
    engine.initialize()

    # Read questions
    questions = []
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    if args.limit:
        questions = questions[:args.limit]

    logger.info(f"Processing {len(questions)} questions...")

    # Answer each question
    results = []
    for q in tqdm(questions, desc="Answering"):
        qid = q["id"]
        question_text = q["question"]
        start = time.time()

        try:
            result = engine.answer(question=question_text)
            answer = result["answer"]
            image_ids = result.get("image_ids", [])

            # Format ret field: answer text + image IDs array
            if image_ids:
                import json
                ret = f"{answer}\n{json.dumps(image_ids, ensure_ascii=False)}"
            else:
                ret = answer

        except Exception as e:
            logger.error(f"Error on question {qid}: {e}")
            ret = "您好，您的问题已收到，请您耐心等待处理结果，谢谢。"

        elapsed = time.time() - start
        results.append({"id": qid, "ret": ret})

        if elapsed > 25:
            logger.warning(f"Slow response for Q{qid}: {elapsed:.1f}s")

    # Write submission
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "ret"])
        writer.writeheader()
        writer.writerows(results)

    logger.info(f"Submission saved to {args.output} ({len(results)} answers)")


if __name__ == "__main__":
    main()
