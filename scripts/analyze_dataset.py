#!/usr/bin/env python3
"""Dataset analysis script for the multimodal customer service competition."""

import csv
import os
import re
import json
from pathlib import Path
from collections import Counter, defaultdict

BASE_DIR = Path(__file__).parent.parent
MANUAL_DIR = BASE_DIR / "手册"
IMAGE_DIR = MANUAL_DIR / "插图"
QUESTION_FILE = BASE_DIR / "question_public.csv"
SUBMISSION_FILE = BASE_DIR / "submission_example.csv"


def analyze_manuals():
    """Analyze all manual text files."""
    print("=" * 60)
    print("1. MANUAL ANALYSIS")
    print("=" * 60)

    manuals = sorted(MANUAL_DIR.glob("*.txt"))
    total_chars = 0
    total_pics = 0
    manual_stats = []

    for mf in manuals:
        content = mf.read_text(encoding="utf-8")
        chars = len(content)
        pic_count = content.count("<PIC>")
        lines = content.count("\n") + 1
        # Extract image IDs mentioned in the manual
        pic_ids = re.findall(r'"([^"]+)"', content.split("[")[1].split("]")[0]) if "[" in content else []

        manual_stats.append({
            "name": mf.stem,
            "chars": chars,
            "lines": lines,
            "pic_count": pic_count,
        })
        total_chars += chars
        total_pics += pic_count

    print(f"\nTotal manuals: {len(manuals)}")
    print(f"Total characters: {total_chars:,} ({total_chars / 10000:.1f}万字)")
    print(f"Total <PIC> placeholders: {total_pics}")
    print(f"\nPer-manual breakdown:")
    print(f"{'Name':<25} {'Chars':>8} {'Lines':>7} {'PICs':>6}")
    print("-" * 50)
    for s in sorted(manual_stats, key=lambda x: -x["chars"]):
        print(f"{s['name']:<25} {s['chars']:>8,} {s['lines']:>7} {s['pic_count']:>6}")

    return manual_stats


def analyze_images():
    """Analyze image files."""
    print("\n" + "=" * 60)
    print("2. IMAGE ANALYSIS")
    print("=" * 60)

    images = list(IMAGE_DIR.glob("*.*"))
    print(f"\nTotal images: {len(images)}")

    # Group by prefix
    prefix_counter = Counter()
    for img in images:
        name = img.stem
        # Extract prefix (everything before last _number)
        match = re.match(r"(.+?)_\d+$", name)
        if match:
            prefix_counter[match.group(1)] += 1
        else:
            prefix_counter[name] += 1

    print(f"Unique product categories: {len(prefix_counter)}")
    print(f"\nImages per category (top 20):")
    for prefix, count in prefix_counter.most_common(20):
        print(f"  {prefix:<35} {count:>4}")

    # File size stats
    sizes = [img.stat().st_size for img in images]
    print(f"\nImage size stats:")
    print(f"  Total: {sum(sizes) / 1024 / 1024:.1f} MB")
    print(f"  Min: {min(sizes) / 1024:.1f} KB")
    print(f"  Max: {max(sizes) / 1024:.1f} KB")
    print(f"  Avg: {sum(sizes) / len(sizes) / 1024:.1f} KB")

    return prefix_counter


def analyze_questions():
    """Analyze question CSV."""
    print("\n" + "=" * 60)
    print("3. QUESTION ANALYSIS")
    print("=" * 60)

    with open(QUESTION_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse CSV manually due to quoting issues
    lines = content.strip().split("\n")
    header = lines[0]
    print(f"Header: {header}")
    print(f"Total lines: {len(lines)} (incl. header)")

    # Parse questions
    questions = []
    i = 1
    while i < len(lines):
        line = lines[i]
        # Check if this is a multi-line entry (starts with id,""" )
        match = re.match(r'^(\d+),"""(.*)', line)
        if match:
            qid = match.group(1)
            rest = match.group(2)
            # Collect continuation lines until we find the closing """
            while not rest.rstrip().endswith('"""'):
                i += 1
                if i < len(lines):
                    rest += "\n" + lines[i]
            # Clean up the question text
            qtext = rest.rstrip().rstrip('"').strip()
            questions.append({"id": qid, "question": qtext})
        else:
            # Simple single-line format
            parts = line.split(",", 1)
            if len(parts) == 2:
                qid = parts[0]
                qtext = parts[1].strip().strip('"')
                questions.append({"id": qid, "question": qtext})
        i += 1

    print(f"Parsed questions: {len(questions)}")

    # Categorize questions
    product_keywords_zh = [
        "钻", "表带", "健身", "冰箱", "烤箱", "键盘", "相机", "洗碗", "空调",
        "VR", "鼠标", "耳机", "摩托艇", "水泵", "清洁", "温控", "吹风",
        "风扇", "空气净化", "座椅", "椅子", "电视", "摄像", "电话", "剃须",
        "牙刷", "割草", "雪地", "压力锅", "洗衣机", "充电", "电池",
    ]

    service_keywords = [
        "退货", "换货", "退款", "发票", "物流", "快递", "运费", "投诉",
        "保修", "维修", "售后", "客服", "发货", "收货", "赔偿",
        "7天无理由", "假货", "破损", "缺件", "补发",
    ]

    zh_questions = []
    en_questions = []

    for q in questions:
        # Check if primarily English
        en_chars = sum(1 for c in q["question"] if c.isascii() and c.isalpha())
        zh_chars = sum(1 for c in q["question"] if "一" <= c <= "鿿")
        if en_chars > zh_chars * 2:
            en_questions.append(q)
        else:
            zh_questions.append(q)

    print(f"\nChinese questions: {len(zh_questions)}")
    print(f"English questions: {len(en_questions)}")

    # Categorize Chinese questions
    product_qs = []
    service_qs = []
    ambiguous_qs = []

    for q in zh_questions:
        text = q["question"]
        is_product = any(kw in text for kw in product_keywords_zh)
        is_service = any(kw in text for kw in service_keywords)
        if is_product and not is_service:
            product_qs.append(q)
        elif is_service and not is_product:
            service_qs.append(q)
        elif is_product and is_service:
            ambiguous_qs.append(q)
        else:
            ambiguous_qs.append(q)

    print(f"\n  Product-related: {len(product_qs)}")
    print(f"  Service-related: {len(service_qs)}")
    print(f"  Ambiguous/Other: {len(ambiguous_qs)}")

    # Show examples
    print("\nSample English questions (product manual):")
    for q in en_questions[:5]:
        print(f"  [{q['id']}] {q['question'][:80]}")

    print("\nSample Chinese product questions:")
    for q in product_qs[:5]:
        print(f"  [{q['id']}] {q['question'][:80]}")

    print("\nSample Chinese service questions:")
    for q in service_qs[:5]:
        print(f"  [{q['id']}] {q['question'][:80]}")

    # Multi-turn detection
    multi_turn = [q for q in questions if "\n" in q["question"] or '","' in q["question"] or "？" in q["question"] and q["question"].count("？") > 1]
    print(f"\nPotential multi-turn questions: {len(multi_turn)}")

    return questions


def analyze_submission():
    """Analyze submission format."""
    print("\n" + "=" * 60)
    print("4. SUBMISSION FORMAT ANALYSIS")
    print("=" * 60)

    with open(SUBMISSION_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Header: {lines[0].strip()}")
    print(f"Total entries: {len(lines) - 1}")
    print(f"\nFirst 5 entries:")
    for line in lines[:6]:
        print(f"  {line.strip()[:100]}")
    print(f"\nLast 3 entries:")
    for line in lines[-3:]:
        print(f"  {line.strip()[:100]}")


def map_manuals_to_images():
    """Try to map manual text files to their image prefixes."""
    print("\n" + "=" * 60)
    print("5. MANUAL-IMAGE MAPPING")
    print("=" * 60)

    manuals = sorted(MANUAL_DIR.glob("*.txt"))

    for mf in manuals:
        content = mf.read_text(encoding="utf-8")
        # Find image references in format [...]
        bracket_content = re.findall(r'\[([^\]]+)\]', content)
        image_ids = []
        for bc in bracket_content:
            ids = re.findall(r'"([^"]+)"', bc)
            image_ids.extend(ids)

        if image_ids:
            # Get unique prefixes
            prefixes = set()
            for iid in image_ids:
                match = re.match(r"(.+?)_\d+$", iid)
                if match:
                    prefixes.add(match.group(1))
                else:
                    prefixes.add(iid)
            print(f"\n{mf.stem}:")
            print(f"  Referenced images: {len(image_ids)}")
            print(f"  Image prefixes: {', '.join(sorted(prefixes))}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    manual_stats = analyze_manuals()
    prefix_counter = analyze_images()
    questions = analyze_questions()
    analyze_submission()
    map_manuals_to_images()

    # Save analysis results
    results = {
        "manual_count": len(manual_stats),
        "total_characters": sum(s["chars"] for s in manual_stats),
        "total_images": sum(prefix_counter.values()),
        "image_categories": len(prefix_counter),
        "total_questions": len(questions),
        "manual_stats": manual_stats,
        "image_distribution": dict(prefix_counter.most_common()),
    }

    output_path = BASE_DIR / "data" / "analysis_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n\nAnalysis results saved to {output_path}")
