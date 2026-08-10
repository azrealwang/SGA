#!/usr/bin/env python3
"""Convert VQAv2 val annotations + questions to generic jsonl for this codebase."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--questions-json", required=True, help="v2_OpenEnded_mscoco_val2014_questions.json")
    p.add_argument("--annotations-json", required=True, help="v2_mscoco_val2014_annotations.json")
    p.add_argument("--output-jsonl", required=True)
    p.add_argument("--image-prefix", default="COCO_val2014_", help="Filename prefix inside val2014/")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    with open(args.questions_json, "r", encoding="utf-8") as f:
        qdata = json.load(f)["questions"]
    with open(args.annotations_json, "r", encoding="utf-8") as f:
        adata = {a["question_id"]: a for a in json.load(f)["annotations"]}

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for q in qdata:
            qid = q["question_id"]
            if qid not in adata:
                continue
            ann = adata[qid]
            answers = [x["answer"] for x in ann["answers"]]
            if not answers:
                continue
            image_file = f"{args.image_prefix}{q['image_id']:012d}.jpg"
            row = {
                "id": str(qid),
                "image": image_file,
                "question": q["question"],
                "answer": answers[0],
                "answers": answers,
            }
            out.write(json.dumps(row) + "\n")
            n += 1
            if args.max_samples is not None and n >= args.max_samples:
                break
    print(f"Wrote {n} samples to {out_path}")


if __name__ == "__main__":
    main()
