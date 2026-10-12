"""Original | imagined | wrong strips with the question underneath, for eyeballing.

    python scripts/contact_sheet.py data/pilot --task stability --k 4
"""
import argparse
import json
import os
import textwrap

from PIL import Image, ImageDraw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--task")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--out")
    a = p.parse_args()

    items = [json.loads(line) for line in open(os.path.join(a.root, "questions.jsonl"))]
    items = [it for it in items if a.task in (None, it["task"])][: a.k]
    s, text_h = a.size, 70
    sheet = Image.new("RGB", (3 * s, len(items) * (s + text_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, it in enumerate(items):
        y = r * (s + text_h)
        for c, st in enumerate(("original", "imagined", "wrong")):
            img = Image.open(os.path.join(a.root, it[st]["image"])).resize((s, s))
            sheet.paste(img, (c * s, y))
            draw.text((c * s + 4, y + 4), st, fill="black")
        caption = f"[{it['id']}] {it['question']}  ANSWER: {it['answer']}  (wrong img -> {it['wrong_answer']})"
        draw.multiline_text((4, y + s + 4), "\n".join(textwrap.wrap(caption, 125)), fill="black")
    out = a.out or os.path.join(a.root, f"sheet_{a.task or 'all'}.png")
    sheet.save(out)
    print(out)


if __name__ == "__main__":
    main()
