Run `python -m cancerflex.synthetic --out-dir examples/synthetic` to generate a synthetic image and prompt,
then `cancerflex-infer --checkpoint /path/to/weights --image examples/synthetic/sample.png --prompt "$(python -c 'import json;print(json.load(open("examples/synthetic/sample.json"))["prompt"])')"`.
