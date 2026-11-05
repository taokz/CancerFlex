import json
import subprocess
import sys

from cancerflex.infer import main as infer_main
from cancerflex.synthetic import write_sample


def test_synthetic_sample(tmp_path):
    rec = write_sample(tmp_path, size=64)
    assert (tmp_path / "sample.png").exists() and rec["synthetic"] is True


def test_infer_cli_synthetic(tiny_checkpoint, capsys):
    rc = infer_main(["--checkpoint", str(tiny_checkpoint), "--synthetic", "--width", "16", "--device", "cpu",
                     "--dtype", "fp32", "--max-new-tokens", "3", "--json"])
    assert rc == 0
    rec = json.loads(capsys.readouterr().out)
    assert rec["width"] == 16 and rec["trained_widths"] == [8, 16, 32, 64]


def test_infer_cli_help_subprocess():
    out = subprocess.run([sys.executable, "-m", "cancerflex.infer", "--help"], capture_output=True, text=True)
    assert out.returncode == 0 and "--checkpoint" in out.stdout
