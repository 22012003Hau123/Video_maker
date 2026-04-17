"""
Verify Smart Assembly — Chạy pipeline cho nhiều thời lượng (15s, 10s, 6s) 
để verify logic "Cắt thông minh".
"""
import sys, os, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from src.mastering.gold_render import render_gold_job
from src.mastering.gold_manifest import GoldJobSpec

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_BASE = PROJECT_ROOT / "outputs" / "verify_assembly"
if WORK_BASE.exists():
    shutil.rmtree(WORK_BASE)
WORK_BASE.mkdir(parents=True)

def test_duration(dur):
    print(f"\n{'#'*60}")
    print(f"  TESTING DURATION: {dur}s")
    print(f"{'#'*60}")
    
    out_dir = WORK_BASE / f"dur_{dur}"
    out_dir.mkdir()
    
    spec = GoldJobSpec(
        vo="vo1",
        line_id="duo_branded_sub",
        line_label="VO1 Duo Branded Sub",
        layout="duo",
        branded=True,
        subtitles=True,
        video_format="16x9",
        duration_seconds=dur,
        export_codec_id="mp4_h264_1pass_draft",
        output_basename_hint=f"BurberryGold_{dur}s_Smart"
    )
    
    def status_cb(msg):
        print(f"  [{dur}s] {msg}")

    final_out, findings = render_gold_job(
        spec,
        PROJECT_ROOT,
        out_dir,
        status_callback=status_cb
    )
    
    print(f"  ✅ DONE: {final_out}")
    print(f"  🧠 Findings: {findings.get('subtitle_sync', {}).get('status', 'N/A')}")

# Test the critical four
for d in [15, 10, 6, 5]:
    try:
        test_duration(d)
    except Exception as e:
        print(f"  ❌ FAILED {d}s: {e}")

print(f"\nAll tests completed in {WORK_BASE}")
