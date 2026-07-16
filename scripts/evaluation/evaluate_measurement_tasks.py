from __future__ import annotations
import json,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from src.vision.measurement_tasks import measure_normal,measure_numbered

DATASETS=(
 (ROOT/"assets/targets/generated/modular/basic","shapes","normal"),
 (ROOT/"assets/targets/generated/modular/separate","shapes","normal"),
 (ROOT/"assets/targets/generated/overlap_controlled","squares","normal"),
 (ROOT/"assets/targets/generated/numbered_controlled","squares","numbered"),)

def main()->int:
    failures=0; normal_total=normal_ok=number_total=number_ok=0
    for directory,key,kind in DATASETS:
        for image_path in sorted(directory.glob("*.png")):
            image=cv2.imread(str(image_path))
            # 普通任务先只读取 PNG；任务返回后才读取 JSON 评分。
            normal=measure_normal(image)
            truth=json.loads(image_path.with_suffix(".json").read_text(encoding="utf-8"))[key]
            expected=min(float(item["size_mm"]) for item in truth)
            error=abs(normal.x_mm-expected); ok=error<=1.0
            normal_total+=1;normal_ok+=int(ok);failures+=int(not ok)
            print(f"{'PASS' if ok else 'FAIL'} normal {image_path.name} x={normal.x_mm:.2f} truth={expected:.2f} error={error:.2f}")
            if kind!="numbered": continue
            # JSON 仅提供题目保证存在的 IO 输入数字；几何参数不传入任务。
            for item in truth:
                requested=int(item["digit"]); result=measure_numbered(image,requested)
                error=abs(result.x_mm-float(item["size_mm"])); ok=error<=1.0
                number_total+=1;number_ok+=int(ok);failures+=int(not ok)
                print(f"{'PASS' if ok else 'FAIL'} numbered {image_path.name} digit={requested} x={result.x_mm:.2f} truth={item['size_mm']:.2f} error={error:.2f}")
    print(f"SUMMARY normal={normal_ok}/{normal_total} numbered={number_ok}/{number_total}")
    return 1 if failures else 0
if __name__=="__main__": raise SystemExit(main())
