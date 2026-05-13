import json, random
from collections import Counter

with open(r'D:\mWork\工作\小论文v2\code\data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

random.seed(42)
samples = []
stages_seen = set()
for d in data:
    stage = int(d['Fibrosis_Stage_0_4'])
    if stage not in stages_seen or len(samples) < 8:
        if stage not in stages_seen:
            stages_seen.add(stage)
            samples.append(d)
        elif len(samples) < 8:
            samples.append(d)
    if len(samples) >= 8 and len(stages_seen) >= 5:
        break

deid = []
for d in samples[:8]:
    deid.append({
        'sample_id': f'SAMPLE_{len(deid)+1:02d}',
        'fibrosis_stage': int(d['Fibrosis_Stage_0_4']),
        'inflammation_grade': int(d['Inflammation_Grade_0_4']),
        'steatosis_grade': int(d['Steatosis_Grade_1_3']),
        'pathology_text_with_diagnosis': d['病理描述_有诊断'],
        'pathology_text_without_diagnosis': d['病理描述_删除诊断'],
    })

out_path = r'D:\mWork\工作\小论文v2\code\github用\sample_data.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(deid, f, ensure_ascii=False, indent=2)

print(f'Created {len(deid)} de-identified samples')
c = Counter(d['fibrosis_stage'] for d in deid)
print(f'Stage distribution: {dict(sorted(c.items()))}')
print(f'Saved to: {out_path}')
