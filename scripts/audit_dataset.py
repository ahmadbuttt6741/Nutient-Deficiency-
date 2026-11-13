import os
from collections import defaultdict

stats = {'train': defaultdict(int), 'val': defaultdict(int), 'test': defaultdict(int)}
total_imgs = 0
for split in ['train', 'val', 'test']:
    p = os.path.join('data', 'processed', split)
    if os.path.exists(p):
        for cls_name in sorted(os.listdir(p)):
            cls_p = os.path.join(p, cls_name)
            if os.path.isdir(cls_p):
                cnt = len([f for f in os.listdir(cls_p) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))])
                stats[split][cls_name] = cnt
                total_imgs += cnt

print(f"Total Processed Images Across Splits: {total_imgs}\n")
all_classes = sorted(list(stats['train'].keys()))
header = f"{'Class Name':25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}"
print(header)
print("-" * len(header))
for c in all_classes:
    tr = stats['train'].get(c, 0)
    va = stats['val'].get(c, 0)
    te = stats['test'].get(c, 0)
    print(f"{c:25} {tr:>8} {va:>8} {te:>8} {tr+va+te:>8}")
print("-" * len(header))
tr_tot = sum(stats['train'].values())
va_tot = sum(stats['val'].values())
te_tot = sum(stats['test'].values())
print(f"{'TOTAL':25} {tr_tot:>8} {va_tot:>8} {te_tot:>8} {tr_tot+va_tot+te_tot:>8}")
