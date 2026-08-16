from __future__ import annotations
import csv, json, re
from pathlib import Path

import matplotlib.pyplot as plt

out_dir = Path(r'E:\CODEXRESEARCH\RESEARCH-1\artifacts\benchmark_runs\language\plots_20260615')
out_dir.mkdir(parents=True, exist_ok=True)

log_paths = [
    Path(r'E:\CODEXRESEARCH\house_compute_hub\runs\20260614-223723-f1c35a\mwstroud-mwstr-6aea1cf3.log'),
    Path(r'E:\CODEXRESEARCH\house_compute_hub\runs\20260614-235439-ba54d5\mwstroud-mwstr-6aea1cf3.log'),
    Path(r'E:\CODEXRESEARCH\house_compute_hub\runs\20260615-133210-8023da\mwstroud-mwstr-6aea1cf3.log'),
    Path(r'E:\CODEXRESEARCH\house_compute_hub\runs\20260615-141307-266267\mwstroud-mwstr-6aea1cf3.log'),
]
# Include any older hub log mentioning the source 76M lowrank run.
for p in Path(r'E:\CODEXRESEARCH\house_compute_hub\runs').rglob('*.log'):
    try:
        text = p.read_text(encoding='utf-8', errors='replace')
    except OSError:
        continue
    if 'wave10_3080_lowrank_conv_memory_76m_3b_scratch_existingcache_20260605' in text:
        log_paths.append(p)

seen_paths = []
seen = set()
for p in log_paths:
    if p.exists() and p not in seen:
        seen.add(p)
        seen_paths.append(p)

train_rows = []
eval_rows = []
notes = []
train_re = re.compile(r'TRAIN step=(\d+)/(\d+) tokens=(\d+) loss=([0-9.]+) lr=([0-9.eE+-]+)(?: pure_tok_s=([0-9.]+))?')
eval_re = re.compile(r'EVAL step=(\d+)/(\d+) tokens=(\d+) train=([0-9.]+) val=([0-9.]+)')
resume_re = re.compile(r'RESUME_META (\{.*\})')
ckpt_re = re.compile(r'CKPT_META (\{.*\})')
result_re = re.compile(r'RESULT (\{.*\})')

for path in seen_paths:
    text = path.read_text(encoding='utf-8', errors='replace')
    source = str(path)
    for line in text.splitlines():
        m = train_re.search(line)
        if m:
            step, total, tokens, loss, lr, tok_s = m.groups()
            train_rows.append({
                'source': source, 'kind': 'train_log', 'step': int(step), 'target_step': int(total),
                'tokens_seen': int(tokens), 'tokens_b': int(tokens)/1e9, 'train_loss': float(loss),
                'val_loss': '', 'lr': float(lr), 'tok_s': float(tok_s) if tok_s else '',
            })
            continue
        m = eval_re.search(line)
        if m:
            step, total, tokens, train_loss, val_loss = m.groups()
            eval_rows.append({
                'source': source, 'kind': 'eval_log', 'step': int(step), 'target_step': int(total),
                'tokens_seen': int(tokens), 'tokens_b': int(tokens)/1e9,
                'train_loss': float(train_loss), 'val_loss': float(val_loss), 'lr': '', 'tok_s': '',
            })
            continue
        for regex, label in [(resume_re, 'resume_meta'), (ckpt_re, 'ckpt_meta')]:
            m = regex.search(line)
            if m:
                try:
                    meta = json.loads(m.group(1))
                    notes.append({'source': source, 'kind': label, **meta})
                except Exception:
                    pass
        m = result_re.search(line)
        if m:
            try:
                result = json.loads(m.group(1))
            except Exception:
                result = None
            if isinstance(result, dict):
                for h in result.get('history') or []:
                    if not isinstance(h, dict) or 'tokens_seen' not in h:
                        continue
                    eval_rows.append({
                        'source': source, 'kind': 'result_history', 'step': int(h.get('step', -1)), 'target_step': '',
                        'tokens_seen': int(h['tokens_seen']), 'tokens_b': int(h['tokens_seen'])/1e9,
                        'train_loss': float(h.get('train_loss')) if h.get('train_loss') is not None else '',
                        'val_loss': float(h.get('val_loss')) if h.get('val_loss') is not None else '',
                        'lr': float(h.get('learning_rate')) if h.get('learning_rate') is not None else '', 'tok_s': '',
                    })

# Deduplicate by kind/tokens/loss so repeated dashboard snapshots do not dominate.
def dedupe(rows, fields):
    out = []
    keys = set()
    for row in sorted(rows, key=lambda r: (int(r.get('tokens_seen', -1)), str(r.get('kind')), str(r.get('source')))):
        key = tuple(row.get(f) for f in fields)
        if key in keys:
            continue
        keys.add(key)
        out.append(row)
    return out

train_rows = dedupe(train_rows, ['kind', 'tokens_seen', 'train_loss'])
eval_rows = dedupe(eval_rows, ['kind', 'tokens_seen', 'train_loss', 'val_loss'])
all_rows = sorted(train_rows + eval_rows, key=lambda r: (int(r['tokens_seen']), str(r['kind'])))

csv_path = out_dir / 'wave10_76m_to5b_loss_so_far_20260615.csv'
with csv_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['source','kind','step','target_step','tokens_seen','tokens_b','train_loss','val_loss','lr','tok_s'])
    writer.writeheader()
    writer.writerows(all_rows)

notes_path = out_dir / 'wave10_76m_to5b_loss_so_far_notes_20260615.json'
notes_path.write_text(json.dumps({'logs_used': [str(p) for p in seen_paths], 'notes': notes}, indent=2), encoding='utf-8')

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
if train_rows:
    xs = [r['tokens_b'] for r in train_rows]
    ys = [r['train_loss'] for r in train_rows]
    ax.plot(xs, ys, color='#4C78A8', linewidth=1.1, alpha=0.45, label='train loss (logged every 500 steps)')
if eval_rows:
    ev_val = [r for r in eval_rows if r['val_loss'] != '']
    ev_train = [r for r in eval_rows if r['train_loss'] != '']
    if ev_train:
        ax.scatter([r['tokens_b'] for r in ev_train], [r['train_loss'] for r in ev_train], s=22, color='#72B7B2', label='eval-point train loss', zorder=3)
    if ev_val:
        ax.plot([r['tokens_b'] for r in ev_val], [r['val_loss'] for r in ev_val], color='#E45756', marker='o', markersize=4, linewidth=2.0, label='validation loss', zorder=4)

# Mark resume/source boundaries when present.
for token_b, label in [(2.20010736, '2.20B source ckpt'), (2.88544, '2.885B restart ckpt'), (5.00000016, '5B target')]:
    ax.axvline(token_b, color='#999999', linewidth=0.9, linestyle='--', alpha=0.6)
    ymin, ymax = ax.get_ylim()
    ax.text(token_b, ymax, label, rotation=90, va='top', ha='right', fontsize=8, color='#555555')

ax.set_title('Wave10 76M low-rank conv-memory loss over tokens so far')
ax.set_xlabel('Tokens seen (billions)')
ax.set_ylabel('Loss')
ax.set_xlim(left=max(0, min([r['tokens_b'] for r in all_rows], default=0) - 0.05), right=5.05)
if all_rows:
    vals = []
    for r in all_rows:
        if r['train_loss'] != '': vals.append(float(r['train_loss']))
        if r['val_loss'] != '': vals.append(float(r['val_loss']))
    if vals:
        ax.set_ylim(max(3.2, min(vals)-0.2), min(5.4, max(vals)+0.2))
ax.legend(loc='upper right')
fig.tight_layout()
png_path = out_dir / 'wave10_76m_to5b_loss_so_far_20260615.png'
fig.savefig(png_path)

summary = {
    'png': str(png_path),
    'csv': str(csv_path),
    'notes': str(notes_path),
    'train_points': len(train_rows),
    'eval_points': len(eval_rows),
    'min_token_b': min([r['tokens_b'] for r in all_rows], default=None),
    'max_token_b': max([r['tokens_b'] for r in all_rows], default=None),
    'latest_train': max(train_rows, key=lambda r: r['tokens_seen']) if train_rows else None,
    'latest_eval': max([r for r in eval_rows if r['val_loss'] != ''], key=lambda r: r['tokens_seen']) if any(r['val_loss'] != '' for r in eval_rows) else None,
    'logs_used': [str(p) for p in seen_paths],
}
print(json.dumps(summary, indent=2))
