import os
import json
import argparse
import csv
import traceback
import sys

# ensure repo root is on sys.path so 'evaluation' package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluation.meteor.meteor import Meteor


def load_test_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle structure like {"results": [ {id: [..], gens: {...}, gts: {...}}, ... ] }
    if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list):
        gts = {}
        res = {}
        for entry in data['results']:
            # id may be list or scalar
            _id = entry.get('id')
            if isinstance(_id, list) and _id:
                key = str(_id[0])
            else:
                key = str(_id)

            # gens: take first value
            gens = entry.get('gens', {})
            preds = list(gens.values()) if isinstance(gens, dict) else []
            pred = preds[0] if preds else ''

            # gts: may have multiple
            gts_map = entry.get('gts', {})
            refs = list(gts_map.values()) if isinstance(gts_map, dict) else []
            refs = [r for r in refs if isinstance(r, str)]

            res[key] = [pred]
            gts[key] = refs if refs else ['']

        return gts, res

    # If format is already a mapping: {id: {"gts": [...], "preds": [...]}}
    if isinstance(data, dict):
        # try to detect mapping of ids
        gts = {}
        res = {}
        for k, v in data.items():
            if isinstance(v, dict):
                if 'gts' in v:
                    gts[k] = v['gts'] if isinstance(v['gts'], list) else [v['gts']]
                if 'preds' in v:
                    res[k] = v['preds'] if isinstance(v['preds'], list) else [v['preds']]
        if gts and res:
            return gts, res

    raise ValueError(f"Unrecognized test_results.json format: {path}")


def main(saved_models_dir, out_csv):
    models = sorted([d for d in os.listdir(saved_models_dir) if os.path.isdir(os.path.join(saved_models_dir, d))])
    meteor = None

    summary_rows = []

    for m in models:
        folder = os.path.join(saved_models_dir, m)
        test_path = os.path.join(folder, 'test_results.json')
        if not os.path.isfile(test_path):
            continue

        try:
            gts, res = load_test_results(test_path)
        except Exception as e:
            summary_rows.append((m, 'ERROR', str(e)))
            continue

        try:
            if meteor is None:
                meteor = Meteor()

            overall, per_example = meteor.compute_score(gts, res)

            # Save per-model meteor results
            outp = os.path.join(folder, 'meteor_recomputed.json')
            with open(outp, 'w', encoding='utf-8') as f:
                json.dump({'overall': overall, 'per_example': per_example}, f, ensure_ascii=False, indent=2)

            summary_rows.append((m, 'OK', overall))
        except Exception as e:
            tb = traceback.format_exc()
            summary_rows.append((m, 'ERROR', tb))

    # write CSV summary
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'status', 'overall_or_error'])
        for r in summary_rows:
            writer.writerow(r)

    print('Wrote summary to', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recompute METEOR for saved_models')
    parser.add_argument('--saved_models', default='saved_models', help='Saved models dir')
    parser.add_argument('--out', default=os.path.join('saved_models', 'meteor_summary.csv'), help='Output CSV')
    args = parser.parse_args()
    main(args.saved_models, args.out)
