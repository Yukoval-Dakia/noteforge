"""Plans a song before drafting: density curve (for draft_song --density-curve) and suggested style shares.

  plan_song.py --features-dir DIR --audio-sha SHA --mode phira --density 6 --out curve.json
Omit --density to let the planner guess the overall level too (weaker on the frozen validation panel).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parents[2])]
from noteforge.planner import extract
from noteforge.planner.model import EPS, Planner

STYLE = ('hold_ratio', 'chord_ratio', 'flick_ratio', 'drag_ratio')


def plan(cells, is_phira, density, net, mean, std, device):
    x = torch.from_numpy(((cells-mean)/std).astype(np.float32))[None].to(device)
    mask = torch.ones(1, len(cells), dtype=torch.bool, device=device)
    known = density is not None
    level = torch.tensor([float(np.log(density+EPS)) if known else 0.], dtype=torch.float32, device=device)
    with torch.no_grad():
        curve, _, style = net(x, mask, torch.tensor([is_phira], device=device), level, torch.tensor([known], device=device))
    d = np.maximum(np.exp(curve[0].cpu().numpy())-EPS, 0.)
    if known:
        d = d*density/max(float(d.mean()), 1e-3)
    return d, dict(zip(STYLE, torch.sigmoid(style[0]).cpu().tolist()))


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--features-dir', type=Path, required=True)
    a.add_argument('--audio-sha', required=True)
    a.add_argument('--mode', choices=('mania_4k', 'phira'), required=True)
    a.add_argument('--density', type=float, help='requested overall notes per second')
    a.add_argument('--planner-dir', type=Path, default=Path('planner'))
    a.add_argument('--tag', default='f')
    a.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a.add_argument('--out', type=Path, required=True)
    args = a.parse_args()
    d = args.planner_dir
    saved = torch.load(d/f'planner_{args.tag}_{args.mode}.pt', map_location=args.device, weights_only=False)
    cfg = saved['cfg']
    net = Planner(width=cfg['width'], layers=cfg['layers'], dropout=cfg['dropout']).to(args.device)
    net.load_state_dict(saved['model'])
    net.eval()
    pca, norm = np.load(d/'pca.npz'), np.load(d/'norm.npz')
    extract.FEATURES = args.features_dir
    mert, extra = extract.song_cells(args.audio_sha)
    cells = np.concatenate([(mert-pca['mean'])@pca['basis'].T, extra], 1)
    density, style = plan(cells, args.mode == 'phira', args.density, net, norm['mean'], norm['std'], args.device)
    segments = [dict(start_s=2.*k, end_s=2.*(k+1), density=float(v)) for k, v in enumerate(density)]
    args.out.write_text(json.dumps(segments))
    print(json.dumps(dict(cells=len(segments), mean_density=round(float(density.mean()), 2), suggested_style={k: round(v, 3) for k, v in style.items()})))


if __name__ == '__main__':
    main()
