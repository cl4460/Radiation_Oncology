#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


def _safe_float(x, default=None):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def read_metrics_from_ckpt(ckpt_path: Path) -> Dict[str, Optional[float]]:
    """
    Try multiple key conventions across different scripts:
    - val_uno / val_brier_24m
    - uno_integral / brier_24m
    - best_uno / best_brier_24m
    - metrics dict variants
    """
    if not ckpt_path.exists():
        return {"uno": None, "brier_24m": None}

    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # direct keys
    uno = (
        ckpt.get("val_uno", None)
        or ckpt.get("uno_integral", None)
        or ckpt.get("best_uno", None)
        or ckpt.get("uno", None)
    )
    brier = (
        ckpt.get("val_brier_24m", None)
        or ckpt.get("brier_24m", None)
        or ckpt.get("best_brier_24m", None)
    )

    # nested dict fallback
    if uno is None and isinstance(ckpt.get("metrics", None), dict):
        m = ckpt["metrics"]
        uno = m.get("val_uno", None) or m.get("uno_integral", None) or m.get("uno", None)
        brier = m.get("brier_24m", None) or m.get("val_brier_24m", None)

    return {"uno": _safe_float(uno, None), "brier_24m": _safe_float(brier, None)}


def read_metrics_from_selection_json(sel_path: Path) -> Optional[Dict[str, Any]]:
    """
    Expect keys like:
      img_uno, imgclin_uno, img_brier_24m, imgclin_brier_24m
    """
    if not sel_path.exists():
        return None
    try:
        d = json.loads(sel_path.read_text())
        # accept multiple naming styles
        def _get_any(*keys):
            for k in keys:
                if k in d:
                    return d[k]
            return None

        out = {
            "img_uno": _safe_float(_get_any("img_uno", "uno_img", "model_img_uno"), None),
            "imgclin_uno": _safe_float(_get_any("imgclin_uno", "uno_imgclin", "model_imgclin_uno"), None),
            "img_brier_24m": _safe_float(_get_any("img_brier_24m", "brier_img_24m"), None),
            "imgclin_brier_24m": _safe_float(_get_any("imgclin_brier_24m", "brier_imgclin_24m"), None),
        }
        # must have at least both unos
        if out["img_uno"] is None or out["imgclin_uno"] is None:
            return None
        return out
    except Exception:
        return None


def choose_winner(
    uno_img: float,
    uno_imgclin: float,
    brier_img: Optional[float],
    brier_imgclin: Optional[float],
    margin: float,
    brier_margin: Optional[float],
) -> str:
    """
    Rule:
      - pick imgclin only if it beats img by >= margin
      - and (optional) brier is not worse by more than brier_margin
    """
    if uno_imgclin >= uno_img + margin:
        if brier_margin is not None and brier_img is not None and brier_imgclin is not None:
            if brier_imgclin <= brier_img + brier_margin:
                return "model_imgclin"
            else:
                return "model_img"
        return "model_imgclin"
    return "model_img"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_root", required=True, type=str)
    ap.add_argument("--margin", default=0.02, type=float)
    ap.add_argument("--brier_margin", default=None, type=float,
                    help="Optional: allow imgclin brier_24m to be worse than img by at most this amount.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    root = Path(args.ckpt_root)
    folds = sorted([p for p in root.glob("fold_*") if p.is_dir()])

    if not folds:
        raise SystemExit(f"No fold_* dirs found under: {root}")

    summary = {}

    for fold_dir in folds:
        sel_json = fold_dir / "selection.json"

        sel = read_metrics_from_selection_json(sel_json)

        if sel is not None:
            uno_img = sel["img_uno"]
            uno_imgclin = sel["imgclin_uno"]
            brier_img = sel["img_brier_24m"]
            brier_imgclin = sel["imgclin_brier_24m"]
            source = "selection.json"
        else:
            # fallback to ckpt
            m_img = read_metrics_from_ckpt(fold_dir / "model_img" / "best.pt")
            m_imgclin = read_metrics_from_ckpt(fold_dir / "model_imgclin" / "best.pt")
            uno_img = m_img["uno"]
            uno_imgclin = m_imgclin["uno"]
            brier_img = m_img["brier_24m"]
            brier_imgclin = m_imgclin["brier_24m"]
            source = "best.pt"

        if uno_img is None or uno_imgclin is None:
            print(f"{fold_dir.name}: cannot read metrics from {source}, skipping.")
            continue

        winner = choose_winner(
            uno_img=uno_img,
            uno_imgclin=uno_imgclin,
            brier_img=brier_img,
            brier_imgclin=brier_imgclin,
            margin=args.margin,
            brier_margin=args.brier_margin,
        )

        print(f"{fold_dir.name}: source={source} | "
              f"uno_img={uno_img:.4f} uno_imgclin={uno_imgclin:.4f} | "
              f"brier_img={brier_img if brier_img is not None else 'NA'} "
              f"brier_imgclin={brier_imgclin if brier_imgclin is not None else 'NA'} "
              f"=> winner={winner}")

        # Copy winner to fold_dir/best.pt (back up if exists)
        winner_pt = fold_dir / winner / "best.pt"
        dst = fold_dir / "best.pt"
        if not args.dry_run:
            if dst.exists():
                bak = fold_dir / "best.pt.bak"
                shutil.copy2(dst, bak)
            shutil.copy2(winner_pt, dst)

        summary[fold_dir.name] = {
            "source": source,
            "uno_img": uno_img,
            "uno_imgclin": uno_imgclin,
            "brier_img_24m": brier_img,
            "brier_imgclin_24m": brier_imgclin,
            "winner": winner,
        }

    out_path = root / f"donoharm_selection_margin_v2.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved selection summary: {out_path}")
    if args.dry_run:
        print("dry_run=True, no files copied.")


if __name__ == "__main__":
    main()
