from __future__ import annotations

import argparse
from pathlib import Path

from ..config import AnnulusConfig, PipelineConfig, ProjectPaths, TileConfig
from ..pipeline import run_all, run_annuli, run_fits, run_ratios, run_tiles


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    if args.repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    else:
        repo_root = Path(args.repo_root).resolve()
    assets_dir = Path(args.assets_dir).resolve() if args.assets_dir is not None else (repo_root / "assets")
    output_dir = Path(args.output_dir).resolve() if args.output_dir is not None else (repo_root / "output")
    paths = ProjectPaths(
        root=repo_root,
        assets_dir=assets_dir,
        output_dir=output_dir,
    )
    annuli_cfg = AnnulusConfig(
        sag_a_reference=(args.sag_x, args.sag_y),
        n_annuli=args.n_annuli,
        min_stars=args.min_stars,
    )
    tiles_cfg = TileConfig(
        n_tiles=args.n_tiles,
        bins_per_tile=args.bins_per_tile,
        max_y_err=args.max_y_err,
    )
    return PipelineConfig(paths=paths, annuli=annuli_cfg, tiles=tiles_cfg, run_id=args.run_id)


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--repo-root", default=None, help="Repo root (default: package root)")
    p.add_argument("--assets-dir", default=None, help="Input assets directory (default: <repo-root>/assets)")
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory",
    )
    p.add_argument("--sag-x", type=float, default=0.0)
    p.add_argument("--sag-y", type=float, default=20.0)
    p.add_argument(
        "--n-annuli",
        type=int,
        default=None,
        help="Equal-area annuli count (omit to use min-stars radial growth)",
    )
    p.add_argument("--min-stars", type=int, default=1000)
    p.add_argument("--n-tiles", type=int, default=10)
    p.add_argument("--bins-per-tile", type=int, default=50)
    p.add_argument("--max-y-err", type=float, default=0.20)
    p.add_argument("--plot", action="store_true", help="Write plots")
    p.add_argument("--run-id", required=True, help="Run identifier (e.g., 14)")


def main() -> None:
    parser = argparse.ArgumentParser(prog="jwst-extinction")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_all = sub.add_parser("all", help="Run annuli -> tiles -> fits -> ratios")
    _add_common_args(p_all)

    p_ann = sub.add_parser("annuli", help="Generate annuli around Sgr A*")
    _add_common_args(p_ann)

    p_tiles = sub.add_parser("tiles", help="Generate tiles for all annuli")
    _add_common_args(p_tiles)

    p_fits = sub.add_parser("fits", help="Fit slopes for all annuli")
    _add_common_args(p_fits)

    p_ratios = sub.add_parser("ratios", help="Compute ratios for all annuli")
    _add_common_args(p_ratios)

    p_plots = sub.add_parser("plots", help="Recreate analysis plots from output/")
    _add_common_args(p_plots)
    p_plots.add_argument("--extinction-law", action="store_true", help="Write extinction law plot")
    p_plots.add_argument("--heatmap", action="store_true", help="Write heatmap plot")
    p_plots.add_argument("--significance", action="store_true", help="Write significance plot")
    p_plots.add_argument("--adjust-annuli0", action="store_true", help="Apply Annuli_0 offsets (legacy)")
    p_plots.add_argument("--alpha", type=float, default=0.05, help="Heatmap significance alpha")
    p_plots.add_argument("--sigma-thresh", type=float, default=3.0, help="Significance sigma threshold")


    args = parser.parse_args()
    cfg = _build_config(args)

    if args.cmd == "all":
        run_all(cfg, plot=args.plot)
    elif args.cmd == "annuli":
        run_annuli(cfg, plot=args.plot)
    elif args.cmd == "tiles":
        run_tiles(cfg, plot=args.plot)
    elif args.cmd == "fits":
        run_fits(cfg, plot=args.plot)
    elif args.cmd == "ratios":
        run_ratios(cfg)
    elif args.cmd == "plots":
        from ..analysis.plots import plot_extinction_law, plot_heatmap, plot_significance

        out_dir = cfg.paths.output_dir
        run_meta = out_dir / "runs" / cfg.run_id / "summary" / "run.json"
        if run_meta.exists():
            try:
                import json
                meta = json.loads(run_meta.read_text())
                eff = meta.get("effective_annuli")
                req = meta.get("requested_annuli")
                if eff is not None and req is not None:
                    print(f"run_id={cfg.run_id} requested_annuli={req} effective_annuli={eff}")
            except Exception:
                pass
        any_flag = args.extinction_law or args.heatmap or args.significance
        if not any_flag:
            args.extinction_law = True
            args.heatmap = True
            args.significance = True
        if args.extinction_law:
            plot_extinction_law(out_dir, cfg.run_id, adjust_annuli0=args.adjust_annuli0)
        if args.heatmap:
            plot_heatmap(out_dir, cfg.run_id, alpha=args.alpha)
        if args.significance:
            plot_significance(out_dir, cfg.run_id, sigma_thresh=args.sigma_thresh)


if __name__ == "__main__":
    main()
