from __future__ import annotations

import argparse

from cfgflow.app import run_app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cfgflow")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run local desktop app")
    run.add_argument("--sumocfg", required=True, help="Path to .sumocfg (e.g. scenario\\due.actuated.sumocfg)")
    run.add_argument("--net", required=True, help="Path to net.xml (e.g. scenario\\lust.net.xml)")
    run.add_argument("--sumo-binary", default="sumo-gui", help="sumo or sumo-gui")
    run.add_argument("--port", type=int, default=8813, help="TraCI port")
    run.add_argument("--step-length", type=float, default=1.0, help="Simulation step length (seconds)")
    run.add_argument(
        "--publish-every-steps", type=int, default=5, help="Publish UI update every N steps"
    )
    run.add_argument("--sqlite", default="", help="Optional path to sqlite db for recording (empty = disabled)")
    run.add_argument("--host", default="127.0.0.1", help="Bind host")
    run.add_argument("--ui-port", type=int, default=8088, help="UI port")
    run.add_argument("--native", action="store_true", help="Run in a desktop window (pywebview)")
    run.add_argument("--model", default="", help="Optional path to a trained torch model checkpoint (.pt)")

    doctor = sub.add_parser("doctor", help="Check environment and inputs")
    doctor.add_argument("--sumocfg", required=True)
    doctor.add_argument("--net", required=True)
    doctor.add_argument("--sumo-binary", default="sumo-gui")

    export = sub.add_parser("export", help="Export sqlite recording to CSV")
    export.add_argument("--sqlite", required=True)
    export.add_argument("--out", required=True)
    export.add_argument("--limit", type=int, default=0)

    train = sub.add_parser("train", help="Train a local spatiotemporal model (requires torch)")
    train.add_argument("--net", required=True)
    train.add_argument("--sqlite", required=True)
    train.add_argument("--out", required=True, help="Output checkpoint path (e.g. models\\model.pt)")
    train.add_argument("--max-edges", type=int, default=1200)
    train.add_argument("--t-in", type=int, default=12, help="Input sequence length in steps")
    train.add_argument("--t-out", type=int, default=3, help="Output horizon length in steps")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--batch", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--device", default="cpu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "run":
        run_app(
            sumocfg=args.sumocfg,
            net_path=args.net,
            sumo_binary=args.sumo_binary,
            traci_port=args.port,
            step_length_s=args.step_length,
            publish_every_steps=args.publish_every_steps,
            sqlite_path=args.sqlite,
            host=args.host,
            ui_port=args.ui_port,
            native=args.native,
            model_path=args.model,
        )
    elif args.cmd == "doctor":
        from cfgflow.cli.doctor import run_doctor

        run_doctor(sumocfg=args.sumocfg, net=args.net, sumo_binary=args.sumo_binary)
    elif args.cmd == "export":
        from cfgflow.cli.export import export_sqlite_to_csv

        export_sqlite_to_csv(sqlite_path=args.sqlite, out_csv=args.out, limit=args.limit)
    elif args.cmd == "train":
        from cfgflow.ml.train import train_model

        train_model(
            net_path=args.net,
            sqlite_path=args.sqlite,
            out_path=args.out,
            max_edges=args.max_edges,
            t_in=args.t_in,
            t_out=args.t_out,
            epochs=args.epochs,
            batch_size=args.batch,
            lr=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()
