from __future__ import annotations

from .shared import *  # noqa: F401,F403
from tradebot.config_io import load_define_file, warn_missing_config_keys

def main() -> bool:
    args = parse_args()
    print("=" * 60)
    print("MT5 Data Export Tool")
    print("=" * 60)
    print()

    try:
        print("[STEP 1] Resolving symbol configuration...")
        symbol, config_path = resolve_symbol_config(args.symbol)
        cfg_values = load_define_file(config_path)
        warn_missing_config_keys(cfg_values, source_label=str(config_path))
        print(f"[STEP 1] SUCCESS - Symbol: {symbol}")
        print(f"[STEP 1] Config: {config_path}")
        print()

        print("[STEP 2] Resolving MT5 runtime...")
        runtime = resolve_mt5_runtime(
            instance_root_override=args.instance_root,
            terminal_path_override=args.terminal_path,
            metaeditor_path_override=args.metaeditor_path,
        )
        print(f"[STEP 2] SUCCESS - Terminal: {runtime.terminal_path}")
        print(f"[STEP 2] SUCCESS - MetaEditor: {runtime.metaeditor_path}")
        print()

        if config_path != ACTIVE_CONFIG_PATH:
            shutil.copy2(config_path, ACTIVE_CONFIG_PATH)

        print("[STEP 3] Compiling data.mq5...")
        compiled_path = compile_data_script(runtime, config_path, profile=args.profile)
        print(f"[STEP 3] SUCCESS - Compiled: {compiled_path}")
        print()

        print(f"[STEP 4] Launching MT5 terminal with {profile_script_name(args.profile)}.mq5...")
        stale_output = runtime.files_dir / args.output_file
        stale_output.unlink(missing_ok=True)
        run_script(runtime, symbol, profile=args.profile)
        print("[STEP 4] SUCCESS")
        print()

        print(f"[STEP 5] Waiting for script output (up to {args.timeout_seconds} seconds)...")
        csv_path = find_csv_file(runtime.files_dir, args.output_file, max_wait=float(args.timeout_seconds))
        if csv_path is None:
            print("[STEP 5] FAILED - Timeout waiting for CSV file")
            print("[INFO] Possible issues:")
            print(f"  - MT5 Files folder: {runtime.files_dir}")
            print("  - Check MT5 Experts and Tester logs for errors")
            print("  - Symbol may not have tick history downloaded")
            return False
        print("[STEP 5] SUCCESS")
        print()

        print("[STEP 6] Moving file to data directory...")
        move_to_data_dir(symbol, csv_path)
        print("[STEP 6] SUCCESS")
        print()

        print("=" * 60)
        print("Export completed successfully!")
        print("=" * 60)
        return True
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return False
