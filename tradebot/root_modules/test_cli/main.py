from __future__ import annotations

from .shared import *  # noqa: F401,F403

def main() -> None:
    args = parse_args()
    symbol = args.symbol or configured_symbol()
    model_dir = resolve_model_dir(symbol, args.revision)
    test_config = merged_test_config(args, default_symbol=symbol, model_dir=model_dir)
    runtime = resolve_mt5_runtime(
        instance_root_override=args.instance_root,
        terminal_path_override=args.terminal_path,
        metaeditor_path_override=args.metaeditor_path,
    )

    activate_model(model_dir)

    daily_mode = args.day is not None
    single_day = None
    if daily_mode:
        single_day = parse_single_day(args.day) if args.day else (date.today() - timedelta(days=1))

    run_stamp = format_model_stamp()
    run_dir_name = f"{run_stamp} d" if daily_mode else run_stamp
    run_dir = model_tests_dir(model_dir) / run_dir_name
    config_dir = run_dir / "configs"
    raw_log_dir = run_dir / "raw_logs"
    compile_dir = run_dir / "compile"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    raw_log_dir.mkdir(parents=True, exist_ok=True)
    compile_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_live_compile:
        compile_log_path = compile_live_expert(runtime, model_dir=model_dir)
        (compile_dir / compile_log_path.name).write_text(
            read_text_best_effort(compile_log_path),
            encoding="utf-8",
        )

    config = load_define_file(model_config_path(model_dir))
    warn_missing_config_keys(config, source_label=str(model_config_path(model_dir)))
    if daily_mode:
        if single_day is None:
            raise RuntimeError("Daily mode failed to resolve a test date.")
        report_scope = single_day.isoformat()
        day_values = [single_day]
    else:
        month_value = str(test_config["month"]) or None
        month_start, month_end = parse_month(month_value)
        report_scope = month_start.strftime("%Y-%m")
        day_values = filter_days(
            iter_days(month_start, month_end),
            str(test_config["from_date"]),
            str(test_config["to_date"]),
        )

    set_name = f"{sanitize_symbol(str(test_config['symbol']))}_daily_backtest_{run_stamp}.set"
    set_path = runtime.tester_profile_dir / set_name
    build_set_file(set_path, config=config)
    (run_dir / "tester_inputs.set").write_text(set_path.read_text(encoding="ascii"), encoding="ascii")
    (run_dir / "config.mqh").write_text(model_config_path(model_dir).read_text(encoding="utf-8"), encoding="utf-8")
    (run_dir / "backtest_config_snapshot.json").write_text(
        json.dumps(test_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    today_stamp = current_log_stamp()
    main_log_path = runtime.terminal_log_dir / f"{today_stamp}.log"

    rows: list[BacktestResult] = []
    for day_value in day_values:
        if not runtime.terminal_path.exists():
            raise FileNotFoundError(f"terminal64.exe not found at {runtime.terminal_path}")

        config_path = config_dir / f"{day_value.isoformat()}.ini"
        launch_config_dir = Path(tempfile.gettempdir()) / "mt5_tester_configs"
        launch_config_dir.mkdir(parents=True, exist_ok=True)
        launch_config_path = launch_config_dir / f"{sanitize_symbol(str(test_config['symbol']))}_{day_value.isoformat()}.ini"
        build_ini_file(
            path=config_path,
            set_name=set_name,
            day_value=day_value,
            deposit=float(test_config["deposit"]),
            currency=str(test_config["currency"]),
            leverage=str(test_config["leverage"]),
            symbol=str(test_config["symbol"]),
        )
        launch_config_path.write_text(config_path.read_text(encoding="ascii"), encoding="ascii")

        last_error = ""
        day_result: BacktestResult | None = None
        tester_log_output = raw_log_dir / f"{day_value.isoformat()}_tester.log"
        agent_log_output = raw_log_dir / f"{day_value.isoformat()}_agent.log"
        for _attempt in range(int(test_config["retries"]) + 1):
            tester_offsets = log_offsets([main_log_path])
            agent_log_paths = list(iter_agent_log_paths(runtime))
            agent_offsets = log_offsets(agent_log_paths)
            try:
                launch_mt5_terminal(
                    runtime,
                    launch_config_path,
                    timeout_seconds=int(test_config["timeout_seconds"]),
                    detach=True,
                    stop_existing=True,
                )
                tester_text = wait_for_tester_completion(
                    main_log_path=main_log_path,
                    offset=tester_offsets.get(main_log_path, 0),
                    timeout_seconds=int(test_config["timeout_seconds"]),
                )
                time.sleep(1.0)
                agent_chunks: list[str] = []
                for path in iter_agent_log_paths(runtime):
                    agent_chunks.append(read_appended_text(path, agent_offsets.get(path, 0)))
                agent_text = "\n".join(chunk for chunk in agent_chunks if chunk)

                tester_log_output.write_text(tester_text, encoding="utf-8")
                agent_log_output.write_text(agent_text, encoding="utf-8")

                day_result = parse_result(day_value=day_value, tester_text=tester_text, agent_text=agent_text)
                if day_result.error == "agent_log_missing":
                    raise RuntimeError(day_result.error)
                break
            except Exception as exc:
                last_error = str(exc)
                tester_text = read_appended_text(main_log_path, tester_offsets.get(main_log_path, 0))
                tester_log_output.write_text(tester_text, encoding="utf-8")
                agent_chunks = []
                for path in iter_agent_log_paths(runtime):
                    agent_chunks.append(read_appended_text(path, agent_offsets.get(path, 0)))
                agent_text = "\n".join(chunk for chunk in agent_chunks if chunk)
                agent_log_output.write_text(agent_text, encoding="utf-8")
                time.sleep(2.0)
        if day_result is None:
            day_result = error_result(day_value, last_error or "unknown_error")

        rows.append(day_result)
        write_csv(run_dir / "daily_results.csv", rows)
        write_report(run_dir / "report.md", scope_label=report_scope, rows=rows, daily_mode=daily_mode)

    write_csv(run_dir / "daily_results.csv", rows)
    write_report(run_dir / "report.md", scope_label=report_scope, rows=rows, daily_mode=daily_mode)
    print(run_dir)
