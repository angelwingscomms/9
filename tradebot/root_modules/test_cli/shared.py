"""Backtest one archived model folder through the MT5 daily tester flow."""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from tradebot.root_modules.mt5_runtime import (
    PROJECT_DIR_NAME,
    iter_agent_log_paths,
    launch_terminal as launch_mt5_terminal,
    resolve_mt5_runtime,
)
from tradebot.config_io import load_define_file, read_text_best_effort, sanitize_symbol, warn_missing_config_keys
from tradebot.workspace import (
    ACTIVE_DIAGNOSTICS_DIR,
    ACTIVE_CONFIG_PATH,
    activate_model,
    compile_live_expert,
    configured_symbol,
    ensure_default_test_config,
    format_model_stamp,
    load_test_config,
    model_config_path,
    model_tests_dir,
    resolve_model_dir,
)


SUMMARY_PATTERN = re.compile(r"\[SUMMARY\]\s+(.*)")
NUMBER_PATTERN = re.compile(r"^-?\d+(?:\.\d+)?$")
BOOL_PATTERN = re.compile(r"^(true|false)$", re.IGNORECASE)
INITIAL_DEPOSIT_PATTERN = re.compile(r"initial deposit\s+(-?\d+(?:\.\d+)?)\s+([A-Z]+), leverage\s+(.+)$")
FINAL_BALANCE_PATTERN = re.compile(r"final balance\s+(-?\d+(?:\.\d+)?)\s+([A-Z]+)$")
GENERATED_PATTERN = re.compile(
    r":\s+(\d+)\s+ticks,\s+(\d+)\s+bars generated\..*?Test passed in\s+([0-9:.]+)",
    re.IGNORECASE,
)
STATUS_LINE_PATTERN = re.compile(r"testing of .* from (\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}) to (\d{4}\.\d{2}\.\d{2} \d{2}:\d{2})")

SCRIPT_DIR = Path(__file__).resolve().parents[3]
CONFIG_PATH = ACTIVE_CONFIG_PATH
