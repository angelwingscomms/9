from __future__ import annotations

from .shared import *  # noqa: F401,F403

def resolve_architecture(args: argparse.Namespace) -> str:
    configured_architecture = str(getattr(args, "config_architecture", "")).strip()
    if configured_architecture:
        return configured_architecture
    if args.use_chronos_bolt:
        return "chronos_bolt"
    if args.use_timesfm:
        return "timesfm"
    if args.gold:
        return "gold_legacy"
    if args.gold_new:
        return "gold_new"
    if args.use_legacy_lstm_attention:
        return "legacy_lstm_attention"
    if args.ela:
        return "ela"
    if args.use_fusion_lstm_encoder:
        return "fusion_lstm"
    if args.use_bilstm_encoder:
        return "bilstm"
    if args.use_gru_encoder:
        return "gru"
    if args.use_tcn_encoder:
        return "tcn"
    if args.use_tla_encoder:
        return "tla"
    if args.use_tkan_encoder:
        return "tkan"
    if args.use_minirocket_encoder:
        return "minirocket"
    if args.use_castor_encoder:
        return "castor"
    return "mamba"
