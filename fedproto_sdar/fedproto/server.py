"""
FedProto server — orchestrates prototype aggregation and SDAR attack.

The server:
    1. Receives per-class prototypes from each client
    2. Aggregates them into global prototypes (standard FedProto)
    3. If attack enabled: runs SDAR attack training and inference
    4. Sends global prototypes back to clients
"""

import copy
import sys
from pathlib import Path

root = Path(__file__).parent.parent.resolve()
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from fedproto.proto_utils import agg_func, proto_aggregation


class FedProtoServer:
    """
    FedProto server with optional SDAR attack.

    Attributes:
        global_protos: dict {label: [proto_tensor]} — current global prototypes
        attacker: SDARAttackerFedProto instance (None if attack disabled)
    """

    def __init__(self, args, attacker=None):
        self.args = args
        self.global_protos = []  # empty list = no protos yet (first round)
        self.attacker = attacker

    def receive_and_aggregate(self, local_protos, round_num, raw_protos=None,
                              smashed_data=None):
        """
        Process prototypes from all clients.

        Args:
            local_protos: dict {client_idx: {label: proto_tensor}}
                Per-class aggregated prototypes from each client.
            round_num: current round number.
            raw_protos: optional dict {client_idx: {label: [proto1, proto2, ...]}}
                Individual per-image prototypes (no-averaging ablation).
                If provided, the attacker trains on these instead of local_protos.
                FedProto aggregation always uses local_protos (averaged).
            smashed_data: optional dict {client_idx: {label: [smashed1, ...]}}
                Per-image intermediate features for hybrid attack.

        Returns:
            global_protos: dict {label: [proto_tensor]}
                Updated global prototypes to send back to clients.
            attack_results: dict with attack info (or empty if no attack)
        """
        # ── Step 1: Aggregate prototypes across clients ──
        # Always uses averaged local_protos — FedProto math is unchanged.
        self.global_protos = proto_aggregation(local_protos)

        # ── Step 2: Run SDAR attack if enabled ──
        attack_results = {}
        if self.attacker is not None:
            # Send raw_protos (and smashed_data if hybrid mode) to attacker
            protos_to_attack = raw_protos if raw_protos is not None else local_protos
            round_log = self.attacker.train_attack_round(
                protos_to_attack, round_num, smashed_data=smashed_data, global_protos=self.global_protos)
            attack_results['train_log'] = round_log

            # Perform attack inference on a subset of clients
            attack_results['reconstructions'] = {}
            for client_idx in local_protos.keys():
                recon = self.attacker.attack(
                    local_protos[client_idx], client_idx=client_idx)
                attack_results['reconstructions'][client_idx] = recon

        return self.global_protos, attack_results

    def get_global_protos(self):
        """Return current global prototypes."""
        return self.global_protos
