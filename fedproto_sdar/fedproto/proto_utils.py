"""
Prototype aggregation utilities for FedProto.
Ported from FedProto's lib/utils.py — agg_func and proto_aggregation.
"""

import copy
import torch


def agg_func(protos):
    """
    Average prototype embeddings per class for a single client.

    Args:
        protos: dict {label: [proto_tensor, proto_tensor, ...]}
            Each proto_tensor has shape matching the model's prototype output.

    Returns:
        dict {label: averaged_proto_tensor}
    """
    for label, proto_list in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for p in proto_list:
                proto += p.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0].data

    return protos


def proto_aggregation(local_protos_list):
    """
    Aggregate prototypes from all clients into global prototypes.

    Args:
        local_protos_list: dict {client_idx: {label: proto_tensor}}

    Returns:
        dict {label: [averaged_global_proto_tensor]}
            Each value is a list with a single tensor (for compatibility
            with FedProto's original interface).
    """
    agg_protos_label = dict()

    # Collect all protos per label across clients
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    # Average per label
    for label, proto_list in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for p in proto_list:
                proto += p.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label
