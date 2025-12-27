#!/usr/bin/env python3
"""
Rarity Agent Module

Implements the rarest-first strategy inspired by BitTorrent. Tracks
packet frequency across the local neighborhood and prioritizes
requesting the rarest packets first.

This directly addresses the coupon collector bottleneck by ensuring
rare packets propagate faster through the network.

Source: Cohen (2003) BitTorrent, Legout et al. "Rarest First and 
Choke Algorithms Are Enough" (IMC 2006)
"""

import random
from collections import defaultdict
from typing import Dict, Set, Any, Optional, List, Tuple

from .base_agent import BaseAgent


class RarityAgent(BaseAgent):
    """
    Agent that prioritizes requesting the rarest packets.

    Strategy:
    1. Count how many neighbors have each packet (local rarity)
    2. For each neighbor, request the rarest packet they have that we need
    3. Avoid duplicate requests within the same round

    This maintains high entropy in packet distribution, approaching
    optimal O(n) completion time similar to network coding benefits.

    References
    ----------
    - Cohen, "Incentives Build Robustness in BitTorrent" (2003)
    - Legout, Urvoy-Keller, Michiardi, "Rarest First and Choke 
      Algorithms Are Enough" (IMC 2006)
    """

    name = "rarity"
    description = "Requests rarest packets first (BitTorrent-style)"

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Request the rarest useful packet from each neighbor.

        Parameters
        ----------
        neighbor_broadcasts : Dict[int, Dict[str, Any]]
            Mapping of neighbor_id -> their broadcast state

        Returns
        -------
        Dict[int, int]
            Mapping of neighbor_id -> packet_idx to request
        """
        if self.has_all_packets():
            return {}

        missing = self.get_missing_packets()

        # Phase 1: Count packet frequency across all neighbors
        packet_counts: Dict[int, int] = defaultdict(int)
        for neighbor_id, broadcast in neighbor_broadcasts.items():
            neighbor_packets = broadcast.get("packets", set())
            for packet in neighbor_packets:
                packet_counts[packet] += 1

        # Phase 2: Request rarest available packet from each neighbor
        requests: Dict[int, int] = {}
        already_requested: Set[int] = set()

        # Sort neighbors randomly to avoid bias when rarity ties occur
        neighbor_items = list(neighbor_broadcasts.items())
        random.shuffle(neighbor_items)

        for neighbor_id, broadcast in neighbor_items:
            neighbor_packets = broadcast.get("packets", set())

            # Find packets neighbor has that we need and haven't requested
            useful = (neighbor_packets & missing) - already_requested

            if useful:
                # Find the rarest packet (lowest count)
                # Break ties by packet index for determinism
                rarest_packet = min(
                    useful,
                    key=lambda p: (packet_counts[p], p)
                )
                requests[neighbor_id] = rarest_packet
                already_requested.add(rarest_packet)

        return requests