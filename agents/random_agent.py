#!/usr/bin/env python3
"""
Random Agent Module

Implements the simplest possible strategy: request any random missing
packet from each neighbor. This serves as the lower-bound baseline
for comparing more sophisticated algorithms.

Source: Standard gossip baseline from epidemic algorithm literature.
"""

import random
from typing import Dict, Set, Any, Optional

from .base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that requests random useful packets from neighbors.

    Strategy:
    1. For each neighbor, identify packets they have that we need
    2. Request a random packet from that set
    
    No coordination between requests - may request the same packet
    from multiple neighbors. This represents the simplest possible
    strategy and serves as a baseline for comparison.

    References
    ----------
    - Demers et al., "Epidemic Algorithms for Replicated Database 
      Maintenance" (PODC 1987)
    """

    name = "random"
    description = "Requests random useful packets from each neighbor (baseline)"

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Request a random useful packet from each neighbor.

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

        requests: Dict[int, int] = {}
        missing = self.get_missing_packets()

        for neighbor_id, broadcast in neighbor_broadcasts.items():
            neighbor_packets = broadcast.get("packets", set())

            # Find packets neighbor has that we need
            useful = neighbor_packets & missing

            if useful:
                # Request a random useful packet
                packet_to_request = random.choice(list(useful))
                requests[neighbor_id] = packet_to_request

        return requests