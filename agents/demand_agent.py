#!/usr/bin/env python3
"""
Demand Agent Module

Implements a demand-aware strategy where satellites broadcast not just
what they have, but also what they need. Responders can then prioritize
sending packets that are most requested, accelerating delivery of
bottleneck packets.

This is particularly useful for satellite networks where contacts are
brief and nodes should maximize information gain per exchange.

Source: Inspired by pull-based gossip with hints (Shah 2009) and
demand-driven dissemination protocols.
"""

import random
from collections import defaultdict
from typing import Dict, Set, Any, Optional, List

from .base_agent import BaseAgent


class DemandAgent(BaseAgent):
    """
    Agent that tracks and responds to network-wide packet demand.

    Strategy for broadcasting:
    - Share both packets held AND packets wanted (missing)
    
    Strategy for requesting:
    - Prefer packets that multiple neighbors also want (viral spread)
    - This focuses network effort on bottleneck packets
    
    Strategy for responding:
    - Track which packets are being requested (demand signal)
    - When multiple requests come in, prioritize high-demand packets
    - Also prioritize more needy requesters (fewer packets)

    References
    ----------
    - Shah, "Gossip Algorithms" (Foundations and Trends 2009)
    - Demand-driven epidemic dissemination protocols
    """

    name = "demand"
    description = "Tracks packet demand across network, prioritizes bottleneck packets"

    def __init__(
        self,
        agent_id: int,
        num_packets: int,
        num_satellites: int,
        is_base_station: bool = False,
    ):
        super().__init__(agent_id, num_packets, num_satellites, is_base_station)
        
        # Track demand for packets (how often each is requested)
        self.demand_counts: Dict[int, int] = defaultdict(int)
        
        # Track neighbor completion levels from broadcasts
        self.neighbor_completion: Dict[int, float] = {}
        
        # Track what neighbors want (for smart requesting)
        self.neighbor_wants: Dict[int, Set[int]] = {}

    def broadcast_state(self) -> Dict[str, Any]:
        """
        Phase 1: Broadcast packets held AND packets wanted.

        Returns
        -------
        Dict[str, Any]
            State including both 'packets' and 'wanted' sets
        """
        return {
            "packets": self.packets.copy(),
            "wanted": self.get_missing_packets(),
            "completion": self.get_completion_percentage(),
            "agent_id": self.agent_id,
        }

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Request packets that maximize network-wide benefit.

        Prioritize packets that:
        1. We need
        2. Multiple neighbors also want (high global demand)

        This focuses effort on bottleneck packets that will enable
        further spreading when we acquire them.

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

        # Update our knowledge of neighbor states
        self.neighbor_wants.clear()
        self.neighbor_completion.clear()
        
        for neighbor_id, broadcast in neighbor_broadcasts.items():
            self.neighbor_completion[neighbor_id] = broadcast.get("completion", 0.0)
            wanted = broadcast.get("wanted", set())
            if isinstance(wanted, set):
                self.neighbor_wants[neighbor_id] = wanted
            else:
                self.neighbor_wants[neighbor_id] = set(wanted) if wanted else set()

        # Calculate global demand: count how many neighbors want each packet
        global_demand: Dict[int, int] = defaultdict(int)
        for neighbor_id, wanted in self.neighbor_wants.items():
            for packet in wanted:
                global_demand[packet] += 1

        # Make requests prioritizing high-demand packets
        requests: Dict[int, int] = {}
        already_requested: Set[int] = set()

        # Shuffle neighbors to avoid bias
        neighbor_items = list(neighbor_broadcasts.items())
        random.shuffle(neighbor_items)

        for neighbor_id, broadcast in neighbor_items:
            neighbor_packets = broadcast.get("packets", set())

            # Find useful packets we haven't already requested
            useful = (neighbor_packets & missing) - already_requested

            if useful:
                # Prefer packets with highest global demand
                # (more neighbors want this = acquiring it helps the network)
                # Tie-break by packet index for determinism
                best_packet = max(
                    useful,
                    key=lambda p: (global_demand[p], -p)
                )
                requests[neighbor_id] = best_packet
                already_requested.add(best_packet)

        return requests

    def receive_requests_and_update(
        self, requests: Dict[int, int]
    ) -> Dict[int, Optional[int]]:
        """
        Phase 3: Respond to requests with demand-aware prioritization.

        When multiple requests come in:
        1. Update demand counts for requested packets
        2. Prioritize sending to needier requesters (lower completion)
        3. For tie-breaking, prioritize high-demand packets

        Parameters
        ----------
        requests : Dict[int, int]
            Mapping of requester_id -> packet_idx they want

        Returns
        -------
        Dict[int, Optional[int]]
            Mapping of requester_id -> packet_idx sent (None if not sent)
        """
        # Update demand tracking
        for requester_id, packet_idx in requests.items():
            self.demand_counts[packet_idx] += 1

        # Build response, prioritizing needier requesters
        responses: Dict[int, Optional[int]] = {}

        # Sort requests by requester need (lower completion = higher priority)
        # Use completion from last broadcast cycle if available
        sorted_requests = sorted(
            requests.items(),
            key=lambda x: self.neighbor_completion.get(x[0], 100.0)
        )

        for requester_id, packet_idx in sorted_requests:
            if packet_idx in self.packets:
                responses[requester_id] = packet_idx
            else:
                responses[requester_id] = None

        return responses

    def get_demand_stats(self) -> Dict[str, Any]:
        """
        Get statistics about observed packet demand.

        Returns
        -------
        Dict[str, Any]
            Demand statistics for analysis
        """
        if not self.demand_counts:
            return {"total_requests": 0, "unique_packets": 0}

        return {
            "total_requests": sum(self.demand_counts.values()),
            "unique_packets": len(self.demand_counts),
            "most_demanded": max(self.demand_counts.items(), key=lambda x: x[1]),
            "demand_distribution": dict(self.demand_counts),
        }