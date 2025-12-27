#!/usr/bin/env python3
"""
Rank Agent Module

Implements a network-coding-inspired strategy that uses 2-hop topology
awareness to prioritize packets with fewer alternative sources. This
approximates the key insight from Random Linear Network Coding (RLNC):
maximize the "rank" of information at each exchange.

While true RLNC requires actual encoding, we simulate the benefit by
tracking which packets could be obtained through the transitive closure
of our neighborhood and prioritizing rare/bottleneck packets.

Source: Inspired by Deb, Médard, Choute "Algebraic Gossip" (2006) and
Haeupler "Analyzing Network Coding Gossip Made Easy" (STOC 2011)
"""

import random
from collections import defaultdict
from typing import Dict, Set, Any, Optional, List, Tuple

from .base_agent import BaseAgent


class RankAgent(BaseAgent):
    """
    Agent that uses 2-hop topology awareness to maximize information gain.

    Strategy:
    1. Broadcast not just packets held, but also what neighbors have
       (2-hop reachability information)
    2. When requesting, prioritize packets with fewest alternative sources
       (if we don't get it now, we may not get another chance)
    3. This approximates RLNC's "any new information is useful" property

    The key insight is that in dynamic networks with brief contacts,
    you should prioritize information that's harder to obtain elsewhere.

    References
    ----------
    - Deb, Médard, Choute, "Algebraic Gossip: A Network Coding Approach 
      to Optimal Multiple Rumor Mongering" (IEEE Trans. IT 2006)
    - Haeupler, "Analyzing Network Coding Gossip Made Easy" (STOC 2011)
    """

    name = "rank"
    description = "Uses 2-hop awareness to prioritize rare packets (RLNC-inspired)"

    def __init__(
        self,
        agent_id: int,
        num_packets: int,
        num_satellites: int,
        is_base_station: bool = False,
    ):
        super().__init__(agent_id, num_packets, num_satellites, is_base_station)
        
        # Track what we've observed neighbors having (for 2-hop sharing)
        self.observed_neighbor_packets: Dict[int, Set[int]] = {}
        
        # Track our current direct neighbors (from most recent broadcast phase)
        self.current_neighbors: Set[int] = set()

    def broadcast_state(self) -> Dict[str, Any]:
        """
        Phase 1: Broadcast packets AND 2-hop reachability information.

        We share what packets our neighbors had last round, enabling
        recipients to build a 2-hop picture of packet availability.

        Returns
        -------
        Dict[str, Any]
            State including packets and neighbor packet summaries
        """
        # Summarize neighbor packets for 2-hop awareness
        # Only share recent observations to limit state size
        neighbor_summary = {
            neighbor_id: packets.copy()
            for neighbor_id, packets in self.observed_neighbor_packets.items()
        }
        
        return {
            "packets": self.packets.copy(),
            "completion": self.get_completion_percentage(),
            "agent_id": self.agent_id,
            "neighbor_packets": neighbor_summary,
        }

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Request packets that have the fewest alternative sources.

        Build a reachability map from 2-hop information and prioritize
        packets that are hardest to obtain (fewest sources in our
        extended neighborhood).

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

        # Update our observation of neighbor packets
        self.observed_neighbor_packets.clear()
        self.current_neighbors = set(neighbor_broadcasts.keys())
        
        for neighbor_id, broadcast in neighbor_broadcasts.items():
            neighbor_packets = broadcast.get("packets", set())
            if isinstance(neighbor_packets, set):
                self.observed_neighbor_packets[neighbor_id] = neighbor_packets.copy()
            else:
                self.observed_neighbor_packets[neighbor_id] = set(neighbor_packets)

        # Build reachability map: for each packet, count how many ways we could get it
        # Include both direct (1-hop) and indirect (2-hop) sources
        packet_sources: Dict[int, int] = defaultdict(int)
        
        # Direct sources (1-hop neighbors who have the packet)
        for neighbor_id, broadcast in neighbor_broadcasts.items():
            neighbor_packets = broadcast.get("packets", set())
            for packet in neighbor_packets:
                packet_sources[packet] += 1
        
        # Indirect sources (2-hop: neighbors of neighbors)
        for neighbor_id, broadcast in neighbor_broadcasts.items():
            neighbor_neighbor_packets = broadcast.get("neighbor_packets", {})
            
            if isinstance(neighbor_neighbor_packets, dict):
                for nn_id, nn_packets in neighbor_neighbor_packets.items():
                    # Don't count ourselves or direct neighbors
                    if nn_id != self.agent_id and nn_id not in self.current_neighbors:
                        if isinstance(nn_packets, set):
                            for packet in nn_packets:
                                # Weight 2-hop sources less (harder to reach)
                                packet_sources[packet] += 0.5
                        elif nn_packets:
                            for packet in nn_packets:
                                packet_sources[packet] += 0.5

        # Make requests prioritizing packets with fewest sources
        requests: Dict[int, int] = {}
        already_requested: Set[int] = set()

        # Shuffle neighbors to avoid systematic bias
        neighbor_items = list(neighbor_broadcasts.items())
        random.shuffle(neighbor_items)

        for neighbor_id, broadcast in neighbor_items:
            neighbor_packets = broadcast.get("packets", set())
            
            # Find useful packets we haven't requested
            useful = (neighbor_packets & missing) - already_requested

            if useful:
                # Request the packet with fewest alternative sources
                # This is the packet we're least likely to get elsewhere
                # Tie-break by packet index for determinism
                rarest_packet = min(
                    useful,
                    key=lambda p: (packet_sources.get(p, 0), p)
                )
                requests[neighbor_id] = rarest_packet
                already_requested.add(rarest_packet)

        return requests

    def get_reachability_stats(self) -> Dict[str, Any]:
        """
        Get statistics about 2-hop packet reachability.

        Returns
        -------
        Dict[str, Any]
            Reachability statistics for analysis
        """
        if not self.observed_neighbor_packets:
            return {
                "direct_neighbors": 0,
                "reachable_packets": 0,
            }

        # Compute reachable packets through 1-hop
        reachable = set()
        for packets in self.observed_neighbor_packets.values():
            reachable.update(packets)

        return {
            "direct_neighbors": len(self.observed_neighbor_packets),
            "reachable_packets": len(reachable),
            "missing_but_reachable": len(reachable & self.get_missing_packets()),
        }