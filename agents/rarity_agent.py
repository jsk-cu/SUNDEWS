#!/usr/bin/env python3
"""
Rarity Agent Module - Global Knowledge Variant

Implements an enhanced rarest-first strategy where each satellite maintains
knowledge of what packets every other satellite has. This information is
shared during broadcasts and used to make better-informed rarity decisions.

Key features:
1. Each satellite tracks every packet that every other satellite has
2. During broadcast, satellites share their global knowledge
3. When making requests, knowledge is updated via union with neighbor broadcasts
4. Request decisions are based on global rarity (fewest holders constellation-wide)
5. Requests are deduplicated to avoid requesting the same packet from multiple neighbors

This approach provides more accurate rarity estimates compared to local-only
knowledge, approaching the benefits of network coding without the complexity.

Source: Inspired by BitTorrent's rarest-first and informed gossip protocols.
"""

import random
from collections import defaultdict
from typing import Dict, Set, Any, Optional, List, Tuple

from .base_agent import BaseAgent


class RarityAgent(BaseAgent):
    """
    Agent with global packet knowledge that prioritizes requesting rarest packets.

    Each agent maintains a local view of what packets every satellite has.
    This knowledge is:
    - Shared in broadcasts so neighbors can update their views
    - Updated when making requests (union with neighbor knowledge)
    - Updated when responding to requests (requestor gains packet)
    - Updated when receiving packets (self gains packet)

    The request strategy prioritizes packets that the fewest satellites have
    according to the agent's global knowledge, avoiding duplicate requests
    to different neighbors within the same round.

    Parameters
    ----------
    agent_id : int
        Unique identifier. 0 = base station, 1+ = satellites.
    num_packets : int
        Total packets in the software update.
    num_satellites : int
        Total satellites in the constellation.
    is_base_station : bool
        True if this is the base station agent.

    Attributes
    ----------
    knowledge : Dict[int, Set[int]]
        Maps agent_id -> set of packets we believe that agent has.
        Includes ourselves and all other known agents.
    """

    name = "rarity"
    description = "Global knowledge rarest-first (tracks all satellite packets)"

    def __init__(
        self,
        agent_id: int,
        num_packets: int,
        num_satellites: int,
        is_base_station: bool = False,
    ):
        super().__init__(agent_id, num_packets, num_satellites, is_base_station)

        # Initialize global knowledge dictionary
        # Maps agent_id -> set of packets we think they have
        self.knowledge: Dict[int, Set[int]] = {}

        # Initialize our own knowledge entry
        self.knowledge[self.agent_id] = self.packets.copy()

        # Base station (agent_id 0) starts with all packets
        if is_base_station:
            self.knowledge[self.agent_id] = set(range(num_packets))

    def broadcast_state(self) -> Dict[str, Any]:
        """
        Phase 1: Broadcast state including global packet knowledge.

        Broadcasts:
        - Standard state (packets, completion, etc.)
        - Global knowledge: what packets we think each satellite has

        Returns
        -------
        dict
            State information including global knowledge.
        """
        # Get base broadcast
        state = super().broadcast_state()

        # Add our global knowledge (deep copy to prevent mutation)
        state["knowledge"] = {
            agent_id: packets.copy()
            for agent_id, packets in self.knowledge.items()
        }

        return state

    def make_requests(
        self, neighbor_broadcasts: Dict[int, Dict[str, Any]]
    ) -> Dict[int, int]:
        """
        Phase 2: Update knowledge from broadcasts and request rarest packets.

        Strategy:
        1. Update our global knowledge by taking union with each neighbor's
           knowledge for each agent they know about
        2. Calculate global rarity for each packet we need
        3. For each neighbor, request the rarest packet they have that we need
        4. Deduplicate: don't request the same packet from multiple neighbors

        Parameters
        ----------
        neighbor_broadcasts : Dict[int, Dict[str, Any]]
            Mapping of neighbor_id -> their broadcast state.

        Returns
        -------
        Dict[int, int]
            Mapping of neighbor_id -> packet_idx to request.
        """
        if self.has_all_packets():
            return {}

        # Step 1: Update our global knowledge from neighbor broadcasts
        for neighbor_id, broadcast in neighbor_broadcasts.items():
            # Update knowledge about the neighbor's own packets
            neighbor_packets = broadcast.get("packets", set())
            if neighbor_id not in self.knowledge:
                self.knowledge[neighbor_id] = set()
            self.knowledge[neighbor_id] = self.knowledge[neighbor_id] | neighbor_packets

            # Update knowledge from the neighbor's global knowledge
            neighbor_knowledge = broadcast.get("knowledge", {})
            for agent_id, packets in neighbor_knowledge.items():
                if agent_id not in self.knowledge:
                    self.knowledge[agent_id] = set()
                # Union: if they know an agent has a packet, we now know too
                self.knowledge[agent_id] = self.knowledge[agent_id] | packets

        # Step 2: Calculate global rarity (how many satellites have each packet)
        # Based on our updated global knowledge
        packet_holder_count: Dict[int, int] = defaultdict(int)
        for agent_id, packets in self.knowledge.items():
            for packet in packets:
                packet_holder_count[packet] += 1

        # Step 3: For each neighbor, find rarest packet they have that we need
        missing = self.get_missing_packets()
        requests: Dict[int, int] = {}
        already_requested: Set[int] = set()

        # Sort neighbors randomly to avoid bias when rarity ties occur
        neighbor_items = list(neighbor_broadcasts.items())
        random.shuffle(neighbor_items)

        for neighbor_id, broadcast in neighbor_items:
            neighbor_packets = broadcast.get("packets", set())

            # Find packets neighbor has that we need and haven't requested yet
            #useful = (neighbor_packets & missing) - already_requested
            useful = list((neighbor_packets & missing) - already_requested)
            random.shuffle(useful)

            if useful:
                # Find the rarest packet (fewest holders according to our knowledge)
                # If packet not in knowledge counts, treat as count 0 (very rare)
                # Break ties by packet index for determinism
                rarest_packet = min(
                    useful,
                    key=lambda p: packet_holder_count.get(p, 0)
                )
                requests[neighbor_id] = rarest_packet
                already_requested.add(rarest_packet)

        return requests

    def receive_requests_and_update(
        self, requests: Dict[int, int]
    ) -> Dict[int, Optional[int]]:
        """
        Phase 3: Respond to requests and update knowledge.

        For each request we grant, update our knowledge to reflect that
        the requestor will now have that packet.

        Parameters
        ----------
        requests : Dict[int, int]
            Requester agent IDs mapped to requested packet indices.

        Returns
        -------
        Dict[int, Optional[int]]
            Requester IDs mapped to packet indices being sent (or None).
        """
        responses = {}

        for requester_id, packet_idx in requests.items():
            if packet_idx in self.packets:
                responses[requester_id] = packet_idx

                # Update our knowledge: requestor will now have this packet
                if requester_id not in self.knowledge:
                    self.knowledge[requester_id] = set()
                self.knowledge[requester_id].add(packet_idx)
            else:
                responses[requester_id] = None

        return responses

    def receive_packets_and_update(
        self, received: Dict[int, Optional[int]]
    ) -> None:
        """
        Phase 4: Receive packets and update our own knowledge.

        Parameters
        ----------
        received : Dict[int, Optional[int]]
            Sender agent IDs mapped to received packet indices.
        """
        for sender_id, packet_idx in received.items():
            if packet_idx is not None:
                # Add to our packet set
                self.packets.add(packet_idx)

                # Update our knowledge about ourselves
                self.knowledge[self.agent_id].add(packet_idx)

    def reset(self) -> None:
        """Reset agent to initial state including knowledge."""
        super().reset()

        # Reset knowledge
        self.knowledge = {}
        self.knowledge[self.agent_id] = self.packets.copy()

    def get_knowledge_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the agent's global knowledge.

        Returns
        -------
        Dict[str, Any]
            Summary including known agents and average packets known.
        """
        if not self.knowledge:
            return {
                "known_agents": 0,
                "avg_packets_known": 0.0,
                "total_packet_knowledge": 0,
            }

        total_packets_known = sum(len(pkts) for pkts in self.knowledge.values())
        return {
            "known_agents": len(self.knowledge),
            "avg_packets_known": total_packets_known / len(self.knowledge),
            "total_packet_knowledge": total_packets_known,
        }

    def __repr__(self) -> str:
        agent_type = "BaseStation" if self.is_base_station else "Satellite"
        knowledge_summary = self.get_knowledge_summary()
        return (
            f"RarityAgent({agent_type}-{self.agent_id}, "
            f"packets={len(self.packets)}/{self.num_packets}, "
            f"{self.get_completion_percentage():.1f}%, "
            f"knows {knowledge_summary['known_agents']} agents)"
        )