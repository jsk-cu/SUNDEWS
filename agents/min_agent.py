#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Min Agent Class for Satellite Constellation Packet Distribution

Instance of a greedy lowest-indexed-packet-first class for agents that
manage packet distribution in a satellite constellation. Each satellite 
and the base station receives an agent that decides how to broadcast state,
make requests, and fulfill requests from neighbors.

The protocol runs in 4 phases each timestep:
1. broadcast_state() - All agents broadcast their state
2. make_requests() - All agents request packets from neighbors
3. receive_requests_and_update() - All agents decide what to send
4. receive_packets_and_update() - All agents receive packets
"""

from typing import Any, Dict, Optional, Set


class Agent:
    """
    Base template class for satellite constellation agents.
    
    This template class provides the interface for agents that manage
    packet distribution in a satellite constellation. Subclasses should
    implement the actual logic for deciding what to broadcast, request,
    and send.
    
    Parameters
    ----------
    agent_id : int
        Unique integer identifier for this agent
    num_packets : int
        Total number of packets that make up the software update
    num_satellites : int
        Total number of satellites in the constellation
    is_base_station : bool
        True if this agent represents the base station (which starts
        with all packets), False for satellite agents
    
    Attributes
    ----------
    agent_id : int
        This agent's unique identifier
    num_packets : int
        Total packets in the update
    num_satellites : int
        Total satellites in constellation
    is_base_station : bool
        Whether this is the base station agent
    packets : Set[int]
        Set of packet indices this agent currently has
    """
    
    def __init__(
        self,
        agent_id: int,
        num_packets: int,
        num_satellites: int,
        is_base_station: bool = False
    ):
        self.agent_id = agent_id
        self.num_packets = num_packets
        self.num_satellites = num_satellites
        self.is_base_station = is_base_station
        
        # Initialize packet set
        # Base station starts with all packets
        # Satellites start with no packets
        if is_base_station:
            self.packets: Set[int] = set(range(num_packets))
        else:
            self.packets: Set[int] = set()
    
    def broadcast_state(self) -> Any:
        """
        Generate state information to broadcast to neighbors.
        
        This method is called at the start of each timestep. The returned
        state will be provided to all neighboring agents when they call
        make_requests().
        
        Returns
        -------
        Any
            State information to broadcast. Format depends on implementation.
            The template returns an empty dictionary.
        """
        return self.packets
    
    def make_requests(self, neighbor_broadcasts: Dict[int, Any]) -> Dict[int, int]:
        """
        Given broadcasts from neighbors, decide what packets to request.
        
        This method is called after all agents have broadcast their state.
        The agent receives the broadcasts from all currently connected
        neighbors and should return requests for packets it wants.
        
        Parameters
        ----------
        neighbor_broadcasts : Dict[int, Any]
            Dictionary mapping neighbor agent IDs to their broadcast state.
            Only currently connected neighbors are included.
        
        Returns
        -------
        Dict[int, int]
            Dictionary mapping neighbor agent IDs to packet indices to request.
            Each neighbor can receive at most one request.
            The template returns an empty dictionary (no requests).
        """
        
        current_highest = -1
        if len(self.packets):
            current_highest = max(self.packets)
            
        sorted_broadcasts = [(neigh_idx, neigh_bc) for neigh_idx, neigh_bc in sorted(neighbor_broadcasts.items(), key = lambda x : len(x[1])) if len(neigh_bc) > len(self.packets)]
        
        outgoing_requests = {}
        
        for neigh_idx, neigh_bc in sorted_broadcasts:
            if current_highest + 1 in neigh_bc:
                current_highest += 1
                outgoing_requests[neigh_idx] = current_highest
        
        return outgoing_requests
    
    def receive_requests_and_update(
        self,
        requests: Dict[int, int]
    ) -> Dict[int, Optional[int]]:
        """
        Receive requests from neighbors and decide what packets to send.
        
        This method is called after all agents have made their requests.
        The agent receives all requests directed at it and should decide
        which requests to fulfill.
        
        Parameters
        ----------
        requests : Dict[int, int]
            Dictionary mapping requester agent IDs to the packet indices
            they requested. Only neighbors who made requests are included.
        
        Returns
        -------
        Dict[int, Optional[int]]
            Dictionary mapping requester agent IDs to:
            - The packet index to send (if fulfilling the request)
            - None (if declining the request)
            The template returns None for all requests.
        """
        # Template: decline all requests
        outgoing_packets = {}
        for requester_id, requested_packet in requests.items():
            if requested_packet in self.packets:
                outgoing_packets[requester_id] = requested_packet
            else:
                outgoing_packets[requester_id] = None
        
        return outgoing_packets
    
    def receive_packets_and_update(
        self,
        packets: Dict[int, Optional[int]]
    ) -> None:
        """
        Receive packets from neighbors and update local state.
        
        This method is called at the end of each timestep. The agent
        receives the results of its requests - either the packet index
        that was sent, or None if the request was declined.
        
        Parameters
        ----------
        packets : Dict[int, Optional[int]]
            Dictionary mapping sender agent IDs (the agents that were
            requested from) to:
            - The packet index that was sent (if request was fulfilled)
            - None (if request was declined or couldn't be fulfilled)
        """
        # Template: do nothing
        for sender_id, received_packet in packets.items():
            if received_packet is not None:
                self.packets.add(received_packet)
    
    def has_all_packets(self) -> bool:
        """
        Check if this agent has received all packets.
        
        Returns
        -------
        bool
            True if the agent has all num_packets packets
        """
        return len(self.packets) == self.num_packets
    
    def get_packet_count(self) -> int:
        """
        Get the number of packets this agent currently has.
        
        Returns
        -------
        int
            Number of unique packets in this agent's possession
        """
        return len(self.packets)
    
    def get_missing_packets(self) -> Set[int]:
        """
        Get the set of packet indices this agent is missing.
        
        Returns
        -------
        Set[int]
            Set of packet indices not yet received
        """
        return set(range(self.num_packets)) - self.packets
    
    def get_completion_percentage(self) -> float:
        """
        Get the percentage of packets this agent has received.
        
        Returns
        -------
        float
            Percentage from 0.0 to 100.0
        """
        if self.num_packets == 0:
            return 100.0
        return (len(self.packets) / self.num_packets) * 100.0
    
    def __repr__(self) -> str:
        agent_type = "BaseStation" if self.is_base_station else "Satellite"
        return (
            f"Agent(id={self.agent_id}, "
            f"type={agent_type}, "
            f"packets={len(self.packets)}/{self.num_packets})"
        )