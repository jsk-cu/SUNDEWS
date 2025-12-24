#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SatUpdate Agents Package

This package provides agent classes for managing packet distribution
in satellite constellations. Each satellite and the base station is
assigned an agent that decides how to broadcast state, request packets,
and fulfill requests from neighbors.

Protocol Overview:
    Each simulation timestep runs a 4-phase protocol:
    
    1. Broadcast Phase: All agents call broadcast_state() and the results
       are collected by the simulation.
    
    2. Request Phase: Each agent receives the broadcasts from its current
       neighbors and calls make_requests() to decide what packets to request.
    
    3. Response Phase: Each agent receives the requests from neighbors
       via receive_requests_and_update() and decides what to send.
    
    4. Delivery Phase: Each agent receives the packets sent to it via
       receive_packets_and_update() and updates its local state.

Usage:
    from SatUpdate.agents import Agent
    
    # Create a satellite agent
    agent = Agent(
        agent_id=1,
        num_packets=100,
        num_satellites=24,
        is_base_station=False
    )
    
    # Create a base station agent (starts with all packets)
    base_agent = Agent(
        agent_id=0,
        num_packets=100,
        num_satellites=24,
        is_base_station=True
    )

Creating Custom Agents:
    To implement a custom distribution strategy, subclass Agent and
    override the broadcast_state(), make_requests(), 
    receive_requests_and_update(), and receive_packets_and_update() methods.
    
    See the agents directory for example implementations.
"""

#from .base_agent import Agent
from .min_agent import Agent


__all__ = [
    "Agent",
]

__version__ = "1.0.0"