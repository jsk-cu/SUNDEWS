#!/usr/bin/env python3
"""
Agents Package

Provides agent implementations for the packet distribution protocol.
Each agent implements a strategy for deciding which packets to request
from neighbors during the 4-phase communication protocol.

Available Agents
----------------
base : BaseAgent
    Dummy agent that makes no requests (control case)
min : MinAgent
    Orders neighbors by completion, requests lowest-indexed missing packets
random : RandomAgent
    Requests random useful packets from each neighbor (baseline)
rarity : RarityAgent
    Requests rarest packets first (BitTorrent-style)
demand : DemandAgent
    Tracks packet demand, prioritizes bottleneck packets
rank : RankAgent
    Uses 2-hop awareness to prioritize rare packets (RLNC-inspired)

Usage
-----
    from agents import get_agent_class, list_agents
    
    # Get a specific agent class
    AgentClass = get_agent_class("rarity")
    
    # Create an agent
    agent = AgentClass(
        agent_id=1,
        num_packets=100,
        num_satellites=24,
        is_base_station=False
    )
    
    # List all available agents
    for name, description in list_agents():
        print(f"{name}: {description}")
"""

from typing import Dict, Type, List, Tuple, Optional

from .base_agent import BaseAgent
from .min_agent import MinAgent
from .random_agent import RandomAgent
from .rarity_agent import RarityAgent
from .demand_agent import DemandAgent
from .rank_agent import RankAgent


# Registry of available agents
_AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    "base": BaseAgent,
    "min": MinAgent,
    "random": RandomAgent,
    "rarity": RarityAgent,
    "demand": DemandAgent,
    "rank": RankAgent,
}


def register_agent(name: str, agent_class: Type[BaseAgent]) -> None:
    """
    Register a new agent class.

    Parameters
    ----------
    name : str
        Name to register the agent under
    agent_class : Type[BaseAgent]
        The agent class to register
    """
    _AGENT_REGISTRY[name] = agent_class


def get_agent_class(name: str) -> Type[BaseAgent]:
    """
    Get an agent class by name.

    Parameters
    ----------
    name : str
        Name of the agent

    Returns
    -------
    Type[BaseAgent]
        The agent class

    Raises
    ------
    ValueError
        If agent name is not found in registry
    """
    if name not in _AGENT_REGISTRY:
        available = ", ".join(_AGENT_REGISTRY.keys())
        raise ValueError(
            f"Unknown agent: '{name}'. Available agents: {available}"
        )
    return _AGENT_REGISTRY[name]


def list_agents() -> List[Tuple[str, str]]:
    """
    List all available agents with their descriptions.

    Returns
    -------
    List[Tuple[str, str]]
        List of (name, description) tuples
    """
    return [
        (name, cls.description)
        for name, cls in _AGENT_REGISTRY.items()
    ]


def get_agent_names() -> List[str]:
    """
    Get list of all registered agent names.

    Returns
    -------
    List[str]
        List of agent names
    """
    return list(_AGENT_REGISTRY.keys())


# Convenience alias
Agent = MinAgent  # Default agent


__all__ = [
    # Base classes
    "BaseAgent",
    "Agent",
    
    # Agent implementations
    "MinAgent",
    "RandomAgent",
    "RarityAgent",
    "DemandAgent",
    "RankAgent",
    
    # Registry functions
    "register_agent",
    "get_agent_class",
    "list_agents",
    "get_agent_names",
]