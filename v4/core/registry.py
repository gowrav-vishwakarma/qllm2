#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v4 Registry: Factory pattern for creating modules from config

Allows dynamic instantiation of any registered component:
- PhaseBank implementations
- Coupler implementations
- Backbone implementations
- Memory implementations
- Objective implementations
- Sampler implementations

Usage:
    registry = get_registry()
    registry.register_bank('semantic', SemanticPhaseBank)
    
    bank = registry.create_bank('semantic', dim=256)
"""

from typing import Dict, Type, Any, Callable, Optional, List
from dataclasses import dataclass, field
import importlib


@dataclass
class RegistryEntry:
    """Entry in the registry"""
    cls: Type
    description: str = ""
    default_config: Dict[str, Any] = field(default_factory=dict)


class Registry:
    """
    Central registry for all v4 components.
    
    Supports:
    - Registration of component implementations
    - Factory creation from config
    - Discovery of available implementations
    """
    
    def __init__(self):
        self._banks: Dict[str, RegistryEntry] = {}
        self._couplers: Dict[str, RegistryEntry] = {}
        self._backbones: Dict[str, RegistryEntry] = {}
        self._memories: Dict[str, RegistryEntry] = {}
        self._objectives: Dict[str, RegistryEntry] = {}
        self._samplers: Dict[str, RegistryEntry] = {}
    
    # =========================================================================
    # Registration
    # =========================================================================
    
    def register_bank(
        self, 
        name: str, 
        cls: Type, 
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a PhaseBank implementation"""
        self._banks[name] = RegistryEntry(
            cls=cls, 
            description=description,
            default_config=default_config or {}
        )
    
    def register_coupler(
        self, 
        name: str, 
        cls: Type, 
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a Coupler implementation"""
        self._couplers[name] = RegistryEntry(
            cls=cls,
            description=description,
            default_config=default_config or {}
        )
    
    def register_backbone(
        self, 
        name: str, 
        cls: Type, 
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a Backbone implementation"""
        self._backbones[name] = RegistryEntry(
            cls=cls,
            description=description,
            default_config=default_config or {}
        )
    
    def register_memory(
        self, 
        name: str, 
        cls: Type, 
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a Memory implementation"""
        self._memories[name] = RegistryEntry(
            cls=cls,
            description=description,
            default_config=default_config or {}
        )
    
    def register_objective(
        self, 
        name: str, 
        cls: Type, 
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register an Objective implementation"""
        self._objectives[name] = RegistryEntry(
            cls=cls,
            description=description,
            default_config=default_config or {}
        )
    
    def register_sampler(
        self, 
        name: str, 
        cls: Type, 
        description: str = "",
        default_config: Optional[Dict[str, Any]] = None
    ) -> None:
        """Register a Sampler implementation"""
        self._samplers[name] = RegistryEntry(
            cls=cls,
            description=description,
            default_config=default_config or {}
        )
    
    # =========================================================================
    # Factory Creation
    # =========================================================================
    
    def _create(self, registry: Dict[str, RegistryEntry], name: str, **kwargs) -> Any:
        """Generic factory method"""
        if name not in registry:
            available = list(registry.keys())
            raise ValueError(f"Unknown component: {name}. Available: {available}")
        
        entry = registry[name]
        # Merge default config with provided kwargs
        config = {**entry.default_config, **kwargs}
        return entry.cls(**config)
    
    def create_bank(self, name: str, **kwargs) -> 'PhaseBank':
        """Create a PhaseBank instance"""
        return self._create(self._banks, name, **kwargs)
    
    def create_coupler(self, name: str, **kwargs) -> 'Coupler':
        """Create a Coupler instance"""
        return self._create(self._couplers, name, **kwargs)
    
    def create_backbone(self, name: str, **kwargs) -> 'Backbone':
        """Create a Backbone instance"""
        return self._create(self._backbones, name, **kwargs)
    
    def create_memory(self, name: str, **kwargs) -> 'Memory':
        """Create a Memory instance"""
        return self._create(self._memories, name, **kwargs)
    
    def create_objective(self, name: str, **kwargs) -> 'Objective':
        """Create an Objective instance"""
        return self._create(self._objectives, name, **kwargs)
    
    def create_sampler(self, name: str, **kwargs) -> 'Sampler':
        """Create a Sampler instance"""
        return self._create(self._samplers, name, **kwargs)
    
    # =========================================================================
    # Discovery
    # =========================================================================
    
    def list_banks(self) -> List[str]:
        """List registered bank types"""
        return list(self._banks.keys())
    
    def list_couplers(self) -> List[str]:
        """List registered coupler types"""
        return list(self._couplers.keys())
    
    def list_backbones(self) -> List[str]:
        """List registered backbone types"""
        return list(self._backbones.keys())
    
    def list_memories(self) -> List[str]:
        """List registered memory types"""
        return list(self._memories.keys())
    
    def list_objectives(self) -> List[str]:
        """List registered objective types"""
        return list(self._objectives.keys())
    
    def list_samplers(self) -> List[str]:
        """List registered sampler types"""
        return list(self._samplers.keys())
    
    def describe(self, component_type: str, name: str) -> str:
        """Get description of a component"""
        registries = {
            'bank': self._banks,
            'coupler': self._couplers,
            'backbone': self._backbones,
            'memory': self._memories,
            'objective': self._objectives,
            'sampler': self._samplers,
        }
        if component_type not in registries:
            raise ValueError(f"Unknown component type: {component_type}")
        
        registry = registries[component_type]
        if name not in registry:
            raise ValueError(f"Unknown {component_type}: {name}")
        
        return registry[name].description


# =============================================================================
# Global Registry Singleton
# =============================================================================

_GLOBAL_REGISTRY: Optional[Registry] = None


def get_registry() -> Registry:
    """Get or create the global registry"""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = Registry()
    return _GLOBAL_REGISTRY


def reset_registry() -> None:
    """Reset the global registry (for testing)"""
    global _GLOBAL_REGISTRY
    _GLOBAL_REGISTRY = None


# =============================================================================
# Decorator for easy registration
# =============================================================================

def register_bank(name: str, description: str = "", **default_config):
    """Decorator to register a PhaseBank class"""
    def decorator(cls):
        get_registry().register_bank(name, cls, description, default_config)
        return cls
    return decorator


def register_coupler(name: str, description: str = "", **default_config):
    """Decorator to register a Coupler class"""
    def decorator(cls):
        get_registry().register_coupler(name, cls, description, default_config)
        return cls
    return decorator


def register_backbone(name: str, description: str = "", **default_config):
    """Decorator to register a Backbone class"""
    def decorator(cls):
        get_registry().register_backbone(name, cls, description, default_config)
        return cls
    return decorator


def register_memory(name: str, description: str = "", **default_config):
    """Decorator to register a Memory class"""
    def decorator(cls):
        get_registry().register_memory(name, cls, description, default_config)
        return cls
    return decorator


def register_objective(name: str, description: str = "", **default_config):
    """Decorator to register an Objective class"""
    def decorator(cls):
        get_registry().register_objective(name, cls, description, default_config)
        return cls
    return decorator


def register_sampler(name: str, description: str = "", **default_config):
    """Decorator to register a Sampler class"""
    def decorator(cls):
        get_registry().register_sampler(name, cls, description, default_config)
        return cls
    return decorator
