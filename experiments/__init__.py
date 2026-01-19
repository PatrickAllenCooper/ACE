"""
ACE Experiments Package

Lazy loading to avoid import failures when optional dependencies (scipy, pandas_datareader) 
are not installed. Each experiment module can be imported independently.
"""

def get_duffing_experiment():
    """Lazy import for Duffing oscillators (requires scipy)"""
    from .duffing_oscillators import run_duffing_experiment, DuffingOscillatorChain
    return run_duffing_experiment, DuffingOscillatorChain

def get_phillips_experiment():
    """Lazy import for Phillips curve (requires pandas_datareader)"""
    from .phillips_curve import run_phillips_experiment, PhillipsCurveOracle
    return run_phillips_experiment, PhillipsCurveOracle

__all__ = ['get_duffing_experiment', 'get_phillips_experiment']
