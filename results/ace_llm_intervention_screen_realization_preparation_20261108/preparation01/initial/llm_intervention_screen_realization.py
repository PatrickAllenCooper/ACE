"""Single-use descriptor realization for the exact prospective screen.

The caller must first authenticate the captured source, admit the namespace,
freeze the scientific protocol and own the controller lock. This small helper
does not provide those guarantees. It never collects responses, scores worlds,
selects worlds or retries a partial realization. Do not invoke it during
preparation. Artificial tests replace the generator before any parameter draw.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

from llm_intervention_contract import Node, canonical, digest
from llm_intervention_screen_protocol import REQUIRED_SOURCE_PATHS, validate_proposal
from llm_intervention_world import generate_world

VERSION = 'ace-first-screen-descriptor-realization-v1'


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _save(path, value):
    raw = (canonical(value)+'\n').encode()
    with path.open('xb') as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def realization_recipes(proposal, *, proposal_sha256, source_pins):
    """Return all 56 ordered parameter laws without drawing any parameter."""
    _need(digest(proposal) == proposal_sha256, 'independent proposal pin differs')
    validate_proposal(proposal, expected_source_pins=source_pins)
    rows = proposal['training_worlds'] + proposal['development_worlds']
    result = []
    for row in rows:
        law = row['truth_law']
        graph = tuple(Node(n['name'], tuple(n['parents']), n['unit'],
            None if n['intervention_bounds'] is None else tuple(n['intervention_bounds']))
            for n in row['public']['graph'])
        result.append(dict(scope=row['scope'], world_id=row['world_id'], graph=graph,
            intercept_means=tuple(law['intercept_means']),
            coefficient_means=tuple(tuple(r) for r in law['coefficient_means']),
            noise_variances=tuple(law['noise_variances']), intercept_sd=law['intercept_sd'],
            coefficient_sd=law['coefficient_sd']))
    _need(len(result) == 56 and len({(r['scope'], r['world_id']) for r in result}) == 56,
          'complete unique screen roster required')
    return result


def realize_screen_worlds(*, proposal, proposal_sha256, source_pins, output,
                         freeze_sha256, namespace_admission_sha256):
    """Realize the whole predeclared roster once in a new private directory.

    The two external pins record the caller's independent freeze and namespace
    decisions. This helper cannot authenticate their provenance or authorize
    itself. The outer runner must retain those exact records. A partial output
    is a failed attempt, never an input to another partial or replacement run.
    Returned descriptor pins become independent inputs to later collection.
    """
    for pin in (freeze_sha256, namespace_admission_sha256):
        _need(type(pin) is str and len(pin) == 64 and all(c in '0123456789abcdef' for c in pin),
              'independent freeze and namespace pins required')
    proposal = json.loads(canonical(proposal))
    source_pins = dict(source_pins)
    recipes = realization_recipes(proposal, proposal_sha256=proposal_sha256, source_pins=source_pins)
    # The outer captured loader binds execution. These hashes check its supplied
    # protocol closure again and are not a replacement for that source gate.
    here = Path(__file__).absolute().parent
    _need(set(source_pins) == set(REQUIRED_SOURCE_PATHS)
        and all(hashlib.sha256((here/Path(n).name).read_bytes()).hexdigest() == p
                for n, p in source_pins.items()), 'protocol source bytes differ')
    output = Path(output)
    _need(output.is_absolute() and not output.is_symlink(), 'absolute exclusive output required')
    output.mkdir(exist_ok=False, mode=0o700)
    entries = [dict(scope=r['scope'], world_id=r['world_id'], path=f'world-{i:03d}.json',
                    status='unstarted') for i, r in enumerate(recipes)]
    common = dict(version=VERSION, proposal_sha256=proposal_sha256, source_pins=source_pins,
        freeze_sha256=freeze_sha256, namespace_admission_sha256=namespace_admission_sha256,
        retry_allowed=False, environment_responses=0, policy_updates=0,
        scientific_execution_qualified=False)
    _save(output/'intent.json', dict(common, started_utc=datetime.now(timezone.utc).isoformat(),
                                    entries=entries))
    _save(output/'proposal.json', proposal)
    try:
        for i, recipe in enumerate(recipes):
            entries[i]['status'] = 'started'
            _save(output/f'world-{i:03d}.intent.json', dict(scope=recipe['scope'], world_id=recipe['world_id']))
            world = generate_world(**recipe)
            _need(world.scope == recipe['scope'] and world.world_id == recipe['world_id']
                and world.graph == recipe['graph'], 'generator identity or graph differs')
            payload = world.private_payload()
            _need(digest(payload) == world.sha256, 'descriptor digest differs')
            byte_pin = _save(output/entries[i]['path'], payload)
            entries[i].update(status='complete', descriptor_sha256=world.sha256, file_sha256=byte_pin)
        terminal = dict(common, status='complete', worlds_generated=len(entries), entries=entries,
                        completed_utc=datetime.now(timezone.utc).isoformat())
        _save(output/'terminal.candidate.json', terminal)
        os.link(output/'terminal.candidate.json', output/'terminal.json')
        return terminal
    except BaseException as exc:
        if not os.path.lexists(output/'terminal.json'):
            _save(output/'failure.json', dict(common, status='failed', entries=entries,
                completed_descriptors=sum(e['status'] == 'complete' for e in entries),
                exception_type=type(exc).__name__, message=str(exc)))
        raise
