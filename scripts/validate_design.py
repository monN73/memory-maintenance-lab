"""Validate design artifacts only; no models, network, or runtime evaluation."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load(relative):
    return json.loads((ROOT / relative).read_text())


def unique(items, key):
    values = [item[key] for item in items]
    require(len(values) == len(set(values)), f'Duplicate {key}')
    return set(values)


def main():
    for directory in ('configs', 'templates'):
        for path in (ROOT / directory).rglob('*.json'):
            json.loads(path.read_text(encoding='utf-8'))
    markdown_files = [ROOT / 'README.md', *(ROOT / 'docs').rglob('*.md')]
    for path in markdown_files:
        for label, target in re.findall(r'\[([^\]\n]+)\]\(([^)\n]+)\)', path.read_text()):
            if '://' not in target:
                base, _, anchor = target.partition('#')
                destination = path.parent / base
                require(destination.exists(), f'Missing link: {path.name}: {target}')
                require(label == Path(base).name, f'Link label must be filename: {path.name}: {label}')
                if anchor:
                    headings = re.findall(r'^#{1,6} (.+)$', destination.read_text(), re.M)
                    slugs = [re.sub(r'[^\w\s-]', '', h.lower()).replace(' ', '-') for h in headings]
                    require(anchor in slugs, f'Missing anchor: {path.name}: {target}')
    pilot = load('configs/pilot.json')
    registry = load('configs/strategy-registry.json')
    specs = unique(registry['presets'], 'strategy_spec_id')
    orchestrations = unique(load('configs/orchestration-registry.json')['profiles'],
                            'orchestration_profile_id')
    manifest = load('configs/experiment-cells.json')
    cells = unique(manifest['cells'], 'cell_id')
    for cell in manifest['cells']:
        require(cell['strategy_spec_id'] in specs, f'Unknown strategy: {cell}')
        require(cell['orchestration_profile_id'] in orchestrations,
                f'Unknown orchestration: {cell}')
    for batch in manifest['batches']:
        require(len(batch['cell_ids']) == len(set(batch['cell_ids'])), 'Duplicate batch cell')
        require(set(batch['cell_ids']) <= cells, 'Unknown batch cell')
        require(batch['planned_runs'] == len(batch['cell_ids']) * batch['question_count'] * batch['repeats'],
                'Incorrect batch run count')
    cell_by_id = {cell['cell_id']: cell for cell in manifest['cells']}
    spec_by_id = {spec['strategy_spec_id']: spec for spec in registry['presets']}
    for comparison in manifest['comparisons']:
        require(comparison['control_cell'] in cells, 'Unknown control')
        require(comparison['treatment_cell'] in cells, 'Unknown treatment')
        control = cell_by_id[comparison['control_cell']]
        treatment = cell_by_id[comparison['treatment_cell']]
        left = spec_by_id[control['strategy_spec_id']]
        right = spec_by_id[treatment['strategy_spec_id']]
        changed = {key for key in registry['dimensions'] if left.get(key) != right.get(key)}
        changed |= {key for key in ['track', 'native_profile_id', 'orchestration_profile_id', 'integration_mode']
                    if control.get(key) != treatment.get(key)}
        require(changed == set(comparison['changed_factors']),
                f'Undeclared or incorrect factors: {comparison["comparison_id"]}: {sorted(changed)}')
        if comparison['claim_scope'].startswith('controlled_component'):
            require(len(changed) == 1, 'Component comparison changes multiple factors')
    require(set(pilot['tracks']['direct']) <= specs, 'Unregistered direct strategy')
    for field in ['cell_manifest', 'strategy_registry', 'orchestration_registry']:
        require((ROOT / 'configs' / pilot[field]).exists(), f'Missing {field}')
    batch = manifest['batches'][0]
    for field in ['question_count', 'repeats', 'planned_runs']:
        require(pilot['pilot'][field] == batch[field], f'Pilot/batch {field} mismatch')
    require(pilot['pilot']['strategies'] == len(batch['cell_ids']), 'Pilot/batch strategy count mismatch')
    require(pilot['dataset']['subset_size'] == batch['question_count'], 'Subset size mismatch')
    require(pilot['protocol_version'] == manifest['protocol_version'], 'Protocol version mismatch')
    budget = pilot['budget']
    require(0 < budget['target_ratio'] < budget['trigger_ratio'] <= 1, 'Invalid budget thresholds')
    for spec in registry['presets']:
        if spec.get('prompt_ref'):
            require((ROOT / 'configs' / spec['prompt_ref']).exists(), 'Missing prompt')
    factorial = manifest['agent_factorial_template']
    require(factorial['strategy_spec_id'] in specs, 'Unregistered factorial strategy')
    require(factorial['orchestration_profile_id'] in orchestrations, 'Unregistered factorial orchestration')
    require({(c['external_enabled'], c['native_target_enabled']) for c in factorial['cells']}
            == {(False, False), (True, False), (False, True), (True, True)}, 'Incomplete factorial design')
    print('PASS: design JSON, filename links/anchors, registry references, declared factors, counts, budget thresholds, factorial coverage.')
    print('NOT CHECKED: full schemas, nested parameter equality, runtime isolation, agent capabilities, scientific validity.')
    print('READINESS: design_only; unresolved model/data/adapter fields are expected, not runnable.')


if __name__ == '__main__':
    main()
