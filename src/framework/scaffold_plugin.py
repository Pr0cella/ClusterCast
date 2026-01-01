from __future__ import annotations

"""
CLI to scaffold a new plugin model file under src/plugins/models/.

Usage (from src directory):
  python -m framework.scaffold_plugin --id my_model --name "My Model" --class-name MyModel

Options:
  --out-dir: target directory (default: src/plugins/models)
  --force: overwrite if file exists
"""

import os
import argparse
import re
from datetime import datetime
from string import Template


TEMPLATE = """from __future__ import annotations

from typing import Any, List
import numpy as np
from framework.base import ModelInterface, SearchParam


class $class_name(ModelInterface):
    id = '$model_id'
    name = '$model_name'
    version = '0.1.0'

    def required_features(self) -> List[str]:
        return ['week', 'Target_V']

    def get_search_space(self):
        return [
            SearchParam('param', kind='float', low=0.0, high=1.0, step=0.1),
        ]

    def fit(self, train_df):
        # Example: baseline probability per target as state
        lam = train_df.mean(axis=0).values
        p = 1 - np.exp(-lam)
        return {'p_base': p}

    def predict_proba(self, state: Any, context: dict) -> np.ndarray:
        # Example: use baseline state (and optional param) to produce probabilities
        param = float(self.get_params().get('param', 0.0))
        p = np.clip(state['p_base'] + param, 0.0, 1.0)
        return p
"""


def to_class_name(model_id: str) -> str:
    # Convert snake/kebab to PascalCase
    name = re.sub(r"[^a-zA-Z0-9]+", " ", model_id).title().replace(" ", "")
    if not name or not name[0].isalpha():
        name = f"Model{name}"
    return name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Scaffold a new plugin model under src/plugins/models/")
    parser.add_argument('--id', required=True, help='Model id (e.g. my_model)')
    parser.add_argument('--name', default=None, help='Human-readable model name')
    parser.add_argument('--class-name', default=None, help='Python class name (default derived from id)')
    parser.add_argument('--out-dir', default=os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'plugins', 'models')),
                        help='Output directory for plugin file')
    parser.add_argument('--force', action='store_true', help='Overwrite if file already exists')
    args = parser.parse_args(argv)

    model_id = args.id.strip()
    model_name = args.name or model_id.replace('_', ' ').title()
    class_name = args.class_name or to_class_name(model_id)

    # Validate identifiers
    if not re.match(r'^[a-zA-Z0-9_\-]+$', model_id):
        print("[ERROR] --id must contain only letters, numbers, underscore, or dash")
        return 2
    if not re.match(r'^[A-Za-z][A-Za-z0-9_]*$', class_name):
        print("[ERROR] --class-name must be a valid Python class identifier")
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    filename = f"{model_id}.py"
    out_path = os.path.join(args.out_dir, filename)

    if os.path.exists(out_path) and not args.force:
        print(f"[ERROR] File already exists: {out_path} (use --force to overwrite)")
        return 1

    content = Template(TEMPLATE).substitute(class_name=class_name, model_id=model_id, model_name=model_name)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(content)

    print("Created plugin model:")
    print(" - File:", out_path)
    print(" - Class:", class_name)
    print(" - Id:", model_id)
    print("Next steps:")
    print("  1) Edit the file to implement your model logic (fit/predict_proba/get_search_space).")
    print("  2) List models: python -m framework.modular_runner --list-models")
    print(f"  3) Run: python -m framework.modular_runner --model {model_id} --horizon 13 --train-window 26 --opt-budget 3")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
