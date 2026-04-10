from pathlib import Path
import subprocess
import os
import sys
from visionlaw.utils import get_root, get_script_parser, dict_to_cmds

import argparse

root = get_root(__file__)

def main():
    python_path = Path(sys.executable).resolve()
    program_path = root / 'entry' / 'vision' / 'eval.py'
    base_cmds = [python_path, program_path]


    parser = get_script_parser()
    parser.add_argument("-c", "--config", type=str, default="experiment/configs/finetune-bb.yaml", help="Path to the config file.")
    base_args = parser.parse_args()
    base_args = vars(base_args)

    base_args['overwrite'] = True

    my_env = os.environ.copy()


    args = base_args | {
        'path': f'experiment/log/demo',
        'llm.primitives': '(identity)',
        'llm.entry': 'plasticity',
        'optim.alpha_position': 1e5,
        'optim.alpha_velocity': 1e1,
        'physics.env.physics': 'identity'
    }

    cmds = base_cmds + dict_to_cmds(args)
    str_cmds = [str(cmd) for cmd in cmds]
    subprocess.run(str_cmds, shell=False, check=False, env=my_env)

if __name__ == '__main__':
    main()
