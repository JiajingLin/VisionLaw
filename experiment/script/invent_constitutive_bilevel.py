from pathlib import Path
import subprocess
import os
import sys
from datetime import datetime

from visionlaw.utils import get_root, get_script_parser, dict_to_cmds

root = get_root(__file__)

def main():
    python_path = Path(sys.executable).resolve() 
    program_path = root / 'entry' / 'agent_vision_bilevel.py'  
    base_cmds = [python_path, program_path] 

    parser = get_script_parser()
    parser.add_argument("-c", "--config", type=str, default="experiment/configs/finetune-bb.yaml", help="Path to the config file.")
    parser.add_argument('--llm', type=str, default='openai-gpt-4.1-mini', 
    choices=[
        'openai-gpt-4.1-mini',
        'openai-gpt-4-o3-mini',
        'openai-gpt-4-o4-mini',   
        'openai-gpt-4.1-nano',  
        'openai-gpt-4-1106-preview',
        'openai-gpt-3.5-turbo-0125',
        'mistral-open-mixtral-8x7b',
        'anthropic-claude-3-sonnet-20240229',
    ]) 
    base_args = parser.parse_args() 
    base_args = vars(base_args) 

    name = base_args['config'].split('/')[-1].split('.')[0].split('-')[1]

    my_env = os.environ.copy()

    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    for seed in range(5):
        args = base_args | {
            'seed': seed,
            'path': f'{base_args["llm"]}/invent_constitutive_from_linear_{name}_bilevel/{seed:04d}',
            'llm.primitives': '(elastoplasticity)',
            'physics.env.physics': 'elastoplasticity',
        }

        cmds = base_cmds + dict_to_cmds(args)
        str_cmds = [str(cmd) for cmd in cmds]
        subprocess.run(str_cmds, shell=False, check=False, env=my_env)

if __name__ == '__main__':
    main()
