"""Enumerate every paper cell, and run the implemented Toy/MuJoCo subset.

Unknown original budgets are explicit input, never silently inferred as facts.
"""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=['toy','mujoco','isaac','all'], default='all')
    parser.add_argument('--horizon', type=int, required=True)
    parser.add_argument('--output-dir', default='runs_paper')
    parser.add_argument('--manifest', default='paper_matrix.json')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--algorithms', nargs='+', choices=['sac','ppo','sql','flow'])
    args = parser.parse_args()
    protocol = json.loads((ROOT/'configs/paper_protocol.json').read_text())
    families = protocol['families'] if args.family=='all' else {args.family: protocol['families'][args.family]}
    rows=[]
    for family,tasks in families.items():
        for alg,task,drift,variant,seed in itertools.product(args.algorithms or protocol['algorithms'],tasks,protocol['drifts'],protocol['variants'],args.seeds or protocol['seeds']):
            directory=Path(args.output_dir).resolve()/family/alg/task/drift/variant/f'seed_{seed}'
            blocked=family=='isaac'
            command=[]
            if not blocked:
                script=ROOT/'cleanrl/cleanrl'/f'{alg}_continuous_action.py'
                command=[sys.executable,str(script),'--env-id',task,'--seed',str(seed),'--total-timesteps',str(args.horizon),
                         '--description',str(directory),'--use-nonstationary-env','--drift-pattern',drift]
                if alg=='ppo':
                    # Full rollouts must divide the requested horizon. This is a reconstruction setting.
                    if args.horizon % 1000:
                        raise ValueError('Paper matrix PPO requires a horizon divisible by 1000; custom smoke runs use the carrier directly')
                    command+=['--num-steps','1000','--num-minibatches','10']
                if variant=='aes': command+=['--use-aes']
                if args.no_cuda: command+=['--no-cuda']
            rows.append(dict(family=family,algorithm=alg,task=task,drift=drift,variant=variant,seed=seed,
                             status='blocked: Isaac simulator/evaluation integrations missing' if blocked else 'runnable; numerical replication unverified',
                             command=command,output=str(directory)))
    manifest=Path(args.manifest);manifest.parent.mkdir(parents=True,exist_ok=True)
    manifest.write_text(json.dumps({'protocol':protocol,'horizon':args.horizon,'runs':rows},indent=2)+'\n')
    print(f'{len(rows)} cells, {sum(bool(r["command"]) for r in rows)} runnable; manifest: {manifest}')
    if args.execute:
        if any(not r['command'] for r in rows):
            raise SystemExit('Selected matrix includes blocked cells. Select --family toy or mujoco; no partial matrix is launched silently.')
        for row in rows:
            print(' '.join(row['command']),flush=True)
            subprocess.run(row['command'],cwd=ROOT,check=True)


if __name__=='__main__':
    main()
