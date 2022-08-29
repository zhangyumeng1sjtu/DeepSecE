import os
from argparse import ArgumentParser

import pandas as pd


def read_system_names(summary_file):
    system2label = {'T1SS': 'I', 'T2SS': 'II',
                    'T3SS': 'III', 'T4SS': 'IV', 'T6SS': 'VI'}
    summary = pd.read_table(summary_file, comment='#')
    systems = set(item[5:9] for item in summary.iloc[0, 1:]
                  [summary.iloc[0, 1:] > 0].keys())
    return sorted([system2label[system] for system in list(systems)])


def main(args):
    
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
        
    macsyfinde_cmd = f"macsyfinder --db-type ordered_replicon --sequence-db {args.fasta_path} " \
        f"--models-dir {args.data_dir} --models TXSS T1SS T2SS T3SS T4SS_typeI T4SS_typeT T6SSi T6SSii T6SSiii " \
        f"-o {args.out_dir}/TXSS -w {args.num_workers}"
    
    print(macsyfinde_cmd)
    os.system(macsyfinde_cmd)
    systems = read_system_names(f'{args.out_dir}/TXSS/best_solution_summary.tsv')

    if len(systems) > 0:
        print(f'Type {", ".join(systems)} secretion system(s) are found.')
        pred_cmd = f"python predict.py --fasta_path {args.fasta_path} --model_location {args.model_location} " \
            f"--secretion_systems {' '.join(systems)} --out_dir {args.out_dir}"
        if args.save_attn:
            pred_cmd += " --save_attn"
        if args.no_cuda:
            pred_cmd += " --no_cuda"
        print(pred_cmd)
        os.system(pred_cmd)
    else:
        print(f'No secretion system is found in the input fasta file.')


if __name__ == '__main__':

    parser = ArgumentParser(
        description="Predict secretion systems and secreted effectors from ordered protein sequences in a FASTA file.")
    
    parser.add_argument('--fasta_path', required=True, type=str,
                        help='input ordered protein sequences.')
    parser.add_argument('--model_location', required=True, type=str,
                        help='path to the model weights.')
    parser.add_argument('--data_dir', default='./data', type=str,
                        help="directory containing TXSS data. (default: ./data)")
    parser.add_argument('--out_dir', default='./', type=str,
                        help='output directory of prediction results.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='num. of workers used in MacSyFinder. (default: 4)')
    parser.add_argument('--save_attn', action='store_true',
                        help='save the sequence attention of effectors.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='add when CUDA is not available.')

    args = parser.parse_args()

    main(args)
