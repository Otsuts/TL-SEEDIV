import subprocess
import sys

PREFIX = ['python', 'main.py']
MODEL = sys.argv[1]
for LR in ['5e-4', '1e-4', '5e-5', '1e-5']:
    for PR in ['5e-4', '1e-4', '5e-5', '1e-5']:
        for WD in ['0.0', '1e-4', '1e-5', '1e-6']:
            subprocess.run(PREFIX+[
                f'--model={MODEL}',
                f'--learn_rate={LR}',
                f'--pretrain_lr={PR}',
                f'--weight_decay={WD}'
            ])
