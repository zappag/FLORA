#!/usr/bin/env python3

import subprocess

mode = 'EFAS'
#mode = 'SEAS5'

# Define the list of ensembles and regions to loop through
ensembles = range(0, 25)
#ensembles = [0, 1]

#regions = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno', 'Turia']
regions = ['Panaro']

surrogates = ['monthly', 'trimestral', 'quadrimestral']
#modes = ['monthly']

# SLURM submission script
slurm_script = 'EFAS_surrogate.py'  # The SLURM script you created earlier

if mode == 'EFAS':
    variables = ['dis24']
elif mode == 'SEAS5':
    variables = ['msl']
else:
    raise KeyError(f'Cannot recognize {mode} mode')

# Loop through each ensemble and region, submitting a job for each combination
for ensemble in ensembles:
    for region in regions:
        for surrogate in surrogates:
            for variable in variables:
            # Create the sbatch command with the ensemble and region as arguments
                command = [
                    'sbatch',
                    f'--job-name={mode}_{ensemble}_{region}',            # Job name
                    f'--output=/home/davini/log/{mode}-{ensemble}-{region}-%j.out', # Output file
                    f'--error=/home/davini/log/{mode}-{ensemble}-{region}-%j.err',  # Error file
                    '--mem=4000M',                      # Memory
                    '--time=01:00:00',                  # Max run time
                    '--partition=batch',                # SLURM partition
                    '--wrap',                           # Wrap the Python command within SLURM
                    f'python {slurm_script} {mode} --region {region} --ensemble {ensemble} --mode {mode} --variable {variable}'  # Command to run
                ]
                
                # Print the command to the terminal for debugging purposes
                print(f"Submitting job for {mode} for ensemble: {ensemble}, Region: {region}")
                
                # Submit the job to SLURM
                try:
                    result = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"Job submitted successfully: {result.stdout}")
                except subprocess.CalledProcessError as e:
                    print(f"Error submitting job: {e.stderr}")
