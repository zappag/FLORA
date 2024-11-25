#!/usr/bin/env python3

import subprocess

# Define the list of ensembles and regions to loop through
ensembles = range(0, 25)
#ensembles = [0, 1]

#regions = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno', 'Turia']
regions = ['Panaro']

modes = ['monthly', 'trimestral', 'quadrimestral']
#modes = ['monthly']

# SLURM submission script
slurm_script = 'EFAS_surrogate.py'  # The SLURM script you created earlier

# Loop through each ensemble and region, submitting a job for each combination
for ensemble in ensembles:
    for region in regions:
        for mode in modes:
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
                f'python {slurm_script} {region} --ensemble {ensemble} --mode {mode}'  # Command to run
            ]
            
            # Print the command to the terminal for debugging purposes
            print(f"Submitting job for {mode} for ensemble: {ensemble}, Region: {region}")
            
            # Submit the job to SLURM
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Job submitted successfully: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")
