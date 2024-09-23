#!/usr/bin/env python3

import subprocess

# Define the list of years and regions to loop through
years = range(2006, 2007)
regions = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno']

# SLURM submission script
slurm_script = 'EFAS_seasonal.py'  # The SLURM script you created earlier

# Loop through each year and region, submitting a job for each combination
for year in years:
    for region in regions:
        # Create the sbatch command with the year and region as arguments
        command = [
            'sbatch',
            f'--job-name=EFAS_{year}_{region}',            # Job name
            f'--output=/home/davini/EFAS-{year}-{region}-%j.out', # Output file
            f'--error=/home/davini/EFAS-{year}-{region}-%j.err',  # Error file
            '--mem=4000M',                      # Memory
            '--time=48:00:00',                  # Max run time
            '--partition=batch',                # SLURM partition
            '--wrap',                           # Wrap the Python command within SLURM
            f'python EFAS_seasonal.py --year {year} --region {region}'  # Command to run
        ]
        
        # Print the command to the terminal for debugging purposes
        print(f"Submitting job for Year: {year}, Region: {region}")
        
        # Submit the job to SLURM
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Job submitted successfully: {result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Error submitting job: {e.stderr}")
