#!/usr/bin/env python3

import subprocess
from download_functions import is_job_running, submit_job, manage_parent_job

USERNAME='davini'
# Define the list of years and regions to loop through
#years = range(2004, 2005)
years = range(1999, 2024)
MODE = "EFAS5"
months = range(1, 13)
PARALLEL = 8 # amount of parallel job allowed

if MODE == "EFAS5":
    regions = ['Panaro', 'Timis', 'Lagen', 'Aragon', 'Reno', 'Turia']
    regions = ['Reno', 'Timis', 'Turia']
elif MODE == "SEAS5":
    regions = ['Euro']
else:
    raise ValueError(f"Unknown MODE: {MODE}")

# SLURM submission script
slurm_script = 'EFAS_seasonal_v2.py'  # The SLURM script

# Loop through each year and region, submitting a job for each combination
count = 0 # to count job
jobid = None # to define
parent_job = [] # to define
for year in years:
    for region in regions:
        for month in months:
            jobname = f'{MODE}_{year}{str(month).zfill(2)}_{region}'
            # Create the sbatch command with the year and region as arguments
            command = [
                'sbatch',
                f'--job-name={jobname}',            # Job name
                f'--output=/home/davini/log/{jobname}-%j.out', # Output file
                f'--error=/home/davini/log/{jobname}-%j.err',  # Error file
                '--mem=4000M',                      # Memory
                '--time=48:00:00',                  # Max run time
                '--partition=batch',                # SLURM partition
                '--wrap',                           # Wrap the Python command within SLURM
                f'python {slurm_script} {MODE} --year {year} --region {region} -m {month} -c'  # Command to run
            ]
            
            if is_job_running(jobname, USERNAME):
                print(f'Job {jobname} is already running, skipping...')
                continue
                # Submit the job to SLURM
            try:
                
                # Check if the parent job is running and update the command
                command, count, parent_job = manage_parent_job(command, jobid, count=count,
                                                               parent_job=parent_job, 
                                                               parallel=PARALLEL)

                # Print the command to the terminal for debugging purposes
                print(f"Submitting job for year: {year}, Region: {region}, Month: {month}")
                
                jobid = submit_job(command)
                print(f"Job submitted successfully: {jobid}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")
