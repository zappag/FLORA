#!/usr/bin/env python3

import subprocess
import re

USERNAME='davini'
# Define the list of years and regions to loop through
years = range(1999, 2002)
regions = ['Euro']
ensembles = range(25)
PARALLEL = 20 # amount of parallel job allowed

# SLURM submission script
slurm_script = 'MARS_seasonal.py'  # The SLURM script you created earlier

def is_job_running(job_name, username):
    """verify that a job name is not already submitted in the slurm queue"""
    # Run the squeue command to get the list of jobs
    output = subprocess.run(['squeue', '-u', username, '--format', '%j'],
                            capture_output=True, check=True)
    output = output.stdout.decode('utf-8').splitlines()[1:]

    # Parse the output to check if the job name is in the list
    return job_name in output

# Loop through each year and region, submitting a job for each combination
COUNT = 0 # to count job
jobid = None
PARENT_JOB = [] # to define
for year in years:
    for region in regions:
        for ensemble in ensembles:
            # Create the sbatch command with the year and region as arguments
            jobname = f'MARS_{year}_{region}_{ensemble}'
            command = [
                'sbatch',
                f'--job-name={jobname}',            # Job name
                f'--output=/home/davini/log/MARS-{year}-{region}-{ensemble}-%j.out', # Output file
                f'--error=/home/davini/log/MARS-{year}-{region}-{ensemble}-%j.err',  # Error file
                '--mem=4000M',                      # Memory
                '--time=48:00:00',                  # Max run time
                '--partition=batch',                # SLURM partition
                '--wrap',                           # Wrap the Python command within SLURM
                f'python {slurm_script} --year {year} --region {region} --ensemble {ensemble} -c'  # Command to run
            ]

            if is_job_running(jobname, USERNAME):
                print(f'Job {jobname} is already running, skipping...')
                continue
                # Submit the job to SLURM
            try:
                if COUNT!=0:
                    print('Updating parent job to' + str(jobid))
                    PARENT_JOB.append(str(jobid))
                    if len(PARENT_JOB)>PARALLEL:
                        PARENT_JOB.pop(0)
                COUNT = COUNT + 1
                print(f'Parent jobs: {PARENT_JOB}')
                if PARENT_JOB and len(PARENT_JOB)==PARALLEL:
                    command.insert(5, f'--dependency={PARENT_JOB[0]}')

                # Print the command to the terminal for debugging purposes
                print(f"Submitting job for year: {year}, Region: {region}, Ensemble: {ensemble}")

                result = subprocess.run(command, check=True, capture_output=True).stdout.decode('utf-8')
                jobid = re.findall(r'\b\d+\b', result)[-1]
                print(f"Job submitted successfully: {jobid}")
            except subprocess.CalledProcessError as e:
                print(f"Error submitting job: {e.stderr}")


