"""EFAS/SEAS downloader helper"""

import subprocess
import re

# help functions
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def get_regions(name):
    """Get lat-lon bounds of a region"""
    name = name.lower()
    if name == 'panaro':
        return (44, 45.2), (10.25, 11.75)
    if name == 'timis':
        return (44.6, 46.25), (20, 23)
    if name == 'lagen':
        return (60, 62.5), (7, 12)
    if name == 'aragon':
        return (42, 43.2), (-2.5, -0.25)
    if name == 'reno':
        return (43.9, 44.9), (10.75, 12.3)
    if name == 'turia':
        return (39, 41), (-2, 0)
    if name == 'euro':
        return (30, 70), (-60, 60)
    if name == 'global':
        return (24,72), (-35,75)

    raise ValueError(f"Region {name} not found")

def fixed_region(name, delta=0):
    """Get lat-lon bounds of a region with delta around the mean lat-lon"""
    lat, lon = get_regions(name)
    if name != 'global':
        lat, lon = (lat[0]-delta, lat[1]+delta), (lon[0]-delta, lon[1]+delta)
    return lat, lon

def mask_efas(field, regions):
    """Create a mask for the EFAS reforecast"""
    for region in regions:
        lat, lon = get_regions(region)
        if region == regions[0]:
            mask = ((field.longitude >= lon[0]) & (field.longitude <= lon[1]) & (field.latitude >= lat[0]) & (field.latitude <= lat[1]))
        else:
            mask = mask | ((field.longitude >= lon[0]) & (field.longitude <= lon[1]) & (field.latitude >= lat[0]) & (field.latitude <= lat[1]))

    return field.where(mask)

def is_job_running(job_name, username):
    """verify that a job name is not already submitted in the slurm queue"""
    # Run the squeue command to get the list of jobs
    output = subprocess.run(['squeue', '-u', username, '--format', '%j'],
                            capture_output=True, check=True)
    output = output.stdout.decode('utf-8').splitlines()[1:]

    # Parse the output to check if the job name is in the list
    return job_name in output

def submit_job(command):
    """Get the job id from the output of the sbatch command"""
    result = subprocess.run(command, check=True, capture_output=True).stdout.decode('utf-8')
    return re.findall(r'\b\d+\b', result)[-1]


def manage_parent_job(command, jobid, count=0, parent_job=[], parallel=1):
    """Manage the parent job for the SLURM submission"""

    if count!=0:
        print('Updating parent job to' + str(jobid))
        parent_job.append(str(jobid))
        if len(parent_job)>parallel:
            parent_job.pop(0)
    count += 1
    print(f'Parent jobs: {parent_job}')
    if parent_job and len(parent_job)==parallel:
        command.insert(5, f'--dependency={parent_job[0]}')

    return command, count, parent_job
