#!/usr/bin/env python3
"""
SLURM utilities for managing jobs and checking queue status.
Created during ScalarLM refactoring to help debug training job issues.
"""

import requests
import json
import argparse
import sys
from typing import Dict, List, Any

def check_slurm_status(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Check SLURM queue status via ScalarLM API."""
    try:
        response = requests.get(f"{base_url}/slurm/status")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"❌ Error checking SLURM status: {e}")
        return {}

def cancel_slurm_jobs(job_ids: List[str], base_url: str = "http://localhost:8000") -> bool:
    """Cancel SLURM jobs by ID."""
    success = True
    for job_id in job_ids:
        try:
            response = requests.post(f"{base_url}/slurm/cancel/{job_id}")
            response.raise_for_status()
            print(f"✅ Cancelled job {job_id}")
        except requests.RequestException as e:
            print(f"❌ Error cancelling job {job_id}: {e}")
            success = False
    return success

def get_job_details(job_id: str, base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Get detailed information about a specific SLURM job."""
    try:
        response = requests.get(f"{base_url}/slurm/job/{job_id}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"❌ Error getting job details for {job_id}: {e}")
        return {}

def parse_squeue_output(squeue_text: str) -> List[Dict[str, str]]:
    """Parse raw squeue output into job list."""
    jobs = []
    lines = squeue_text.strip().split('\n')
    
    # Skip header line
    if len(lines) <= 1:
        return jobs
    
    for line in lines[1:]:  # Skip header
        if not line.strip():
            continue
            
        # Parse squeue format: JOBID PARTITION NAME USER STATE TIME TIME_LIMI NODES NODELIST
        parts = line.split()
        if len(parts) >= 8:
            jobs.append({
                'JobId': parts[0],
                'Partition': parts[1], 
                'JobName': parts[2],
                'User': parts[3],
                'JobState': parts[4],
                'Time': parts[5],
                'TimeLimit': parts[6],
                'NumNodes': parts[7],
                'NodeList': ' '.join(parts[8:]) if len(parts) > 8 else ''
            })
    
    return jobs

def print_queue_status(status: Dict[str, Any]):
    """Print formatted queue status."""
    if not status:
        print("❌ No status data available")
        return
    
    # Parse jobs from squeue output
    queue_data = status.get('queue', {})
    squeue_output = queue_data.get('squeue_output', '')
    error_message = queue_data.get('error_message')
    
    if error_message:
        print(f"❌ SLURM Error: {error_message}")
        return
    
    if not squeue_output:
        print("✅ No jobs in queue")
        return
    
    jobs = parse_squeue_output(squeue_output)
    
    if not jobs:
        print("✅ No jobs in queue")
        return
    
    print(f"📊 SLURM Queue Status ({len(jobs)} jobs)")
    print("-" * 80)
    print(f"{'Job ID':<10} {'Name':<20} {'State':<12} {'User':<10} {'Partition':<12} {'Nodes':<6} {'Time':<10}")
    print("-" * 80)
    
    running_jobs = 0
    pending_jobs = 0
    
    for job in jobs:
        job_id = job.get('JobId', 'N/A')
        name = job.get('JobName', 'N/A')[:19]  # Truncate long names
        state = job.get('JobState', 'N/A')
        user = job.get('User', 'N/A')
        partition = job.get('Partition', 'N/A')
        nodes = job.get('NumNodes', 'N/A')
        time = job.get('Time', 'N/A')
        
        # Color coding for different states
        if state == 'RUNNING':
            running_jobs += 1
            state_display = f"🟢 {state}"
        elif state in ['PENDING', 'QUEUED']:
            pending_jobs += 1
            state_display = f"🟡 {state}"
        elif state == 'COMPLETED':
            state_display = f"✅ {state}"
        elif state in ['FAILED', 'CANCELLED']:
            state_display = f"❌ {state}"
        else:
            state_display = f"🔵 {state}"
        
        print(f"{job_id:<10} {name:<20} {state_display:<20} {user:<10} {partition:<12} {nodes:<6} {time:<10}")
    
    print("-" * 80)
    print(f"Summary: {running_jobs} running, {pending_jobs} pending")
    
    if pending_jobs > 0:
        print(f"⚠️  Warning: {pending_jobs} jobs pending - potential resource contention")

def main():
    parser = argparse.ArgumentParser(description="SLURM utilities for ScalarLM")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check SLURM queue status')
    check_parser.add_argument('--url', default='http://localhost:8000', 
                            help='Base URL for ScalarLM API')
    
    # Cancel command
    cancel_parser = subparsers.add_parser('cancel', help='Cancel SLURM jobs')
    cancel_parser.add_argument('job_ids', nargs='+', help='Job IDs to cancel')
    cancel_parser.add_argument('--url', default='http://localhost:8000',
                             help='Base URL for ScalarLM API')
    
    # Details command
    details_parser = subparsers.add_parser('details', help='Get job details')
    details_parser.add_argument('job_id', help='Job ID to get details for')
    details_parser.add_argument('--url', default='http://localhost:8000',
                               help='Base URL for ScalarLM API')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'check':
        status = check_slurm_status(args.url)
        print_queue_status(status)
    
    elif args.command == 'cancel':
        success = cancel_slurm_jobs(args.job_ids, args.url)
        if not success:
            sys.exit(1)
    
    elif args.command == 'details':
        details = get_job_details(args.job_id, args.url)
        if details:
            print(f"📋 Job Details for {args.job_id}")
            print(json.dumps(details, indent=2))
        else:
            sys.exit(1)

if __name__ == '__main__':
    main()