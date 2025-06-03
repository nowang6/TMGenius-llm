import asyncio
import os
import time
from pathlib import Path
import aiohttp
import json
from typing import List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
import argparse
from itertools import cycle

# Configuration
API_URL = "http://localhost:8000/transcribe"
DATA_DIR = "data"

# Performance test parameters
DEFAULT_CONCURRENT_REQUESTS = 2  # Number of concurrent requests
DEFAULT_REPEAT_TIMES = 1000  # Number of times to repeat each file
DEFAULT_CONN_LIMIT = 100  # aiohttp connection limit

console = Console()

async def process_file(session: aiohttp.ClientSession, file_path: Path, request_id: int) -> Dict:
    """Process a single audio file"""
    start_time = time.time()
    
    try:
        # Prepare the file for upload
        data = aiohttp.FormData()
        data.add_field('file',
                      open(file_path, 'rb'),
                      filename=file_path.name,
                      content_type='audio/mpeg')
        
        # Send request
        async with session.post(API_URL, data=data) as response:
            duration = time.time() - start_time
            
            if response.status == 200:
                result = await response.json()
                return {
                    'file_name': file_path.name,
                    'request_id': request_id,
                    'status': 'success',
                    'duration': duration,
                    'text': result['text'],
                    'emotion': result['emotion']
                }
            else:
                error_text = await response.text()
                return {
                    'file_name': file_path.name,
                    'request_id': request_id,
                    'status': 'error',
                    'duration': duration,
                    'error': f"HTTP {response.status}: {error_text}"
                }
                
    except Exception as e:
        duration = time.time() - start_time
        return {
            'file_name': file_path.name,
            'request_id': request_id,
            'status': 'error',
            'duration': duration,
            'error': str(e)
        }

async def process_batch(session: aiohttp.ClientSession, files: List[Path], start_id: int, progress) -> List[Dict]:
    """Process a batch of files concurrently"""
    tasks = []
    for i, file in enumerate(files):
        task = asyncio.create_task(process_file(session, file, start_id + i))
        tasks.append(task)
        progress.advance(task_id)
    return await asyncio.gather(*tasks)

async def process_all_files(concurrent_requests: int, repeat_times: int):
    """Process all audio files with controlled concurrency"""
    # Get all mp3 files
    data_path = Path(DATA_DIR)
    audio_files = list(data_path.glob("*.mp3"))
    
    if not audio_files:
        console.print("[red]No MP3 files found in the data directory![/red]")
        return
    
    # Create repeated file list
    repeated_files = []
    for _ in range(repeat_times):
        repeated_files.extend(audio_files)
    
    total_requests = len(repeated_files)
    console.print(f"[green]Preparing to process {len(audio_files)} files {repeat_times} times each ({total_requests} total requests)[/green]")
    console.print(f"[green]Concurrent requests: {concurrent_requests}[/green]")
    
    # Setup connection pool with limits
    conn = aiohttp.TCPConnector(limit=DEFAULT_CONN_LIMIT)
    timeout = aiohttp.ClientTimeout(total=60)  # 60 seconds timeout
    
    all_results = []
    with Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        global task_id
        task_id = progress.add_task("[cyan]Processing requests...", total=total_requests)
        
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            # Process files in batches
            for i in range(0, len(repeated_files), concurrent_requests):
                batch = repeated_files[i:i + concurrent_requests]
                results = await process_batch(session, batch, i, progress)
                all_results.extend(results)
    
    return all_results

def display_results(results: List[Dict]):
    """Display results in a formatted table with detailed statistics"""
    # Basic statistics
    success_count = sum(1 for r in results if r['status'] == 'success')
    total_count = len(results)
    durations = [r['duration'] for r in results]
    avg_duration = sum(durations) / total_count if durations else 0
    
    # Additional statistics
    min_duration = min(durations) if durations else 0
    max_duration = max(durations) if durations else 0
    
    # Calculate percentiles
    sorted_durations = sorted(durations)
    p50 = sorted_durations[int(len(sorted_durations) * 0.5)] if durations else 0
    p90 = sorted_durations[int(len(sorted_durations) * 0.9)] if durations else 0
    p95 = sorted_durations[int(len(sorted_durations) * 0.95)] if durations else 0
    p99 = sorted_durations[int(len(sorted_durations) * 0.99)] if durations else 0
    
    # Create statistics table
    stats_table = Table(title="Performance Statistics")
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="magenta")
    
    stats_table.add_row("Total Requests", str(total_count))
    stats_table.add_row("Success Rate", f"{success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
    stats_table.add_row("Average Time", f"{avg_duration:.3f}s")
    stats_table.add_row("Min Time", f"{min_duration:.3f}s")
    stats_table.add_row("Max Time", f"{max_duration:.3f}s")
    stats_table.add_row("Median (P50)", f"{p50:.3f}s")
    stats_table.add_row("P90", f"{p90:.3f}s")
    stats_table.add_row("P95", f"{p95:.3f}s")
    stats_table.add_row("P99", f"{p99:.3f}s")
    
    # Create success table
    success_table = Table(title="Sample Successful Transcriptions (showing first 10)")
    success_table.add_column("Request ID", style="dim")
    success_table.add_column("File Name", style="cyan")
    success_table.add_column("Duration (s)", style="magenta")
    success_table.add_column("Emotion", style="green")
    success_table.add_column("Text", style="blue")
    
    # Create error table
    error_table = Table(title="Failed Transcriptions", style="red")
    error_table.add_column("Request ID", style="dim")
    error_table.add_column("File Name", style="cyan")
    error_table.add_column("Duration (s)", style="magenta")
    error_table.add_column("Error", style="red")
    
    # Populate tables
    success_count = 0
    for result in results:
        if result['status'] == 'success':
            if success_count < 10:  # Only show first 10 successful results
                success_table.add_row(
                    str(result['request_id']),
                    result['file_name'],
                    f"{result['duration']:.3f}",
                    result['emotion'],
                    result['text'][:100] + ('...' if len(result['text']) > 100 else '')
                )
                success_count += 1
        else:
            error_table.add_row(
                str(result['request_id']),
                result['file_name'],
                f"{result['duration']:.3f}",
                result['error']
            )
    
    # Display tables
    console.print("\n[bold]Test Results:[/bold]")
    console.print(stats_table)
    console.print(success_table)
    if any(r['status'] == 'error' for r in results):
        console.print(error_table)

async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='API Performance Test')
    parser.add_argument('-c', '--concurrent', type=int, default=DEFAULT_CONCURRENT_REQUESTS,
                      help=f'Number of concurrent requests (default: {DEFAULT_CONCURRENT_REQUESTS})')
    parser.add_argument('-r', '--repeat', type=int, default=DEFAULT_REPEAT_TIMES,
                      help=f'Number of times to repeat each file (default: {DEFAULT_REPEAT_TIMES})')
    args = parser.parse_args()
    
    console.print("[bold blue]Starting API Performance Test...[/bold blue]")
    
    # Check if API is available by sending a small test request
    try:
        test_file = Path(DATA_DIR) / "快乐2.mp3"  # Use an existing file for testing
        if not test_file.exists():
            console.print("[red]Test file not found. Please check the data directory.[/red]")
            return
            
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field('file',
                          open(test_file, 'rb'),
                          filename=test_file.name,
                          content_type='audio/mpeg')
            
            async with session.post(API_URL, data=data) as response:
                if response.status != 200:
                    console.print("[red]API test failed. Please check if the API server is running correctly.[/red]")
                    return
                console.print("[green]API is available and working correctly.[/green]")
    except Exception as e:
        console.print(f"[red]Could not connect to API: {str(e)}[/red]")
        console.print("[red]Please make sure the API server is running at http://localhost:8000[/red]")
        return
    
    # Process files
    start_time = time.time()
    results = await process_all_files(args.concurrent, args.repeat)
    total_time = time.time() - start_time
    
    if results:
        display_results(results)
        
        # Save results to file
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = f"test_results_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'test_config': {
                    'concurrent_requests': args.concurrent,
                    'repeat_times': args.repeat,
                    'total_time': total_time
                },
                'results': results
            }, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Results saved to {output_file}[/green]")

if __name__ == "__main__":
    asyncio.run(main()) 
