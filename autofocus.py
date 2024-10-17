# ============================================================
# AutoFocus Analysis Script for Recon Data Processing
# ============================================================
# This script is designed to analyse reconnaissance data collected by AutoRecon.
# It utilises a local LLM to process chunks of data for identifying version numbers
# and points of interest, streamlining the reconnaissance process for penetration testing.
#
# Key features:
# - Processes files in manageable chunks to handle large datasets
# - Uses AI-powered analysis to detect potential version numbers and points of interest
# - Provides progress updates during the scanning process
# - Outputs findings in a structured JSON format
#
# Author: Michael Aguilera
# Date: 17/10/2024
# Version: 1.99 (Added deduplication and task name in output)
# ============================================================

import os
import ollama
import json
import yaml
import argparse
from datetime import datetime
from colorama import Fore, init
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get model from environment variable
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'qwen2.5')

# Initialise colorama for coloured output
init(autoreset=True)

def validate_tasks_file(tasks_file_path):
    """
    Validate the structure of the tasks.yml file to ensure it matches the expected format.
    """
    try:
        with open(tasks_file_path, 'r', encoding='utf-8') as tasks_file:
            tasks_data = yaml.safe_load(tasks_file)
            
            if 'tasks' not in tasks_data:
                raise ValueError("Missing 'tasks' key in tasks.yml file.")
            
            required_fields = ['name', 'description', 'response', 'output']
            for task in tasks_data['tasks']:
                for field in required_fields:
                    if field not in task:
                        raise ValueError(f"Each task must contain '{field}' field.")
                    if not isinstance(task[field], str):
                        raise ValueError(f"The '{field}' field must be a string.")
        
        print(f"{Fore.GREEN}Tasks file validation passed.")
        return True
    except Exception as e:
        print(f"{Fore.RED}Error: Invalid tasks.yml file - {str(e)}")
        return False

def get_analysis_results(text, task, previous_results):
    system_prompt = "You are an AI assistant specialised in analysing reconnaissance data for specific tasks supplied by the user. If there is no relevant data found, respond with 'false'."
    user_prompt = f"Analyse the following data for the task: '{task['description']}'.\n\nPreviously found results: {previous_results}\n\nNew Data: {text}\n\n{task['response']}\nPlease avoid providing information that has already been identified.\n\nIf there is no relevant data found, respond with 'false'."
    
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                'role': 'system',
                'content': system_prompt,
            },
            {
                'role': 'user',
                'content': user_prompt,
            },
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"{Fore.RED}Error: Unable to get response from Ollama - {str(e)}")
        return None

def analyse_directory(directory_path, tasks, blacklist_dirs, blacklist_file_types, whitelist_file_types, output_path, window_size=500, step_size=250):
    results = {}
    
    # Walk through all directories and files recursively
    for root, dirs, files in os.walk(directory_path):
        if any(blacklisted in root for blacklisted in blacklist_dirs):
            print(f"{Fore.YELLOW}Skipping blacklisted directory: {root}")
            continue
        
        target_name = os.path.basename(root)
        if target_name not in results:
            results[target_name] = {}
        print(f"{Fore.CYAN}Starting analysis of directory: {target_name}")
        
        # Loop through each file in the directory
        for file_name in files:
            if whitelist_file_types:
                if not any(file_name.endswith(whitelist_type) for whitelist_type in whitelist_file_types):
                    print(f"{Fore.YELLOW}Skipping non-whitelisted file type: {file_name}")
                    continue
            elif any(file_name.endswith(blacklisted_type) for blacklisted_type in blacklist_file_types):
                print(f"{Fore.YELLOW}Skipping blacklisted file type: {file_name}")
                continue
            
            file_path = os.path.join(root, file_name)
            print(f"{Fore.CYAN}Analyzing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            total_windows = (len(content) - window_size) // step_size + 1
            processed_windows = 0
            previous_results = []

            # Loop through content in chunks
            for i in range(0, len(content), step_size):
                window = content[i:i+window_size]
                
                # Perform each task on the chunk
                for task in tasks:
                    result = get_analysis_results(window, task, previous_results)
                    if result and result.lower() != "false":  # Check for no result response
                        if task['name'] not in results[target_name]:
                            results[target_name][task['name']] = []
                        if result not in previous_results:
                            results[target_name][task['name']].append({
                                "file": file_name,
                                "chunk_start": i,
                                "chunk_end": i + window_size,
                                "task_name": task['name'],
                                "result": result
                            })
                            previous_results.append(result)
                
                processed_windows += 1
                progress = (processed_windows / total_windows) * 100 if total_windows > 0 else 100
                print(f"\r{Fore.CYAN}Progress: {progress:.2f}%", end="", flush=True)
            
            # Save progress after each file is processed
            with open(output_path, 'w', encoding='utf-8') as json_file:
                json.dump(results, json_file, indent=4, ensure_ascii=False)
                print(f"\n{Fore.GREEN}Intermediate results saved to {output_path}")

    print(f"\n{Fore.GREEN}Analysis complete.")
    return results

def main():
    parser = argparse.ArgumentParser(description="AutoFocus Analysis Script for Recon Data Processing")
    parser.add_argument("-i", "--input", required=True, help="Path to the input directory containing recon data")
    parser.add_argument("-o", "--output", default="output", help="Directory to save the analysis results (default: output)")
    parser.add_argument("-t", "--tasks", required=True, help="Path to the tasks.yml file containing analysis tasks")
    parser.add_argument("-b", "--blacklist", nargs='*', default=["exploit", "loot", "report"], help="List of directories to blacklist from analysis (default: exploit, loot, report)")
    parser.add_argument("-bt", "--blacklist_file_types", nargs='*', default=[], help="List of file types to blacklist from analysis (e.g., .log, .tmp)")
    parser.add_argument("-wt", "--whitelist_file_types", nargs='*', default=[], help="List of file types to whitelist for analysis (e.g., .txt, .json)")
    parser.add_argument("-w", "--window_size", type=int, default=500, help="Size of the data chunk window for analysis (default: 500 characters)")
    parser.add_argument("-s", "--step_size", type=int, default=250, help="Step size for moving through data chunks (default: 250 characters)")
    args = parser.parse_args()

    # Ensure blacklist and whitelist are not used together
    if args.blacklist_file_types and args.whitelist_file_types:
        print(f"{Fore.RED}Error: Cannot use both blacklist and whitelist for file types at the same time")
        return

    # Normalise the path for both Windows and Linux
    input_directory = os.path.normpath(args.input)
    output_directory = os.path.normpath(args.output)
    tasks_file_path = os.path.normpath(args.tasks)
    blacklist_dirs = [os.path.normpath(blacklisted) for blacklisted in args.blacklist]
    blacklist_file_types = args.blacklist_file_types
    whitelist_file_types = args.whitelist_file_types
    
    if not os.path.exists(input_directory):
        print(f"{Fore.RED}Error: Directory not found")
        return
    
    if not os.path.exists(tasks_file_path):
        print(f"{Fore.RED}Error: Tasks file not found")
        return

    # Validate tasks file
    if not validate_tasks_file(tasks_file_path):
        return

    # Load tasks from tasks.yml
    with open(tasks_file_path, 'r', encoding='utf-8') as tasks_file:
        tasks = yaml.safe_load(tasks_file)['tasks']
    
    # Generate a standardized output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"analysis_results_{timestamp}.json"
    output_path = os.path.join(output_directory, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)
    
    # Run the analysis with continuous updates
    analyse_directory(input_directory, tasks, blacklist_dirs, blacklist_file_types, whitelist_file_types, output_path, window_size=args.window_size, step_size=args.step_size)

if __name__ == "__main__":
    main()
