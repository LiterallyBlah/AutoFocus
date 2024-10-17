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
# Version: 1.1 (CLI enhancements)
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

def get_analysis_results(text, task):
    system_prompt = "You are an AI assistant specialised in analysing reconnaissance data for specific tasks supplied by the user. If there is no relevant data found, respond with 'false'."
    user_prompt = f"Analyse the following data for the task: '{task}'. \n\nData: {text}"
    
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

def analyse_directory(directory_path, tasks, window_size=500, step_size=250):
    results = {}
    
    # Walk through all directories and files recursively
    for root, dirs, files in os.walk(directory_path):
        target_name = os.path.basename(root)
        if target_name not in results:
            results[target_name] = {}
        print(f"{Fore.CYAN}Starting analysis of directory: {target_name}")
        
        # Loop through each file in the directory
        for file_name in files:
            file_path = os.path.join(root, file_name)
            print(f"{Fore.CYAN}Analyzing file: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            total_windows = (len(content) - window_size) // step_size + 1
            processed_windows = 0

            # Loop through content in chunks
            for i in range(0, len(content), step_size):
                window = content[i:i+window_size]
                
                # Perform each task on the chunk
                for task in tasks:
                    result = get_analysis_results(window, task['description'])
                    if result and result.lower() != "false":  # Check for no result response
                        if task['name'] not in results[target_name]:
                            results[target_name][task['name']] = []
                        results[target_name][task['name']].append({
                            "file": file_name,
                            "chunk_start": i,
                            "chunk_end": i + window_size,
                            "result": result
                        })
                
                processed_windows += 1
                progress = (processed_windows / total_windows) * 100 if total_windows > 0 else 100
                print(f"\r{Fore.CYAN}Progress: {progress:.2f}%", end="", flush=True)

    print(f"\n{Fore.GREEN}Analysis complete.")
    return results

def main():
    parser = argparse.ArgumentParser(description="AutoFocus Analysis Script for Recon Data Processing")
    parser.add_argument("-i", "--input", required=True, help="Path to the input directory containing recon data")
    parser.add_argument("-o", "--output", default="output", help="Directory to save the analysis results (default: output)")
    parser.add_argument("-t", "--tasks", required=True, help="Path to the tasks.yml file containing analysis tasks")
    parser.add_argument("-w", "--window_size", type=int, default=500, help="Size of the data chunk window for analysis (default: 500 characters)")
    parser.add_argument("-s", "--step_size", type=int, default=250, help="Step size for moving through data chunks (default: 250 characters)")
    args = parser.parse_args()

    # Normalise the path for both Windows and Linux
    input_directory = os.path.normpath(args.input)
    output_directory = os.path.normpath(args.output)
    tasks_file_path = os.path.normpath(args.tasks)
    
    if not os.path.exists(input_directory):
        print(f"{Fore.RED}Error: Directory not found")
        return
    
    if not os.path.exists(tasks_file_path):
        print(f"{Fore.RED}Error: Tasks file not found")
        return
    
    # Load tasks from tasks.yml
    with open(tasks_file_path, 'r', encoding='utf-8') as tasks_file:
        tasks = yaml.safe_load(tasks_file)['tasks']
    
    # Run the analysis
    results = analyse_directory(input_directory, tasks, window_size=args.window_size, step_size=args.step_size)
    
    # Generate a standardized output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"analysis_results_{timestamp}.json"
    output_path = os.path.join(output_directory, output_filename)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Output results to JSON file
    with open(output_path, 'w', encoding='utf-8') as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)
    
    print(f"{Fore.GREEN}Results saved to {output_path}")

if __name__ == "__main__":
    main()
