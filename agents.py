import ollama
import json
import os
import re
import logging
from colorama import Fore
from Levenshtein import distance as levenshtein_distance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global flag to enable/disable logging
LOGGING_ENABLED = True

def log(level, message):
    if LOGGING_ENABLED:
        if level == 'INFO':
            logger.info(message)
        elif level == 'DEBUG':
            logger.debug(message)
        elif level == 'WARNING':
            logger.warning(message)
        elif level == 'ERROR':
            logger.error(message)

# Agent Definitions
class InitialAnalysisAgent:
    def __init__(self, model):
        self.model = model
        log('INFO', f"InitialAnalysisAgent initialized with model: {model}")

    def analyse(self, text, task):
        log('INFO', f"Starting analysis for task: {task['name']}")
        system_prompt = "You are a penetration testing AI assistant specialised in analysing reconnaissance data for specific tasks supplied by the user."
        user_prompt = f"Analyse the following data for the task: '{task['description']}'.\n\nNew Data: {text}\n\nIf there is no relevant data found, respond with 'irrelevant'. Be concise with your response and if there are multiple results, return them as a list."
        try:
            log('DEBUG', f"Sending request to Ollama with system prompt: {system_prompt}")
            response = ollama.chat(model=self.model, messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ])
            result = response['message']['content'].strip()
            log('INFO', f"Analysis completed for task: {task['name']}")
            return result
        except Exception as e:
            log('ERROR', f"Error in InitialAnalysisAgent: {str(e)}")
            print(f"\n{Fore.RED}Error: Unable to get response from InitialAnalysisAgent - {str(e)}")
            return None


class VerificationAgent:
    def __init__(self, model):
        self.model = model
        log('INFO', f"VerificationAgent initialized with model: {model}")

    def verify(self, analysis_results, task):
        log('INFO', f"Starting verification for task: {task['name']}")
        system_prompt = "You are an AI assistant specialised in verifying and formatting reconnaissance analysis results. Your primary goal is to ensure the output strictly adheres to the specified format."
        user_prompt = f"Verify and format the following analysis results for the task: '{task['description']}'.\n\nResults: {analysis_results}\n\nYour response must strictly follow this format: {task['response']}\n\nDo not state anything other than the result."
        while True:
            try:
                log('DEBUG', f"Sending verification request to Ollama for task: {task['name']}")
                response = ollama.chat(model=self.model, messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ])
                result = response['message']['content'].strip()

                # Optional regex validation
                if 'regex' in task:
                    log('DEBUG', f"Performing regex validation for task: {task['name']}")
                    regex = task['regex']
                    matches = re.findall(regex, result)
                    joined_matches = [' '.join(match) for match in matches]
                    log('INFO', f"Verification completed for task: {task['name']} with {len(joined_matches)} matches: {joined_matches}\n {result}")
                    return joined_matches  # Return list of joined matches (empty if no match)
                else:
                    log('INFO', f"Verification completed for task: {task['name']}")
                    return [result]  # Always return list for consistency
            except Exception as e:
                log('ERROR', f"Error in VerificationAgent: {str(e)}")
                print(f"\n{Fore.RED}Error: Unable to get response from VerificationAgent - {str(e)}")
                return None


class DeduplicationAgent:
    def __init__(self, model):
        self.model = model
        log('INFO', f"DeduplicationAgent initialized with model: {model}")

    def deduplicate(self, results, previous_results):
        log('INFO', "Starting deduplication process")
        unique_results = []
        for result in results:
            if self._deduplicate_single(result, previous_results):
                unique_results.append(result)
        log('INFO', f"Deduplication completed. {len(unique_results)} unique results found")
        return unique_results  # Only return unique results

    def _deduplicate_single(self, result, previous_results):
        log('DEBUG', f"Checking for duplication: {result[:50]}...")

        # Check if the result is already in previous_results
        if result in previous_results:
            log('DEBUG', "Duplicate found in previous results")
            return False

        # Use Levenshtein distance to check for similar results
        similarity_threshold = 8  # Define an acceptable threshold for similarity
        for previous_result in previous_results:
            distance = levenshtein_distance(result, previous_result)
            log('DEBUG', f"Levenshtein distance between '{result[:50]}...' and '{previous_result[:50]}...': {distance}")
            if distance <= similarity_threshold:
                log('DEBUG', "Duplicate found based on Levenshtein distance")
                return False

        log('DEBUG', "Result is unique")
        return True


class ConsolidationAgent:
    def consolidate(self, target_name, results, output_path):
        log('INFO', f"Starting consolidation for target: {target_name}")
        # Load existing results if the file already exists
        if os.path.exists(output_path):
            log('DEBUG', f"Loading existing results from {output_path}")
            with open(output_path, 'r', encoding='utf-8') as json_file:
                existing_results = json.load(json_file)
        else:
            log('DEBUG', f"No existing results found. Creating new results structure")
            existing_results = {}

        # Update the results for the target
        if target_name not in existing_results:
            existing_results[target_name] = {}
        for task_name, task_results in results.items():
            log('DEBUG', f"Consolidating results for task: {task_name}")
            if task_name not in existing_results[target_name]:
                existing_results[target_name][task_name] = []
            existing_results[target_name][task_name].extend(task_results)

        # Save updated results
        log('INFO', f"Saving consolidated results to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as json_file:
            json.dump(existing_results, json_file, indent=4, ensure_ascii=False)
        log('INFO', "Consolidation completed successfully")
        print(f"\n{Fore.GREEN}Results consolidated and saved to {output_path}")

def enable_logging():
    global LOGGING_ENABLED
    LOGGING_ENABLED = True
    log('INFO', "Logging enabled")

def disable_logging():
    global LOGGING_ENABLED
    LOGGING_ENABLED = False
    print("Logging disabled")
