import ollama
import json
import os
import re
import logging
from colorama import Fore
from Levenshtein import distance as levenshtein_distance
from html_reporter import HTMLReporter
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

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

# Helper Functions for Text Normalization and Deduplication
def normalize_text(text):
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # Lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove stop words and apply stemming
    words = text.split()
    filtered_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def is_fuzzy_duplicate(str1, str2, threshold=90):
    # Calculate a ratio that allows for partial matching and reordering
    similarity = fuzz.token_sort_ratio(str1, str2)
    return similarity >= threshold

def are_domain_duplicates(str1, str2):
    version_pattern = r'(\d+\.\d+(\.\d+)?)'
    product1 = re.sub(version_pattern, '', str1).strip()
    product2 = re.sub(version_pattern, '', str2).strip()
    
    version1 = re.findall(version_pattern, str1)
    version2 = re.findall(version_pattern, str2)
    
    # Compare product names and versions separately
    return (product1 == product2) and (version1 == version2)

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
            is_duplicate = False
            for previous_result in previous_results:
                # Step 1: Normalize and check if they are identical
                norm_result = normalize_text(result)
                norm_previous = normalize_text(previous_result)
                if norm_result == norm_previous:
                    is_duplicate = True
                    break

                # Step 2: Domain-specific rules
                if are_domain_duplicates(result, previous_result):
                    is_duplicate = True
                    break

                # Step 3: Fuzzy matching if needed
                if is_fuzzy_duplicate(result, previous_result):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_results.append(result)

        log('INFO', f"Deduplication completed. {len(unique_results)} unique results found")
        return unique_results


class ConsolidationAgent:
    def __init__(self):
        self.html_reporter = HTMLReporter()

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
        
        # Generate HTML report
        html_output_path = os.path.splitext(output_path)[0] + '.html'
        self.html_reporter.generate_report(existing_results, html_output_path)
        
        log('INFO', "Consolidation and HTML report generation completed successfully")
        print(f"\n{Fore.GREEN}Results consolidated and saved to {output_path}")
        print(f"{Fore.GREEN}HTML report generated: {html_output_path}")

def enable_logging():
    global LOGGING_ENABLED
    LOGGING_ENABLED = True
    log('INFO', "Logging enabled")

def disable_logging():
    global LOGGING_ENABLED
    LOGGING_ENABLED = False
    print("Logging disabled")
