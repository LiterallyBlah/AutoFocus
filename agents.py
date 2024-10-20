import ollama
from ollama import Client
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

# Global Ollama client
ollama_client = Client(timeout=5)

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
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
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

class BaseAgent:
    def __init__(self, model):
        self.model = model
        log('INFO', f"{self.__class__.__name__} initialized with model: {model}")

    def _send_request(self, system_prompt, user_prompt):
        try:
            log('DEBUG', f"Sending request to Ollama with system prompt: {system_prompt}")
            response = ollama_client.chat(
                model=self.model,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': user_prompt},
                ]
            )
            return response['message']['content'].strip()
        except Exception as e:
            log('ERROR', f"Error in {self.__class__.__name__}: {str(e)}")
            print(f"\n{Fore.RED}Error: Unable to get response from {self.__class__.__name__} - {str(e)}")
            return None

class InitialAnalysisAgent(BaseAgent):
    def analyse(self, text, task):
        log('INFO', f"Starting analysis for task: {task['name']}")
        system_prompt = f"You are a penetration testing AI assistant specialised in analysing data for specific tasks. You are required to conduct the following task: {task['description']}. Ensure your response strictly adheres to the following format: {task['response']}"
        user_prompt = f"Data: {text}\n\nResults should be returned as a list, each result on a new line prefixed with '- '. If there is no relevant data found, respond with 'no matching results'. Be concise with your response."
        
        result = self._send_request(system_prompt, user_prompt)
        if result is None:
            return None

        # Process the result to extract list items
        result_list = [line[2:].strip() for line in result.split('\n') if line.startswith('- ')]
        if result_list:
            if any('no matching results' in item.lower() for item in result_list):
                log('INFO', f"Analysis completed for task: {task['name']} with no relevant results")
                return []
            else:
                if 'blacklist' in task:
                    blacklist_words = task['blacklist']
                    result_list = [item for item in result_list if not any(word.lower() in item.lower() for word in blacklist_words)]

                if 'regex' in task:
                    log('DEBUG', f"Performing regex validation for task: {task['name']}")
                    regex = task['regex']
                    matches = [match for item in result_list for match in re.findall(regex, item)]
                    joined_matches = [':'.join(match) for match in matches]
                    log('INFO', f"Analysis completed for task: {task['name']} with {len(joined_matches)} matches: {joined_matches}")
                    return joined_matches
                else:
                    log('INFO', f"Analysis completed for task: {task['name']} with {len(result_list)} results: {result_list}")
                    return result_list
        else:
            log('INFO', f"Analysis completed for task: {task['name']} with no results")
            return []

class DeduplicationAgent:
    def __init__(self):
        log('INFO', "DeduplicationAgent initialized")

    def deduplicate(self, results, previous_results):
        log('INFO', f"Starting deduplication process with {len(results)} new results and {len(previous_results)} previous results")
        unique_results = []
        prev_results_updated = previous_results
        
        # Loop through new results
        for index, result in enumerate(results):
            log('DEBUG', f"Processing result {index + 1}/{len(results)}: {result}")
            if not isinstance(result, str):
                log('WARNING', f"Skipping non-string result at index {index}: {result}")
                continue
            
            # Normalize the current result
            norm_result = normalize_text(result)
            log('DEBUG', f"Normalized result: {norm_result}")
            is_duplicate = False
            
            # Loop through previous results to check for duplicates
            for prev_index, previous_result in enumerate(prev_results_updated):
                log('DEBUG', f"Comparing with previous result {prev_index + 1}/{len(prev_results_updated)}: {previous_result}")
                if not isinstance(previous_result, str):
                    log('WARNING', f"Skipping non-string previous result at index {prev_index}: {previous_result}")
                    continue
                
                # Normalize the previous result
                norm_previous = normalize_text(previous_result)
                log('DEBUG', f"Normalized previous result: {norm_previous}")

                # Step 1: Check for exact match
                if norm_result == norm_previous:
                    log('INFO', f"Exact match found for result: {result}")
                    is_duplicate = True
                    break

                # Step 2: Check for domain-specific duplicate (version numbers, etc.)
                if are_domain_duplicates(result, previous_result):
                    log('INFO', f"Domain-specific duplicate found for result: {result}")
                    is_duplicate = True
                    break

                # Step 3: Fuzzy matching
                if is_fuzzy_duplicate(result, previous_result):
                    log('INFO', f"Fuzzy duplicate found for result: {result}")
                    is_duplicate = True
                    break
            
            # If no duplicates were found, add to unique results
            if not is_duplicate:
                log('INFO', f"Unique result found: {result}")
                unique_results.append(result)
                prev_results_updated.append(result)
            else:
                log('DEBUG', f"Duplicate result skipped: {result}")

        log('INFO', f"Deduplication completed. {len(unique_results)} unique results found out of {len(results)} total results")
        return unique_results

class ConsolidationAgent:
    def __init__(self):
        self.html_reporter = HTMLReporter()
        self.deduplication_agent = DeduplicationAgent()

    def consolidate(self, target_name, results, output_path):
        log('INFO', f"Starting consolidation for target: {target_name} with results: {results}")
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
            log('DEBUG', f"Creating new target: {target_name}")
            existing_results[target_name] = {}
        
        for task_name, task_results in results.items():
            log('DEBUG', f"Consolidating results for task: {task_name}")
            if task_name not in existing_results[target_name]:
                existing_results[target_name][task_name] = []
            
            # Deduplicate results before adding them
            previous_results = [item['result'] for item in existing_results[target_name][task_name]]
            new_results = [item['result'] for item in task_results]
            deduplicated_results = self.deduplication_agent.deduplicate(new_results, previous_results)
            log('INFO', f"Deduplicated {len(deduplicated_results)} results for task: {task_name}, and the results are: {deduplicated_results}")
            
            # Add only unique results to existing results, including extra information
            for result in task_results:
                if result['result'] in deduplicated_results and result['result'] not in [item['result'] for item in existing_results[target_name][task_name]]:
                    existing_results[target_name][task_name].append(result)

        # Save updated results
        log('INFO', f"Saving consolidated results to {output_path}")
        log('DEBUG', f"Updated results: {existing_results}")
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
