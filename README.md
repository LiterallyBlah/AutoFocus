# AutoFocus

**AutoFocus** is a companion tool designed to work alongside [AutoRecon](https://github.com/Tib3rius/AutoRecon), providing enhanced analysis of reconnaissance data through AI-powered processing. AutoFocus is built to streamline the initial engagement phase of penetration testing by focusing on detecting version numbers, identifying points of interest, and highlighting vulnerabilities with minimal manual effort.

## Features

- **Automatic File Enumeration**: AutoFocus recursively enumerates all files generated by AutoRecon, which are organised into target directories by IP address.

- **AI-Powered Chunked Data Processing**: AutoFocus uses a local Large Language Model (LLM) to analyse files in manageable chunks. This allows for efficient handling of large datasets, especially those generated during extensive reconnaissance efforts.

- **Task-Based Analysis**: The tool utilises configurable tasks, specified via YAML, to define the specific data points that need analysis, such as version numbers, points of interest, and vulnerabilities. Each task is designed to target specific objectives in the data.

- **Structured JSON Output**: All findings are saved in a structured JSON format, with results organised by IP address and task type. This structured output allows for easy integration into other tools or streamlined reporting.

- **Progress Updates and Error Handling**: Provides continuous progress updates during the analysis process and includes error-handling mechanisms to ensure reliable operation during data processing.

- **Extensible and Customisable**: AutoFocus is flexible and can easily be adapted to meet the needs of different reconnaissance scenarios. Users can add or modify tasks by editing the `tasks.yaml` configuration files.

- **Advanced Deduplication**: Implements sophisticated deduplication techniques, including text normalisation, domain-specific rules, and fuzzy matching to ensure unique and relevant results.

- **HTML Reporting**: Generates an interactive HTML report alongside the JSON output, providing a user-friendly interface for reviewing analysis results.

- **Logging System**: Incorporates a comprehensive logging system for better debugging and tracking of the analysis process.

## How It Works

AutoFocus employs a multi-agent system to process and analyse reconnaissance data:

1. **File Enumeration**: The script recursively walks through the directory structure created by AutoRecon, identifying files for analysis.

2. **Data Chunking**: Each file is processed in chunks (default 1000 characters with 500 character overlap) to ensure efficient handling of large files.

3. **Initial Analysis**: The `InitialAnalysisAgent` processes each chunk of data for each defined task. It uses the Ollama API to interact with a local LLM (default: Qwen 2.5) for initial analysis.

4. **Verification**: The `VerificationAgent` takes the initial results and verifies them, ensuring they adhere to the specified format for each task. It can also apply regex validation if defined in the task.

5. **Deduplication**: The `DeduplicationAgent` employs advanced techniques to check for duplicate or highly similar results, ensuring that only unique findings are reported. This includes text normalisation, domain-specific rules, and fuzzy matching.

6. **Consolidation**: The `ConsolidationAgent` combines the results for each target and task, updating the JSON output file after processing each file. It also generates an HTML report for easy result visualisation.

7. **Results Output**: The final results are saved in both structured JSON format and an interactive HTML report, organised by target IP and task type.

## Getting Started

### Prerequisites

- Python 3.8 or later
- A local large language model (e.g., Qwen 2.5) to enable offline processing using Ollama
- AutoRecon for initial data collection

### Installation

Clone the repository:
```bash
git clone https://github.com/LiterallyBlah/AutoFocus.git
cd AutoFocus
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

### Usage

1. Ensure AutoRecon has generated the necessary directories for each target IP.

2. Run AutoFocus:
   ```bash
   python autofocus.py --input /path/to/autorecon/output --output /path/to/output/directory --tasks /path/to/tasks.yml
   ```

3. Configure analysis tasks by editing the `tasks.yaml` file, where you can define the tasks you want AutoFocus to perform.

### Example Command

```bash
python autofocus.py --input /path/to/recon/data --output /path/to/output --tasks /path/to/tasks.yml
```

This command processes the reconnaissance data from the specified input directory and saves the results in the output directory, performing the tasks specified in the tasks file.

## Task Configuration

Tasks are specified in a YAML file, and each task requires careful configuration and prompt engineering to achieve optimal results:

- **name**: A unique identifier for the task.
- **description**: A brief description of what the task checks for. This should be clear and specific to guide the LLM's focus.
- **response**: Instructions for the LLM on what to return (e.g., version numbers, points of interest, vulnerabilities). These instructions need to be fine-tuned to elicit precise and relevant responses.
- **output**: The structure of the result in the final JSON output.
- **regex** (optional): A regular expression pattern to validate and extract structured data from the LLM's response.

Fine-tuning and prompt engineering are crucial for each task:

1. **Precision**: Craft prompts that encourage the LLM to provide specific, targeted information.
2. **Consistency**: Ensure prompts maintain a consistent format across tasks for easier processing.
3. **Context**: Provide enough context in the description to guide the LLM's understanding of the task's purpose.
4. **Iterative Improvement**: Regularly review and refine prompts based on the quality of results obtained.
5. **Avoid Ambiguity**: Use clear, unambiguous language to prevent misinterpretation by the LLM.

Example (`tasks.yaml`):
```yaml
tasks:
  - name: "version_check"
    description: "Identify software version numbers for correlation with known issues. Do not provide any information of the tools used to find the vulnerabilities or the checks performed."
    response: "Return the software name and version numbers in the format: software_name:version_number."
    output: "version_numbers"
    regex: '([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*):([A-Za-z0-9!@#$%^&*()_+.,-]+)'
  - name: "vulnerability_scan"
    description: "Extract and list all vulnerabilities found from the output of tools like nmap, wpscan, etc."
    response: "Only return vulnerabilities found in the data. Do not provide any information of the tools used to find the vulnerabilities or the checks performed."
    output: "vulnerabilities"
```

## Contribution

Contributions are welcome! If you have ideas for improvements, new features, or bug fixes, feel free to open a pull request or raise an issue.

## Licence

This project is licensed under the MIT Licence.
