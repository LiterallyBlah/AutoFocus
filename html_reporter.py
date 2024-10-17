import os
import json
from jinja2 import Environment, FileSystemLoader
from datetime import datetime

class HTMLReporter:
    def __init__(self, template_dir='templates'):
        self.env = Environment(loader=FileSystemLoader(template_dir))
        self.template = self.env.get_template('report_template.html')

    def generate_report(self, consolidated_results, output_path):
        """
        Generate an HTML report from the consolidated results.
        
        :param consolidated_results: Dictionary containing the analysis results
        :param output_path: Path to save the generated HTML report
        """
        report_data = self._prepare_report_data(consolidated_results)
        html_content = self.template.render(report_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"HTML report generated: {output_path}")

    def _prepare_report_data(self, consolidated_results):
        """
        Prepare the data for the HTML template.
        
        :param consolidated_results: Dictionary containing the analysis results
        :return: Dictionary with prepared data for the template
        """
        report_data = {
            'generated_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'targets': []
        }

        for target_name, target_results in consolidated_results.items():
            target_data = {
                'name': target_name,
                'tasks': []
            }

            for task_name, task_results in target_results.items():
                task_data = {
                    'name': task_name,
                    'results': task_results
                }
                target_data['tasks'].append(task_data)

            report_data['targets'].append(target_data)

        return report_data

# Example usage (can be removed in production)
if __name__ == "__main__":
    # Load sample data (replace with actual data in production)
    with open('sample_results.json', 'r') as f:
        sample_results = json.load(f)

    reporter = HTMLReporter()
    reporter.generate_report(sample_results, 'output/analysis_report.html')
