from openai import OpenAI
import json
import sys
import os
from utils import get_response, parsing_llm_fact_checking_output, compute_faithfulness_percentage_score
from utils import get_fact_checking_prompt, ERROR_TYPES

_api_key = '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'
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_api_key)
_model = "meta-llama/llama-3-8b-instruct:free"  # Changed to proven model

def main(input_path, output_path, print_interval=2):
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    error_type_counts = {}
    
    raw_data_writer = open(os.path.join(output_path, 'raw-data.json'), 'w')
    result_writer = open(os.path.join(output_path, 'result.json'), 'w')

    for input_id, input_json in enumerate(inputs):
        doc_id = input_json['doc_id']
        model_name = input_json.get('model', 'human')
        src = input_json['transcript']
        sentences = input_json['sentences']

        prompt = get_fact_checking_prompt(input=src, sentences=sentences)
        output = get_response(client=_client, prompt=prompt, model=_model)

        input_json['llm_output'] = output
        input_json['pred_faithfulness_labels'], input_json['pred_faithfulness_error_type'] = parsing_llm_fact_checking_output(output)

        success_flag = len(input_json['pred_faithfulness_labels']) > 0

        print("\nInput ID:", doc_id, "Model Name:", model_name)
        print("Success:", success_flag)
        print('\t[Error Label]:', input_json['pred_faithfulness_labels'])
        print('\t[Error Type]:', input_json['pred_faithfulness_error_type'])

        cnt_total_inference += 1
        if success_flag:
            cnt_success_inference += 1
        else:
            continue

        faithfulness_score = compute_faithfulness_percentage_score(input_json['pred_faithfulness_labels'])
        print('\t[Faithfulness Score]:', '{:.1%}'.format(faithfulness_score))

        if model_name not in model_labels:
            model_labels[model_name] = {'faithfulness_scores': [], 'binary_labels': []}
            error_type_counts[model_name] = {et: 0 for et in ERROR_TYPES + ['no error']}
        model_labels[model_name]['faithfulness_scores'].append(faithfulness_score)
        model_labels[model_name]['binary_labels'].extend(input_json['pred_faithfulness_labels'])

        for error_type in input_json['pred_faithfulness_error_type']:
            error_type_counts[model_name][error_type] += 1

        def print_results_faithfulness(model_labels, error_type_counts):
            sentence_level_errors = {}
            summary_level_scores = {}
            for model_name, error_labels in model_labels.items():
                sentence_level_errors[model_name] = sum(error_labels['binary_labels']) / len(error_labels['binary_labels'])
                summary_level_scores[model_name] = sum(error_labels['faithfulness_scores']) / len(error_labels['faithfulness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '* sentence-level factuality error ratio (lower is better)\n'
            for model_name, error_rate in sentence_level_errors.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(error_rate)) + '\n'

            text_output += '\n* summary-level faithfulness score (higher is better)\n'
            for model_name, score in summary_level_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* error type breakdown\n'
            for model_name, counts in error_type_counts.items():
                text_output += f"{model_name}:\n"
                for error_type, count in counts.items():
                    text_output += f"  {error_type}: {count}\n"

            success_ratio = '{:.1%}'.format(cnt_success_inference/float(cnt_total_inference))
            text_output += '\n* success rate: ' + str(success_ratio) + '\n\n\n'

            print(text_output)
            return text_output, error_type_counts

        if cnt_total_inference % print_interval == 0:
            text_output, _ = print_results_faithfulness(model_labels, error_type_counts)
        
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
    raw_data_writer.close()

    text_output, error_type_counts = print_results_faithfulness(model_labels, error_type_counts)
    result_writer.write(text_output)
    result_writer.close()

    return model_labels, error_type_counts

if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print_interval = 10
    os.makedirs(output_path, exist_ok=True)
    model_labels, error_type_counts = main(input_path, output_path, print_interval)