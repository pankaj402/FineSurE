from openai import OpenAI
import json
import sys
import os
from utils import get_response, get_keyfact_alighment_prompt, parsing_llm_keyfact_alighment_output
from utils import compute_completeness_percentage_score, compute_conciseness_percentage_score

_api_key = "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"
_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=_api_key)
_model = "meta-llama/llama-3-8b-instruct:free"

def main(input_path, keyfact_path, output_path, print_interval=2):
    inputs = []
    for line in open(input_path, 'r'):
        line = json.loads(line)
        inputs.append(line)

    keyfacts = {}
    for line in open(keyfact_path, 'r'):
        line = json.loads(line)
        keyfacts[line['doc_id']] = line['key_facts']

    cnt_total_inference = 0
    cnt_success_inference = 0
    model_labels = {}
    
    raw_data_writer = open(os.path.join(output_path, 'raw-data.json'), 'w')
    result_writer = open(os.path.join(output_path, 'result.json'), 'w')

    for input_id, input_json in enumerate(inputs):
        doc_id = input_json['doc_id']
        model_name = input_json.get('model', 'human')
        src = input_json['transcript']
        sentences = input_json['sentences']
        list_keyfacts = keyfacts.get(doc_id, [])

        if not list_keyfacts:
            print(f"No key facts for {doc_id}, skipping")
            continue

        prompt = get_keyfact_alighment_prompt(keyfacts=list_keyfacts, sentences=sentences)
        output = get_response(client=_client, prompt=prompt, model=_model)
 
        input_json['llm_output'] = output
        input_json['pred_alignment_labels'], input_json['pred_sentence_line_numbers'] = parsing_llm_keyfact_alighment_output(output)

        success_flag = len(input_json['pred_alignment_labels']) > 0

        print("\nInput ID:", doc_id, "Model Name:", model_name)
        print("Success:", success_flag)
        print('\t[Alignment Label]:', input_json['pred_alignment_labels'])
        print('\t[Matched Sentence Line Numbers]:', input_json['pred_sentence_line_numbers'])

        cnt_total_inference += 1
        if success_flag:
            cnt_success_inference += 1
        else:
            continue      

        completeness_score = compute_completeness_percentage_score(input_json['pred_alignment_labels'])
        conciseness_score = compute_conciseness_percentage_score(input_json['pred_sentence_line_numbers'], len(sentences))

        if model_name not in model_labels:
            model_labels[model_name] = {'completeness_scores': [], 'conciseness_scores': []}
        model_labels[model_name]['completeness_scores'].append(completeness_score)
        model_labels[model_name]['conciseness_scores'].append(conciseness_score)

        print('\t[Completeness Score]:', '{:.1%}'.format(completeness_score))
        print('\t[Conciseness Score]:', '{:.1%}'.format(conciseness_score))

        def print_results(model_labels):
            summary_level_completeness_scores = {}
            summary_level_conciseness_scores = {}

            for model_name, error_labels in model_labels.items():
                summary_level_completeness_scores[model_name] = sum(error_labels['completeness_scores']) / len(error_labels['completeness_scores'])
                summary_level_conciseness_scores[model_name] = sum(error_labels['conciseness_scores']) / len(error_labels['conciseness_scores'])

            text_output = "\n\n\n[Evaluation Results]\n"
            text_output += '\n* completeness score (higher is better)\n'
            for model_name, score in summary_level_completeness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            text_output += '\n* conciseness score (higher is better)\n'
            for model_name, score in summary_level_conciseness_scores.items():
                text_output += model_name + '\t' + str('{:.1%}'.format(score)) + '\n'

            success_ratio = '{:.1%}'.format(cnt_success_inference/float(cnt_total_inference))
            text_output += '\n* success rate: ' + str(success_ratio) + '\n\n\n'

            print(text_output)
            return text_output

        if cnt_total_inference % print_interval == 0:
            print_results(model_labels)
           
        json.dump(input_json, raw_data_writer)
        raw_data_writer.write('\n')
        raw_data_writer.flush()
    raw_data_writer.close()

    text_output = print_results(model_labels)
    result_writer.write(text_output)
    result_writer.close()

    return model_labels

if __name__ == "__main__":
    input_path = sys.argv[1]
    keyfact_path = sys.argv[2]
    output_folder = sys.argv[3]
    print_interval = 10
    os.makedirs(output_folder, exist_ok=True)
    model_labels = main(input_path, keyfact_path, output_folder, print_interval)