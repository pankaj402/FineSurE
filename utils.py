import ast
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ERROR_TYPES = ['out-of-context error', 'entity error', 'predicate error', 'circumstantial error', 'grammatical error', 'coreference error', 'linking error', 'other error']

def get_response(client, prompt, model, temperature=0.0, retries=3, delay=5):
    """Get response from LLM with retries."""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            if not response.choices or len(response.choices) == 0:
                logger.warning("Empty choices in API response, attempt %d/%d", attempt + 1, retries)
                time.sleep(delay)
                continue
            logger.info("API response received: %s", response)
            return response.choices[0].message.content
        except Exception as e:
            logger.error("API call failed, attempt %d/%d: %s", attempt + 1, retries, str(e))
            time.sleep(delay)
            continue
    logger.error("All retries failed for model %s", model)
    return ""  # Return empty string to avoid NoneType

def get_fact_checking_prompt(input, sentences):
    num_sentences = str(len(sentences))
    sentences = '\n'.join(sentences)
    prompt = """
You will receive a transcript followed by a corresponding summary. Your task is to assess the factuality of each summary sentence across nine categories:
* no error: the statement aligns explicitly with the content of the transcript and is factually consistent with it.
* out-of-context error: the statement contains information not present in the transcript.
* entity error: the primary arguments (or their attributes) of the predicate are wrong.
* predicate error: the predicate in the summary statement is inconsistent with the transcript.
* circumstantial error: the additional information (like location or time) specifying the circumstance around a predicate is wrong.
* grammatical error: the grammar of the sentence is so wrong that it becomes meaningless.
* coreference error: a pronoun or reference with wrong or non-existing antecedent.
* linking error: error in how multiple statements are linked together in the discourse (for example temporal ordering or causal link).
* other error: the statement contains any factuality error which is not defined here.

Instruction:
First, compare each summary sentence with the transcript.
Second, provide a single sentence explaining which factuality error the sentence has.
Third, answer the classified error category for each sentence in the summary.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "sentence", "reason", and "category":
[{"sentence": "first sentence", "reason": "your reason", "category": "no error"}, {"sentence": "second sentence", "reason": "your reason", "category": "out-of-context error"}]

Transcript:
%s

Summary with %s sentences:
%s
""" % (input, num_sentences, sentences)
    return prompt

def parsing_llm_fact_checking_output(output):
    try:
        start_idx = output.find('[')
        if start_idx != -1:
            end_idx = output.find(']')
            output = output[start_idx:end_idx+1]
            output = output.replace('\n','')
            output = ast.literal_eval(output)
            pred_labels, pred_types = [], []
            for out in output:
                category = out["category"].replace('\n', '').replace('[', '').replace(']', '')
                pred_labels.append(0 if category.lower() == "no error" else 1)
                pred_types.append(category)
            return pred_labels, pred_types
        else:
            start_idx = output.find('{')
            end_idx = output.find('}')
            output = output[start_idx:end_idx+1]
            output = ast.literal_eval(output)
            category = output["category"].replace('\n', '').replace('[', '').replace(']', '')
            pred_labels = [0 if category.lower() == "no error" else 1]
            pred_types = [category]
            return pred_labels, pred_types
    except Exception as e:
        try:
            subseqs = output.split("category")
            def error_detection(subseq):
                for error_type in ERROR_TYPES:
                    if error_type in subseq:
                        return 1, error_type
                return 0, "no error"
            pred_labels, pred_types = [], []
            for subseq in subseqs:
                error_label, error_type = error_detection(subseq)
                pred_labels.append(error_label)
                pred_types.append(error_type)
            return pred_labels, pred_types
        except Exception as e:
            print('parsing error:', e)
            return [], []

def get_keyfact_alighment_prompt(keyfacts, sentences):
    summary = ['[' + str(line_num + 1) + '] ' + sentence for line_num, sentence in enumerate(sentences)]
    summary = '\n'.join(summary)
    num_key_facts = str(len(keyfacts))
    key_facts = '\n'.join(keyfacts)
    prompt = '''
You will receive a summary and a set of key facts for the same transcript. Your task is to assess if each key fact is inferred from the summary.

Instruction:
First, compare each key fact with the summary.
Second, check if the key fact is inferred from the summary and then response "Yes" or "No" for each key fact. If "Yes", specify the line number(s) of the summary sentence(s) relevant to each key fact.

Provide your answer in JSON format. The answer should be a list of dictionaries whose keys are "key fact", "response", and "line number":
[{"key fact": "first key fact", "response": "Yes", "line number": [1]}, {"key fact": "second key fact", "response": "No", "line number": []}]

Summary:
%s

%s key facts:
%s
''' % (summary, num_key_facts, key_facts)
    return prompt

def parsing_llm_keyfact_alighment_output(output):
    try:
        output = output.replace('```', '')
        start_idx = output.find('[')
        output = output[start_idx:]
        output = ast.literal_eval(output)
        matched_lines = set()
        pred_labels = []
        for out in output:
            category = out["response"]
            pred_labels.append(1 if category.lower() == "yes" else 0)
            if 'line number' in out:
                line_nums = out["line number"]
                for line_num in line_nums:
                    if isinstance(line_num, str):
                        line_num = line_num.replace('[', '').replace(']', '')
                    matched_lines.add(int(line_num))
        return pred_labels, list(matched_lines)
    except Exception as e:
        print('keyfact parsing error:', e)
        return [], []

def compute_faithfulness_percentage_score(pred_faithfulness_labels):
    return 1.0 - sum(pred_faithfulness_labels) / len(pred_faithfulness_labels)

def compute_completeness_percentage_score(pred_alignment_labels):
    return sum(pred_alignment_labels) / len(pred_alignment_labels)

def compute_conciseness_percentage_score(pred_sentence_line_numbers, num_sentences):
    return len(pred_sentence_line_numbers) / num_sentences