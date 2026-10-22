from datasets import load_dataset


def load_python_code_dataset():
    """
    Load the Python code dataset from Hugging Face.

    Returns:
        DatasetDict: A dictionary containing the train, validation, and test datasets.
    """
    ds = load_dataset("jtatman/python-code-dataset-500k")
    return ds


def process_python_code_sample(sample):
    """
    Process a single sample from the Python code dataset.

    Args:
        sample (dict): A dictionary containing the input and output code.

    Returns:
        dict: A dictionary containing the processed input and output code.
    """
    system_prompt = sample['system_prompt']
    instruction = sample['instruction']
    raw_output_code = sample['output']
    python_code = raw_output_code.split('```python')[1].split('```')[0].strip()

    return {
        'input': system_prompt + " " + instruction,
        'output': python_code}

print(ds['train'][1243]['output'])