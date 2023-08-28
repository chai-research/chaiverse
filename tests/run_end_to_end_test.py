"""
End to end test:
    1. fit a llama 7b model
    2. submit it to chai guanaco
"""
import chaiverse as cv
import chai_guanaco as cg


def run_llama_7b_model_fitter_end_to_end():
    model = cv.LLaMA7b()
    dataset = cv.load_dataset('ChaiML/davinci_150_examples', 'chatml')

    model.fit(dataset, output_dir='dummy_run', num_epochs=1)
    model_url = 'ChaiML/llama7b_dummy'
    model.push_to_hub(model_url, private=True)
    submit_model(model_url)


def submit_model(model_url):
    generation_params = {
        'temperature': 1.0,
        'repetition_penalty': 1.13,
        'top_p': 0.2,
        "top_k": 40,
        "stopping_words": ['\n'],
        "presence_penalty": 0.,
        "frequency_penalty": 0.
        }
    submission_parameters = {'model_repo': model_url, 'generation_params': generation_params, 'model_name': 'axolotl-llama2-7b'}
    submitter = cg.ModelSubmitter()
    submitter.submit(submission_parameters)


if __name__ == '__main__':
    run_llama_7b_model_fitter_end_to_end()
