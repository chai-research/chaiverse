[![Chaiverse Banner](https://imgur.com/vUn3OXJ.png)](https://www.chai-research.com/competition.html)
[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![first-timers-only Friendly](https://img.shields.io/badge/first--timers--only-friendly-blue.svg)](http://www.firsttimersonly.com/)

[ChaiVerse](https://www.chai-research.com/competition.html) is the community's one-stop repository for large language model training, deployment and LIVE USERS evaluation package. Train models and win prizes!

## A Collaboration Between
[![Collaboration Banner](https://imgur.com/8oJSWan.png)](https://github.com/OpenAccess-AI-Collective/axolotl)


## Quick Start

To train a LLaMA7b model and push it to huggingface it's just 5 lines of code 🥳

```python
import chaiverse as cv

dataset = cv.load_dataset('ChaiML/davinci_150_examples', 'chatml')

model = cv.LLaMA7b()
model.fit(dataset, output_dir='./my_first_llama', num_epochs=1)

model_url = 'ChaiML/llama7b_dummy' # your huggingface URL
model.push_to_hub(model_url, private=True)
```

## Winning Prizes with Chai Guanaco
```sh
chai-guanaco login
```

Now submit the model you have just trained!

```python
import chai_guanaco as chai

model_url = "ChaiML/llama7b_dummy"

generation_params = {
        'temperature': 1.0,
        'repetition_penalty': 1.13,
        'top_p': 0.2,
        "top_k": 40,
        "stopping_words": ['\n'],
        "presence_penalty": 0.,
        "frequency_penalty": 0.
        }
submission_parameters = {'model_repo': model_url, 'generation_params': generation_params, 'model_name': 'my-awesome-llama'}

submitter = chai.ModelSubmitter()
submission_id = submitter.submit(submission_parameters)
```

## Sponsored By
[![Sponsorship Banner](https://imgur.com/yovi11c.png)](https://www.coreweave.com/)


