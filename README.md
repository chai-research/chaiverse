[![Chaiverse Banner](https://imgur.com/vUn3OXJ.png)](https://www.chai-research.com/competition.html)
[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![first-timers-only Friendly](https://img.shields.io/badge/first--timers--only-friendly-blue.svg)](http://www.firsttimersonly.com/)

[ChaiVerse](https://www.chai-research.com/competition.html) is the community's one-stop repository for large language model training, deployment and LIVE USERS evaluation package. Train models and win prizes!

## A Collaboration Between
[![Collaboration Banner](https://imgur.com/8oJSWan.png)](https://github.com/OpenAccess-AI-Collective/axolotl)

## Quick Start With Run Pod
### Just watch the following 3 minutes tutorial, our repo is made to run on runpod 🚀🚀 
[![Watch this video](https://imgur.com/mBwiQby.png)](https://vimeo.com/858817518?share=copy)

1. login to your runpod.io account then go to https://www.runpod.io/console/gpu-cloud
2. From "GPU Cloud", select "Community Cloud", and choose the required GPU to deploy the RunPod Pytorch 2.0.1 Template.For Lora and Reward model training, 1 x A40 is enough!
        ![image](https://github.com/chai-research/chaiverse/assets/52447514/2bd5ff1b-1934-4188-8736-d229e9be74b4)
   
4. Click "Customize Deployment", and change "Container Image" to "chaiverse/runpod". Set Container Disk and Volume Disk to "100GB". Click "set Overrides" to deploy the changes. Then click "Continue".
        ![image](https://github.com/chai-research/chaiverse/assets/52447514/e9583b73-e915-48b4-8fd9-3bf90a856b88)

        ![image](https://github.com/chai-research/chaiverse/assets/52447514/93593a76-78b7-46d3-ad53-4a8eaa465b4e)

        ![image](https://github.com/chai-research/chaiverse/assets/52447514/036193a4-9c28-4890-a736-612a594710bf)
   
6. Click "Deploy" to get your pod running.
        ![image](https://github.com/chai-research/chaiverse/assets/52447514/c484780d-91e3-4bb9-9c93-bf9003a3a93d)
           
8. Find the pods under "MANAGE: Pods" and your pod is ready there. Click "Connect" to start the journey!
        ![image](https://github.com/chai-research/chaiverse/assets/52447514/d0d4bd4c-c95d-4bf3-9d89-b158956e8ed6)


## Model Training
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
        "presence_penalty": 1.5,
        "frequency_penalty": 1.5
        }
submission_parameters = {'model_repo': model_url, 'generation_params': generation_params, 'model_name': 'my-awesome-llama'}

submitter = chai.ModelSubmitter()
submission_id = submitter.submit(submission_parameters)
```

## Sponsored By
[![Sponsorship Banner](https://imgur.com/yovi11c.png)](https://www.coreweave.com/)


