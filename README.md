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

   <img src='https://github.com/chai-research/chaiverse/assets/52447514/3a38a674-adc9-447b-948e-8e82cf8644f5' width="700">
   
3. Click "Customize Deployment"
   
   <img src="https://github.com/chai-research/chaiverse/assets/52447514/a69b4b8b-8484-43b0-8a13-3aaeb08439e3" width="700">

   Change "Container Image" to "chaiverse/runpod", and "Container Disk" and "Volume Disk" to "100GB". Then click "set Overrides" to save the changes.

   <img src="https://github.com/chai-research/chaiverse/assets/52447514/eb215217-7800-4a57-971a-7d6710633402" width="700">

   Click "Continue" to deploy the changes.

   <img src="https://github.com/chai-research/chaiverse/assets/52447514/991a92f7-32f5-4d92-a014-bfad5a8617e1" width="700">


4. Click "Deploy" to get your pod running.

   <img src="https://github.com/chai-research/chaiverse/assets/52447514/8423bf17-2b71-4c96-9690-36cb97fb3055" width="700">
           
5. Find the pods under "MANAGE: Pods" and your pod is ready there. Click "Connect" to start the journey!

   <img src="https://github.com/chai-research/chaiverse/assets/52447514/4d09d2b4-1fd8-4dff-bb0d-9bc5d88f4345" width="700">


## Model Training
Currently we support efficient training with Lora, and reward model training.
The sample codes can be found under `src/chaiverse/examples`

### Fit a model with Lora
Nice and easy pipeline with Lora implementation to train a LLaMA7b model and push it to huggingface within hours 🥳

The [example notebook](https://github.com/chai-research/chaiverse/blob/clean_dev_main/src/chaiverse/examples/example_lora_llama.ipynb) to fit a Llama7b model and push it to huggingface.

### Reward Model Training
We also implemented a reward model training pipeline which helps to boost your model's performance even more!

The [example notebook](https://github.com/chai-research/chaiverse/blob/clean_dev_main/src/chaiverse/examples/example_reward_model.ipynb) running through the pipeline.

Detailed writeups about the reward model's [Introduction and Data Generation](https://wild-chatter-b52.notion.site/Chai-Prize-Reward-Model-Part-I-Data-026a10a8998a404ca6a52251c0c8d052) and [Model Training](https://wild-chatter-b52.notion.site/Chai-Prize-Reward-Model-Part-II-Training-model-753b574c843f4d0780bf8d85b084da57). 

## Winning Prizes with Chai Guanaco
Check out [Chai Prize](https://www.chai-research.com/competition.html), the world's first open community challenge with real-user evaluations hosted by Chai!

The model submission is managed by the [chai-guanaco](https://pypi.org/project/chai-guanaco/) package, check out the link for more details including quick start tutorials.

```sh
chai-guanaco login
```

Now submit the model you have just trained!

```python
import chai_guanaco as chai

model_url = "ChaiML/llama7b_dummy"

#add reward_url if you also want to submit a reward model
reward_url = "ChaiML/reward_dummy"

generation_params = {
        'temperature': 1.0,
        'repetition_penalty': 1.13,
        'top_p': 0.2,
        "top_k": 40,
        "stopping_words": ['\n'],
        "presence_penalty": 1.5,
        "frequency_penalty": 1.5
        }
submission_parameters = {
                'model_repo': model_url,
                'reward_repo':reward_url,
                'generation_params': generation_params,
                'model_name': 'my-awesome-model'}

submitter = chai.ModelSubmitter()
submission_id = submitter.submit(submission_parameters)
```

## Sponsored By
[![Sponsorship Banner](https://imgur.com/yovi11c.png)](https://www.coreweave.com/)


