[![Guanaco Banner](https://imgur.com/wJHIeAU.png)](https://www.chai-research.com/competition.html)

[![Pull Requests Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat)](http://makeapullrequest.com)
[![first-timers-only Friendly](https://img.shields.io/badge/first--timers--only-friendly-blue.svg)](http://www.firsttimersonly.com/)

[Chai Guanaco](https://www.chai-research.com/competition.html) is part of the Chai Guanaco Competition, accelerating community AGI.

It's the world's first open community challenge with real-user evaluations. You can submit any GPT-J based 6B models, it will be directly deployed on the [Chai App](https://apps.apple.com/us/app/chai-chat-with-ai-bots/id1544750895) where our over 500K daily active users will be providing live feedback. Get to top of the leaderboard and share the $1 million cash prize!


## The Guanaco Guide

🥇 **Evaluation & Prizes:** Depending on the phase of the competition, a suite of user-level evaluation metrics will be used (i.e. thumbs up / thumbs down rate). Your model will be ranked in real-time compared with other models, you can view the leaderboard at anytime with the pip package

🕵️ **Real-time user feedback:** After your model is deployed, it will go through an safety + integrity checker, once passed, it will be deployed directly to our users who will provide written feedback that you can view via the pip package.

🤖 **Model requirements:** Currently, we support *any* models based off [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b) (i.e. 6B parameters with GPT2 tokenizer). All you need to do is push your model directly to huggingface. Support for more model types is coming soon!

⚙️ **Sampling parameters:** During submission, we allow for custom model generation parameters such as temperature. Once your model is deployed on our platform, it will be using the parameters you've provided to generate chat completions.

📚 **Rules:** By default, 1 developer key per person, each key can deploy 1 model to users at a time. Message in discord if you would like the limit bumped up 😀


## How Does It Work?

-   The `chai_guanaco` pip package provides a way to easily submit your language model, all you need to do is ensure it is on HuggingFace 🤗
-   We will automatically **Tritonize** your model for fast inference and host it in our internal GPU cluster 🚀
-   Once deployed, Chai users on our platform who enter the **arena mode** will be rating your model directly, providing you with both quantatitive and verbal feedback 📈
-   Both the public leaderboard and **user feedback** for your model can be directly downloaded via the `chai_guanaco` package 🧠
-   Cash prizes will be allocated according to your position in the leaderboard 💰

[![Chai Pipeline](https://imgur.com/LtMWOAq.png)](https://www.chai-research.com/competition.html)

## 🚀 Getting Started

**Getting Developer Key**

Join the [competition discord](https://discord.gg/7mXdjAkw2s), introduce yourself and ask for a developer key. Login-based authentication is coming next 🤗


**Submitting A Model**

Use [pip](https://github.com/pypa/pip) to install the Chai Guanaco package

```sh
pip install chai-guanaco
```

Upload any GPT-J 6B based language model *with a tokenizer* to huggingface, i.e. [EleutherAI/gpt-j-6b](https://huggingface.co/EleutherAI/gpt-j-6b). Read [this guide](https://huggingface.co/docs/transformers/model_sharing) if you are unsure. Click the *Use in Transformers* button in huggingface to get the your huggingface model ID (i.e. "EleutherAI/gpt-j-6b")

To submit model simply run:

```python
from chai_guanaco.submit import submit_model

model_url = "EleutherAI/gpt-j-6b" # Your model URL
developer_key = "CR_XXXX" # Your developer key
generation_params = {'temperature': 0.75, 'repetition_penalty': 1.13, 'top_p': 0, "top_k": 0}
submission_parameters = {'model_repo': model_url, 'generation_params': generation_params}
response = submit_model(submission_parameters, developer_key)
submission_id = response['submission_id']
print(submission_id)
````
which outputs your submission id, unique to your model submission.

To verify the status of a submission, you can use the following command:

```python
from chai_guanaco.submit import get_model_info

model_info = get_model_info(submission_id, developer_key)
print(model_info)
```
Once the `status` field shows `success` it means your model has been successfully submitted. A submission typically takes around 10 minutes for the tritonisation process to complete.

**Getting User Feedback**

Once your model has been submitted, it is automatically deployed to the Chai Platform where real-life users will evaluate your model performance. To view their feedback, run:

```python
from chai_guanaco.feedback import get_feedback

model_feedback = get_feedback(submission_id, developer_key)
print(model_feedback.df)
```

Here, you will find a pandas dataframe with all the user feedback for your model, including the conversation each user has had with your model.

You can print samples from your model's user feedbacks by running

```python
model_feedback.sample()
```

Which will print out a user's conversation, together with meta information associated with the conversation (i.e. rating and user feedback).

(Advanced): You can also access the raw feedback data by running

```python
raw_data = model_feedback.raw_data
```


**Getting Live Leaderboard**

To see how your model performs against other models, run:
```python
from chai_guanaco.feedback import display_leaderboard

display_leaderboard(developer_key)
```
which prints out the current leaderboard, with your models positions highlighted

**Re-Submitting Models**

Because it is a competition, you are allowed to test a single model at any given time. However, you can deactivate a model and submit a new one. To do this, simply run:

```python
from chai_guanaco.submit import deactivate_model

deactivate_model(submission_id, developer_key)
```
Which will deactive your model, don't worry, all the model feedback will still be saved, it just means the model will no longer be exposed to users. You can then re-submit by repeating the model submission step.

**Retrieve Your Model Submission IDs**

In case you have forgotten your submission ids / want to view all past submissions, run:

```python
from chai_guanaco.submit import get_my_submissions

submission_ids = get_my_submissions(developer_key)
print(submission_ids)
```
Here you will see all your model submission_ids along with their status, which is either `failed`, `inactive` or `deployed`.


## Resources
|                                                                        |                                                                                                 |
| ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------|
| 📒 [Fine tuning guide](https://huggingface.co/docs/transformers/training) | Guide on language model finetuning                                                           |
| 💾 [Datasets](https://dataset-ideas.tiiny.site/) | Curated list of open-sourced datasets to get started with finetuning                                                  |
| 💖 [Guanaco Discord](https://haystack.deepset.ai/tutorials)                   | Our Guanaco competition discord                                                          |
|🚀 [Deepspeed Guide](https://huggingface.co/docs/transformers/main_classes/deepspeed)     | Guide for training with Deepspeed (faster training without GPU bottleneck)    |
|💬 [Example Conversations](https://huggingface.co/docs/transformers/main_classes/deepspeed)     | Here you can find 1000 example conversations from the Chai Platform     |
| ⚒️ [Build with us](https://boards.greenhouse.io/nexus/jobs/5319721003)| If you think what we are building is cool, join us!|


## 🦙 Hosted & Sponsored By

<a href="https://www.chai-research.com/"><img src="https://imgur.com/u3rOQDJ.png" alt="Chai Logo" height="90"/></a>

<a href="https://www.coreweave.com/"><img src="https://imgur.com/oJyuH8q.png" alt="Coreweave Logo" height="70"/></a>
