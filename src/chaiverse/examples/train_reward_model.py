import logging

from chaiverse.dataset import DatasetLoader, RewardDatasetBuilder
from chaiverse.tokenizer import GPT2Tokenizer
from chaiverse.model.reward_model import RewardClassificationTrainer, RewardRegressionTrainer


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # load data
    data_path = 'ChaiML/20231012_chai_prize_reward_model_data'
    data_loader = DatasetLoader(
            hf_path=data_path,
            data_samples=100,
            validation_split_size=0.1,
            shuffle=True,
            )
    df = data_loader.load()
    print(df)

    # process data
    tokenizer_loader = GPT2Tokenizer(
            padding_side='right',
            truncation_side='left',
            )
    data_builder = RewardDatasetBuilder(
            tokenizer_loader=tokenizer_loader,
            block_size=512,
            )
    data = data_builder.generate(df, n_jobs=10)
    print(data)

    # train model
    model = RewardClassificationTrainer(
            model_name='gpt2',
            tokenizer_loader=tokenizer_loader,
            output_dir='test_reward_model',
            # num_labels=2,
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=False,
            logging_strategy='steps',
            logging_steps=2,
            eval_strategy='steps',
            eval_steps=2,
            no_cuda=True,
            )
    model.fit(data)

    # upload model
    # model_path = 'ChaiML/chai_reward_model_gpt2_test'
    # model.push_to_hub(model_path, private=True)
