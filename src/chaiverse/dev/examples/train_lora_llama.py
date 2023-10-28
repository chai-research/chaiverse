from chaiverse.dev.dataset import DatasetLoader, CausalDatasetBuilder
from chaiverse.dev.tokenizer import LlamaTokenizer
from chaiverse.dev.model.lora_model import LoraTrainer


if __name__ == '__main__':
    # load data
    data_path = 'ChaiML/davinci_completions_1.7m_unsampled'
    data_loader = DatasetLoader(
            hf_path=data_path,
            data_samples=100,
            validation_split_size=0.1,
            shuffle=True,
            )
    df = data_loader.load()
    print(df)

    # process data
    tokenizer = LlamaTokenizer()
    data_builder = CausalDatasetBuilder(
            tokenizer_loader=tokenizer,
            block_size=1024,
            )
    data = data_builder.generate(df, n_jobs=10)
    print(data)

    # train model
    base_model = 'NousResearch/Llama-2-7b-hf'
    model = LoraTrainer(
            model_name=base_model,
            output_dir='test_lora',
            )
    print(model.lora_config)
    model.fit(data)
    model.save()
    model.merge()
    model_path = 'ebony59/llama-2-7b-lora-davinci100'
    model.push_to_hub(model_path, private=True)
