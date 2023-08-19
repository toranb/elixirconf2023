defmodule Example do
  @moduledoc false

  def tune() do
    Example.Tune.train()
  end

  def evaluate() do
    Example.Evaluate.evaluate()
  end

  def cancellations() do
    Example.Evaluate.cancellations()
  end

  def get_token_ids() do
    # [101, 1996, 3976, 2001, 25506, 102]
    text = "the price was outrageous"
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, text)
    Tokenizers.Encoding.get_tokens(encoding) |> IO.inspect(label: "tokens")
    Tokenizers.Encoding.get_ids(encoding) |> IO.inspect(label: "token ids")
  end

  def embeddings(token_ids) when is_list(token_ids) do
    {:ok, %{model: model, params: params}} =
      Bumblebee.load_model({:hf, "bert-base-cased"},
        architecture: :base
      )

    inputs = %{
      "input_ids" => Nx.tensor([token_ids])
    }

    outputs = Axon.predict(model, params, inputs)

    outputs.hidden_state
  end
end
