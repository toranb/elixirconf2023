defmodule Example do
  @moduledoc false

  def tune() do
    Example.Tune.train()
  end

  def evaluate() do
    Example.Evaluate.cancellations()
  end

  def encodings() do
    input = "the price was outrageous"
    {:ok, tokenizer} = Tokenizers.Tokenizer.from_pretrained("bert-base-uncased")
    {:ok, encoding} = Tokenizers.Tokenizer.encode(tokenizer, input)
    Tokenizers.Encoding.get_tokens(encoding) |> IO.inspect(label: "vocab token ids")
    Tokenizers.Encoding.get_ids(encoding) |> IO.inspect(label: "encoding")
  end
end
