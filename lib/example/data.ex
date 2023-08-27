defmodule Example.Data do
  NimbleCSV.define(DataParser, separator: ",", escape: "\"")

  def get_data(tokenizer, filename) do
    batch_size = 16
    sequence_length = 65

    filename
    |> File.stream!()
    |> DataParser.parse_stream()
    |> Stream.chunk_every(batch_size)
    |> Stream.map(fn batch ->
      [labels, text] =
        batch
        |> Enum.reduce([[], []], fn [label, txt], acc ->
          l = Enum.at(acc, 0)
          t = Enum.at(acc, 1)
          labels = l ++ [label |> String.to_integer()]
          text = t ++ [txt]
          [labels, text]
        end)

      tokenized = Bumblebee.apply_tokenizer(tokenizer, text, length: sequence_length)
      {tokenized, Nx.stack(labels)}
    end)
  end
end
