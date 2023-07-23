defmodule Example.Tune do
  NimbleCSV.define(DataParser, separator: ",", escape: "\"")

  @model "roberta-base"

  def train() do
    {:ok, spec} =
      Bumblebee.load_spec({:hf, @model},
        architecture: :for_sequence_classification
      )

    spec = Bumblebee.configure(spec, num_labels: 8)

    {:ok, model} = Bumblebee.load_model({:hf, @model}, spec: spec)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    train_data = get_training_data(tokenizer)

    %{model: model, params: params} = model

    [{input, _}] = Enum.take(train_data, 1)
    Axon.get_output_shape(model, input)

    logits_model = Axon.nx(model, & &1.logits)

    loss =
      &Axon.Losses.categorical_cross_entropy(&1, &2,
        reduction: :mean,
        from_logits: true,
        sparse: true
      )

    optimizer = Axon.Optimizers.adam(2.0e-5)
    accuracy = &Axon.Metrics.accuracy(&1, &2, from_logits: true, sparse: true)

    trained_model_state =
      logits_model
      |> Axon.Loop.trainer(loss, optimizer, log: 1)
      |> Axon.Loop.metric(accuracy, "accuracy")
      |> Axon.Loop.run(train_data, params, epochs: 7, compiler: EXLA, strict?: false)

    Nx.serialize(trained_model_state)
    |> then(&File.write!("cancel.axon", &1))
  end

  @doc false
  def get_training_data(tokenizer) do
    batch_size = 32
    sequence_length = 64

    "priv/data/training.csv"
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
