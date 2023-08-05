defmodule Example.Tune do
  NimbleCSV.define(DataParser, separator: ",", escape: "\"")

  @model "roberta-base"

  def train() do
    Nx.default_backend(EXLA.Backend)

    {:ok, spec} =
      Bumblebee.load_spec({:hf, @model},
        architecture: :for_sequence_classification
      )

    spec = Bumblebee.configure(spec, num_labels: 8)

    {:ok, model} = Bumblebee.load_model({:hf, @model}, spec: spec)
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    train_data = Example.Data.get_data(tokenizer, "priv/data/demo.csv")

    %{model: model, params: params} = model

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
end
