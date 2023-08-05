defmodule Example.Evaluate do
  @moduledoc false

  NimbleCSV.define(OrdersParser, separator: ",", escape: "\"")

  @model "roberta-base"

  def evaluate() do
    Nx.default_backend(EXLA.Backend)

    {:ok, spec} =
      Bumblebee.load_spec({:hf, @model},
        architecture: :for_sequence_classification
      )

    spec = Bumblebee.configure(spec, num_labels: 8)

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})
    {:ok, model_info} = Bumblebee.load_model({:hf, @model}, spec: spec)
    %{model: model, params: _} = model_info

    logits_model = Axon.nx(model, & &1.logits)

    params = File.read!("cancel.axon") |> Nx.deserialize()
    test_data = Example.Data.get_data(tokenizer, "priv/data/test.csv")
    accuracy = &Axon.Metrics.accuracy(&1, &2, from_logits: true, sparse: true)

    logits_model
    |> Axon.Loop.evaluator()
    |> Axon.Loop.metric(accuracy, "accuracy")
    |> Axon.Loop.run(test_data, params, compiler: EXLA)
  end

  def cancellations() do
    Nx.default_backend(EXLA.Backend)

    "priv/data/july.csv"
    |> File.stream!()
    |> OrdersParser.parse_stream()
    |> Enum.map(fn [
                     id,
                     total,
                     text,
                     delta,
                     close_reason,
                     previous,
                     month,
                     inserted_at,
                     rfq_state,
                     state
                   ] ->
      %{result: [reason]} = Nx.Serving.batched_run(ExampleClassification, text)

      %{
        id: id,
        total: total,
        reason: reason,
        delta: delta,
        close_reason: close_reason,
        previous: previous,
        month: month,
        inserted_at: inserted_at,
        rfq_state: rfq_state,
        state: state,
        original: text
      }
    end)
    |> writecsv()
  end

  @doc false
  def writecsv(data) do
    file = File.open!("results.csv", [:write, :utf8])

    data
    |> List.flatten()
    |> Enum.map(
      &[
        &1.reason,
        &1.id,
        &1.total,
        &1.delta,
        &1.close_reason,
        &1.previous,
        &1.month,
        &1.inserted_at,
        &1.rfq_state,
        &1.state,
        &1.original
      ]
    )
    |> CSV.encode()
    |> Enum.each(&IO.write(file, &1))
  end
end
