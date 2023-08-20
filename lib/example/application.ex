defmodule Example.Application do
  use Application

  @moduledoc false

  @model "roberta-base"

  @impl true
  def start(_type, _args) do
    {:ok, spec} =
      Bumblebee.load_spec({:hf, @model},
        architecture: :for_sequence_classification
      )

    spec = Bumblebee.configure(spec, num_labels: 8)

    {:ok, bert} = Bumblebee.load_model({:hf, @model}, spec: spec)
    %{model: bert_model, params: _} = bert

    params = File.read!("cancel.axon") |> Nx.deserialize()

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    classification =
      Example.TextClassification.text_classification(
        %{model: bert_model, params: params, spec: spec},
        tokenizer
      )

    children = [
      {Nx.Serving, name: ExampleClassification, serving: classification}
    ]

    opts = [strategy: :one_for_one, name: Example.Supervisor]
    Supervisor.start_link(children, opts)
  end
end
