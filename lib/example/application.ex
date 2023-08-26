defmodule Example.Application do
  use Application

  @moduledoc false

  @model "roberta-base"

  @impl true
  def start(_type, _args) do
    children = [
      {Nx.Serving, name: ExampleClassification, serving: classification()},
      {Nx.Serving, name: ExampleEmbeddings, serving: embeddings()}
    ]

    opts = [strategy: :one_for_one, name: Example.Supervisor]
    Supervisor.start_link(children, opts)
  end

  def classification() do
    {:ok, spec} =
      Bumblebee.load_spec({:hf, @model},
        architecture: :for_sequence_classification
      )

    spec = Bumblebee.configure(spec, num_labels: 8)

    {:ok, bert} = Bumblebee.load_model({:hf, @model}, spec: spec)
    %{model: bert_model, params: _} = bert

    params = File.read!("cancel.axon") |> Nx.deserialize()

    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, @model})

    Example.TextClassification.text_classification(
      %{model: bert_model, params: params, spec: spec},
      tokenizer
    )
  end

  def embeddings() do
    model = "intfloat/e5-large"
    {:ok, model_info} = Bumblebee.load_model({:hf, model})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, model})

    Bumblebee.Text.TextEmbedding.text_embedding(model_info, tokenizer,
      compile: [batch_size: 1, sequence_length: 8],
      defn_options: [compiler: EXLA]
    )
  end
end
