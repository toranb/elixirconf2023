defmodule Example.TextClassification do
  @moduledoc false

  @batch_size 32
  @sequence_length 64

  alias Bumblebee.Shared

  def text_classification(model_info, tokenizer, _opts \\ []) do
    %{model: model, params: params, spec: spec} = model_info
    Shared.validate_architecture!(spec, :for_sequence_classification)

    compile = false
    batch_size = 32
    defn_options = [compiler: EXLA]

    logits_model = Axon.nx(model, & &1.logits)
    {_init_fn, predict_fun} = Axon.build(logits_model, compiler: EXLA)

    scores_fun = fn params, input ->
      predict_fun.(params, input)
    end

    Nx.Serving.new(
      fn defn_options ->
        serving_scores_fun =
          Shared.compile_or_jit(scores_fun, defn_options, compile != nil, fn ->
            inputs = %{
              "input_ids" => Nx.template({@batch_size, @sequence_length}, :u32),
              "attention_mask" => Nx.template({@batch_size, @sequence_length}, :u32)
            }

            [params, inputs]
          end)

        fn inputs ->
          inputs = Shared.maybe_pad(inputs, @batch_size)
          serving_scores_fun.(params, inputs)
        end
      end,
      defn_options
    )
    |> Nx.Serving.process_options(batch_size: batch_size)
    |> Nx.Serving.client_preprocessing(fn input ->
      {texts, multi?} = Shared.validate_serving_input!(input, &Shared.validate_string/1)

      inputs =
        Bumblebee.apply_tokenizer(tokenizer, texts,
          length: @sequence_length,
          return_token_type_ids: false
        )

      {Nx.Batch.concatenate([inputs]), multi?}
    end)
    |> Nx.Serving.client_postprocessing(fn scores, _metadata, multi? ->
      for scores <- Bumblebee.Utils.Nx.batch_to_list(scores) do
        result = scores |> Nx.argmax() |> Nx.to_flat_list()

        %{result: result}
      end
      |> Shared.normalize_output(multi?)
    end)
  end
end
