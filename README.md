# Fine Tuning Language Models With Axon

The fine tuning, evaluation, Nx serving and embedding code from ElixirConf 2023

## Tune

The [tune](https://github.com/toranb/elixirconf2023/blob/main/lib/example/tune.ex) module for fine tuning with Roberta

```elixir
Example.tune()
```

## Evaluate

The [evaluate](https://github.com/toranb/elixirconf2023/blob/main/lib/example/evaluate.ex) module for evaluation

```elixir
Example.evaluate()
```

## Embeddings

The language model [embeddings](https://github.com/toranb/elixirconf2023/blob/main/lib/example.ex#L16-L25) can be generated from the token ids

```elixir
Example.get_token_ids() |> Example.embeddings()
```

## Nx Serving

The [cancellations](https://github.com/toranb/elixirconf2023/blob/main/lib/example/evaluate.ex#L32) can be generated for a given dataset with Nx Serving

```elixir
Example.cancellations()
```

To use this you must first fine tune the model to generate `cancel.axon` then you need to uncomment [this](https://github.com/toranb/elixirconf2023/blob/main/mix.exs#L17) in the mix.exs file and run `iex -S mix run`
