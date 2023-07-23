defmodule Example do
  @moduledoc false

  def tune() do
    Example.Tune.train()
  end

  def evaluate() do
    Example.Evaluate.cancellations()
  end
end
