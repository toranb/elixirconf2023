defmodule Example.Evaluate do
  @moduledoc false

  NimbleCSV.define(OrdersParser, separator: ",", escape: "\"")

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
