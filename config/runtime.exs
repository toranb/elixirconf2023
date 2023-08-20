import Config

config :nx, :default_backend, EXLA.Backend

config :exla, :clients,
  cuda: [platform: :cuda, preallocate: false],
  host: [platform: :host]
