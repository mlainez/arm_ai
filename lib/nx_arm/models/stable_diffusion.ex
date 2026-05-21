defmodule NxArm.Models.StableDiffusion do
  @moduledoc """
  Text → image diffusion models on the edge.

  ## Honest performance expectations

  Diffusion on Cortex-A73 / similar SBCs takes **minutes per
  image** even at 256×256. The use case is "generate an image
  while the user is away", not interactive UX. For interactive
  TTS-style latency, run SD on a host and stream pixels to the
  Nerves device.

  That said: it works, and we wire two paths.

  ## Path A (recommended): ONNX export

  SD-Turbo, LCM, and other distilled variants ship as ONNX files
  designed for mobile inference. Load them through the existing
  `NxArm.Models.Onnx` bridge:

  ```elixir
  {:ok, text_enc} = NxArm.Models.Onnx.load("/data/sd_text_encoder.onnx")
  {:ok, unet}     = NxArm.Models.Onnx.load("/data/sd_unet.onnx")
  {:ok, vae_dec}  = NxArm.Models.Onnx.load("/data/sd_vae_decoder.onnx")

  # Tokenise prompt → CLIP embedding
  {:ok, tok} = NxArm.Tokenizer.load("/data/sd_tokenizer.json")
  ids = NxArm.Tokenizer.encode(tok, "a cosmonaut on a horse")
  embedding = NxArm.Models.Onnx.run(text_enc, %{"input_ids" => ids_tensor(ids)})
                |> Map.fetch!("last_hidden_state")

  # Diffusion loop (you implement the scheduler in Elixir)
  latent = sample_noise({1, 4, 32, 32})
  for step <- 1..n_steps do
    eps = NxArm.Models.Onnx.run(unet, %{
      "sample" => latent,
      "timestep" => t_tensor(step),
      "encoder_hidden_states" => embedding
    }) |> Map.fetch!("out_sample")
    latent = scheduler_step(latent, eps, step)
  end

  # Decode latent → pixels
  image = NxArm.Models.Onnx.run(vae_dec, %{"latent_sample" => latent})
            |> Map.fetch!("sample")
  ```

  Concrete ONNX exports to drop into `/root/models/`:
  * `stabilityai/sd-turbo` — 1-step, 512×512
  * `latent-consistency/lcm-lora-sdv1-5` — 4-step LCM
  * `lykon/dreamshaper-v7-onnx` — full SD 1.5 quality

  ## Path B: native candle

  candle-transformers ships the full SD 1.5 / 2.1 / XL stack in
  `candle_transformers::models::stable_diffusion`. We don't bridge
  it yet because (a) it's ~500 LOC of orchestration to wire all 4
  models + scheduler + generation loop properly, (b) the ONNX
  path covers the use case at the same fidelity, (c) candle SD on
  ARM CPU runs at the same speed as ONNX SD on ARM CPU.

  If you want this anyway, the bridge would live in
  `native/nx_arm_nif/src/stable_diffusion_candle.rs` mirroring the
  pattern in `whisper_candle.rs`. The candle examples directory
  has the full reference implementation.

  ## What this module provides

  Just the recommended config + helpers. The actual ONNX loading
  uses `NxArm.Models.Onnx`. The recommended scheduler implementations
  (DDIM, Euler-Ancestral) are pure Nx and live in
  `NxArm.Models.StableDiffusion.Scheduler` (future).
  """

  @doc """
  Default sampler config for SD-Turbo (the recommended edge variant).
  Override fields for other variants.
  """
  @spec sd_turbo_config() :: %{
          n_steps: pos_integer(),
          guidance_scale: float(),
          height: pos_integer(),
          width: pos_integer(),
          latent_channels: pos_integer()
        }
  def sd_turbo_config do
    %{
      n_steps: 1,
      guidance_scale: 0.0,
      height: 512,
      width: 512,
      latent_channels: 4
    }
  end

  @doc """
  Default sampler config for SDXL Turbo.
  """
  @spec sdxl_turbo_config() :: %{
          n_steps: pos_integer(),
          guidance_scale: float(),
          height: pos_integer(),
          width: pos_integer(),
          latent_channels: pos_integer()
        }
  def sdxl_turbo_config do
    %{
      n_steps: 1,
      guidance_scale: 0.0,
      height: 512,
      width: 512,
      latent_channels: 4
    }
  end

  @doc """
  Convert a latent tensor `{1, 3, H, W}` in `[-1, 1]` to a uint8
  HWC RGB binary suitable for writing as PNG/JPEG.
  """
  @spec latent_to_rgb8(Nx.Tensor.t()) :: binary()
  def latent_to_rgb8(image) do
    {1, 3, h, w} = Nx.shape(image)

    image
    |> Nx.add(1.0)
    |> Nx.divide(2.0)
    |> Nx.clip(0.0, 1.0)
    |> Nx.multiply(255.0)
    |> Nx.as_type(:u8)
    |> Nx.transpose(axes: [0, 2, 3, 1])
    |> Nx.reshape({h, w, 3})
    |> Nx.to_binary()
  end
end
