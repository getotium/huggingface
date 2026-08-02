# huggingface

A small, **dependency-free** Go client for the [Hugging Face Hub](https://huggingface.co) API. It
returns HF-native types (`ModelInfo`, `Safetensors`, gated status, …) and does nothing clever —
search the Hub and fetch model metadata, no third-party dependencies.

- **Stdlib-only.** No dependencies to vet.
- **Configurable** via options: custom `*http.Client`, base URL, user agent.
- **HF-native types.** No opinionated remapping — you get the Hub's own fields.

## Install

```
go get github.com/getotium/huggingface
```

## Usage

```go
c := huggingface.New()

// Search the Hub.
models, err := c.Search(ctx, huggingface.SearchOptions{Search: "qwen", Limit: 20})

// Page through results.
page, cursor, err := c.SearchPage(ctx, huggingface.SearchOptions{Search: "gemma"})

// Fetch one model's metadata (parameters, dtype, gated status, safetensors shard info…).
info, err := c.Model(ctx, "Qwen/Qwen2.5-7B-Instruct")
```

Options:

```go
c := huggingface.New(
    huggingface.WithHTTPClient(myClient),
    huggingface.WithUserAgent("my-app/1.0"),
)
```

## Provenance

Extracted from [Otium](https://getotium.ai), where it powers model discovery for the inference
catalog. Kept deliberately generic (Hub API only — no VRAM math or catalog mapping), so it's a
clean, reusable primitive.

## License

[Apache-2.0](LICENSE).
