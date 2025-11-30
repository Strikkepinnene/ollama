# Ollama Codebase Guide for AI Agents

## Project Overview

Ollama is a Go application (Go 1.24.1) that enables users to run large language models locally. It provides a CLI, HTTP API server, and model management system with native GPU acceleration via llama.cpp integration.

**Key Architecture Components:**
- **CLI (`cmd/`)**: Cobra-based command interface (`ollama serve`, `ollama run`, `ollama create`, etc.)
- **Server (`server/`)**: Gin-based HTTP server with OpenAI-compatible API endpoints
- **Scheduler (`server/sched.go`)**: Manages model loading/unloading across GPUs with memory optimization
- **LLM Backend (`llm/`)**: CGO interface to llama.cpp runners (platform-specific: `llm_darwin.go`, `llm_linux.go`, `llm_windows.go`)
- **Model Conversion (`convert/`)**: Converts PyTorch/Safetensors models to GGUF format (supports Llama, Gemma, Qwen, Mistral, etc.)
- **Model Registry (`types/model/`)**: Model name parsing with format `host/namespace/model:tag` (defaults: `registry.ollama.ai/library/model:latest`)

## Build System

**Critical**: Ollama uses CGO extensively for llama.cpp integration. Build failures often stem from stale CGO cache.

### Standard Build
```bash
# Force clean build if seeing unexpected crashes
go clean -cache

# Development server
go run . serve

# Production build (requires CMake + C++ compiler)
cmake -B build
cmake --build build
go build .
```

### Platform Requirements
- **macOS**: Metal support built-in (Apple Silicon), CMake needed (Intel)
- **Windows**: CMake + Visual Studio 2022 + optional CUDA/ROCm
- **Linux**: CMake + optional CUDA/ROCm

### GPU Support
Uses CMake presets for acceleration. See `CMakePresets.json` and `CMakeLists.txt` for GPU configurations (CUDA, ROCm, Metal).

## Testing Strategy

```bash
# Unit tests only
go test ./...

# Integration tests (requires built binary)
go build .  # Must build first
go test -tags=integration ./...

# Extended model tests (~60m+ runtime)
go test -tags=integration,models -timeout 60m ./...

# Enable synctest (Go 1.24 experimental)
GOEXPERIMENT=synctest go test ./...
```

**Integration Test Modes** (`integration/`):
1. Default: Auto-starts server on random port (Unix only; Windows requires manual `OLLAMA_HOST` setup)
2. `OLLAMA_TEST_EXISTING=1`: Test against running server at `OLLAMA_HOST`

Override test model via `OLLAMA_TEST_DEFAULT_MODEL`.

## Environment Configuration

All configuration via `envconfig/config.go` using `OLLAMA_*` variables:

```bash
OLLAMA_HOST=127.0.0.1:11434          # Server bind address
OLLAMA_MODELS=/path/to/models        # Model storage directory
OLLAMA_KEEP_ALIVE=5m                 # Model memory retention
OLLAMA_NUM_PARALLEL=1                # Concurrent request limit
OLLAMA_DEBUG=1                       # Enable debug logging
OLLAMA_FLASH_ATTENTION=1             # Enable flash attention
OLLAMA_ORIGINS=https://example.com   # CORS allowed origins
OLLAMA_LOAD_TIMEOUT=5m               # Model load stall detection
OLLAMA_EXPERIMENT=client2,harmony    # Feature flags (comma-separated)
```

GPU-specific vars logged in `llm/server.go:filteredEnv` (CUDA_*, ROCM_*, HIP_*, etc.).

## Code Conventions

### Model Names
Use `types/model.Name` struct for all model references. Never parse manually:
```go
// Correct
name, err := model.ParseNameNoDefaults("library/llama3.2:8b")
// name.Namespace = "library", name.Model = "llama3.2", name.Tag = "8b"

// Never do string manipulation on model names
```

### API Types
Requests/responses defined in `api/types.go`. Core types:
- `GenerateRequest/Response`: Text generation
- `ChatRequest/Response`: Chat completions (OpenAI-compatible)
- `EmbedRequest/Response`: Embeddings
- `Options`: Model parameters (temperature, num_ctx, etc.)

### Scheduler Pattern
Models loaded via `Scheduler.GetRunner()` which returns channels. Always cancel context to release:
```go
successCh, errCh := sched.GetRunner(ctx, model, opts, sessionDuration)
select {
case runner := <-successCh:
    defer runner.decrement() // Critical: prevents memory leaks
case err := <-errCh:
    // handle error
}
```

### Modelfile Structure
Defines model configurations (`docs/modelfile.mdx`):
- `FROM`: Base model (required) - accepts model names or local GGUF/Safetensors paths
- `PARAMETER`: Runtime options (temperature, num_ctx, stop sequences)
- `TEMPLATE`: Go template for prompt formatting (variables: `.System`, `.Prompt`, `.Response`)
- `SYSTEM`: Default system message
- `MESSAGE`: Conversation history

## Platform-Specific Code

Uses `//go:build` tags extensively:
- `llm/llm_darwin.go`, `llm_linux.go`, `llm_windows.go`: Platform-specific GPU discovery
- `readline/term_bsd.go`, `term_linux.go`: Terminal handling
- Integration tests: `//go:build integration` or `//go:build integration && models`

## Model Conversion Pipeline

Located in `convert/` - converts external models to Ollama's GGUF format:
1. Read source format (Safetensors via `reader_safetensors.go`, PyTorch via `reader_torch.go`)
2. Architecture detection (`convert_*.go` per model family)
3. Tensor conversion + quantization
4. Tokenizer extraction (`tokenizer.go`, `tokenizer_spm.go` for SentencePiece)

**Important**: Each model family (Llama, Gemma, Qwen, etc.) has dedicated `convert_<family>.go` implementing architecture-specific logic.

## Commit Message Format

Follow strict format from `CONTRIBUTING.md`:
```
<package>: <short description>

Examples:
  llm/backend/mlx: support the llama architecture
  server/sched: fix memory leak in runner cleanup
  envconfig: add OLLAMA_MAX_VRAM setting
```

Use lowercase start, imperative mood ("add", not "adds" or "added"). Avoid generic prefixes (feat:, fix:, chore:).

## Dependencies

Minimal external deps (`go.mod`):
- Gin (HTTP framework)
- Cobra (CLI)
- SQLite (model metadata storage)
- golang.org/x/sync, x/sys (Go extended libs)

**Adding Dependencies**: Requires maintainer approval. Explain necessity and alternative approaches tried (see `CONTRIBUTING.md`).

## Debugging Tips

1. **Model Load Failures**: Check `OLLAMA_DEBUG=1` logs for memory allocation details in `server/sched.go`
2. **CGO Issues**: Run `go clean -cache` before rebuilding
3. **GPU Detection**: Platform-specific in `discover/gpu_*.go` - check `discover.GetSystemInfo()`
4. **Template Errors**: Validate against `template/` gotmpl files and harmony parser (`server/routes.go:shouldUseHarmony`)

## Key File Locations

- Entry point: `main.go` → `cmd/cmd.go`
- HTTP routes: `server/routes.go` (all API handlers)
- Model scheduler: `server/sched.go` (GPU memory management)
- Environment config: `envconfig/config.go` (all OLLAMA_* vars)
- Model format: `fs/ggml/` (GGUF file parsing)
- llama.cpp sync: `Makefile.sync` (syncs from upstream, applies patches)
