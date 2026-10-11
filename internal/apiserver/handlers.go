package apiserver

import (
	"context"
	"crypto/rand"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"time"

	"github.com/fqadri/blizzard/internal/chat"
	stream "github.com/fqadri/blizzard/internal/stream"
)

// how long one write may block on a client that stopped reading; like nginx send_timeout
const writeTimeout = 30 * time.Second

type Generator interface {
	GenerateStream(
		ctx context.Context,
		messages []chat.Message,
		maxTokens int) (stream.Stream, error)
}

// writeEvent writes and flushes one SSE event under a write deadline.
// we are doing this to ensure that each write has a bounded time to complete, preventing a slow client from blocking indefinitely.
func writeEvent(writer http.ResponseWriter, controller *http.ResponseController, event string) error {
	if err := controller.SetWriteDeadline(time.Now().Add(writeTimeout)); err != nil {
		return err
	}
	if _, err := io.WriteString(writer, event); err != nil {
		return err
	}
	if err := controller.Flush(); err != nil {
		return err
	}

	// clear the write deadline so that subsequent writes aren't prematurely timed out
	return controller.SetWriteDeadline(time.Time{})
}

// status is already sent by this point, so failure can only be signaled in-band, as a data line holding an error object
func writeStreamError(writer http.ResponseWriter, controller *http.ResponseController, message string) {
	payload, err := json.Marshal(errorResponse{Error: errorInfo{Message: message, Type: "server_error"}})
	if err != nil {
		slog.Error("encode stream error failed", "error", err)
		return
	}

	if writeEvent(writer, controller, fmt.Sprintf("data: %s\n\n", payload)) == nil {
		_ = writeEvent(writer, controller, "data: [DONE]\n\n")
	}
}

func writeChunk(writer http.ResponseWriter, controller *http.ResponseController, chunk chatCompletionChunk, choice chunkChoice) error {
	chunk.Choices = []chunkChoice{choice}
	payload, err := json.Marshal(chunk)
	if err != nil {
		return err
	}
	return writeEvent(writer, controller, fmt.Sprintf("data: %s\n\n", payload))
}

// handleChatCompletions serves OpenAI chat completions; only streaming responses are supported so far.
func (server *Server) handleChatCompletions(writer http.ResponseWriter, request *http.Request) {
	request.Body = http.MaxBytesReader(writer, request.Body, 1<<20)

	var input chatCompletionRequest
	if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
		http.Error(writer, "invalid JSON request", http.StatusBadRequest)
		return
	}

	if input.Model == "" {
		http.Error(writer, "model is required", http.StatusBadRequest)
		return
	}
	if input.Model != server.config.Model {
		http.Error(writer, fmt.Sprintf("model %q not found", input.Model), http.StatusNotFound)
		return
	}

	// stream defaults to false in the spec, so an omitted stream also asks for a single JSON response
	if !input.Stream {
		http.Error(writer, `only streaming is supported: set "stream": true`, http.StatusBadRequest)
		return
	}

	if len(input.Messages) == 0 {
		http.Error(writer, "messages is required", http.StatusBadRequest)
		return
	}
	for i, message := range input.Messages {
		switch message.Role {
		case "system", "user", "assistant":
		default:
			http.Error(writer, fmt.Sprintf("messages[%d]: role %q is not supported; use system, user or assistant", i, message.Role), http.StatusBadRequest)
			return
		}
	}

	// chat completions have no default length, so an unset limit means the server's cap
	maxTokens := server.config.MaxTokens
	switch {
	case input.MaxCompletionTokens != nil:
		maxTokens = *input.MaxCompletionTokens
	case input.MaxTokens != nil:
		maxTokens = *input.MaxTokens
	}
	if maxTokens < 1 || maxTokens > server.config.MaxTokens {
		http.Error(writer, fmt.Sprintf("max_completion_tokens must be between 1 and %d", server.config.MaxTokens), http.StatusBadRequest)
		return
	}

	ctx := request.Context()

	tokenStream, err := server.generator.GenerateStream(ctx, input.Messages, maxTokens)
	if err != nil {
		slog.Error("generation failed", "error", err)
		// todo: GenerateStream can fail for queue full which should be handled separately
		http.Error(writer, "generation failed", http.StatusInternalServerError)
		return
	}

	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")

	controller := http.NewResponseController(writer)

	// commits headers and verifies this writer supports flushing and write deadlines
	if err := writeEvent(writer, controller, ""); err != nil {
		slog.Error("streaming unsupported", "error", err)
		http.Error(writer, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	// every chunk of one completion shares its id, timestamp and model
	chunk := chatCompletionChunk{
		ID:      "chatcmpl-" + rand.Text(),
		Object:  "chat.completion.chunk",
		Created: time.Now().Unix(),
		Model:   server.config.Model,
	}

	// like OpenAI, the first chunk announces the assistant role with empty content
	empty := ""
	if err := writeChunk(writer, controller, chunk, chunkChoice{Delta: delta{Role: "assistant", Content: &empty}}); err != nil {
		return
	}

	for {
		// first check if the context has been canceled before attempting to receive the next token
		select {
		case <-ctx.Done():
			return
		default:
		}

		token, err := tokenStream.Recv(ctx)
		if err != nil {
			if ctx.Err() != nil { // context error: client is gone, nothing left to write to
				return
			}

			if errors.Is(err, io.EOF) { // this indicates the stream has been closed
				reason := tokenStream.Reason()
				if writeChunk(writer, controller, chunk, chunkChoice{FinishReason: &reason}) == nil {
					_ = writeEvent(writer, controller, "data: [DONE]\n\n")
				}
				return
			}

			// Info, not Error: the engine logs root causes where they happen, and shutdown also ends streams this way
			slog.Info("stream ended with error", "error", err, "finish_reason", tokenStream.Reason())
			writeStreamError(writer, controller, "generation failed")
			return
		}

		if err := writeChunk(writer, controller, chunk, chunkChoice{Delta: delta{Content: &token}}); err != nil {
			return
		}
	}
}
