package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"

	stream "github.com/fqadri/blizzard/internal/stream"
)

type Generator interface {
	GenerateStream(ctx context.Context, prompt string) (stream.Stream, error)
}

type generateRequest struct {
	Prompt string `json:"prompt"`
}

type generateResponse struct {
	GeneratedText string `json:"generated_text"`
}

type terminalEvent struct {
	FinishReason string `json:"finish_reason"`
	Error        string `json:"error,omitempty"`
}

// status is already sent by this point, so completion and failure can only be signaled in-band
func writeTerminalEvent(writer http.ResponseWriter, controller *http.ResponseController, name string, event terminalEvent) {
	payload, err := json.Marshal(event)
	if err != nil {
		slog.Error("encode terminal event failed", "error", err)
		return
	}

	fmt.Fprintf(writer, "event: %s\ndata: %s\n\n", name, payload)
	controller.Flush()
}

// handleStreamGenerate handles streaming generation requests using Server's generator.
func (server *Server) handleStreamGenerate(writer http.ResponseWriter, request *http.Request) {
	request.Body = http.MaxBytesReader(writer, request.Body, 1<<20)

	var input generateRequest
	if err := json.NewDecoder(request.Body).Decode(&input); err != nil {
		http.Error(writer, "invalid JSON request", http.StatusBadRequest)
		return
	}

	if strings.TrimSpace(input.Prompt) == "" {
		http.Error(writer, "prompt is required", http.StatusBadRequest)
		return
	}

	ctx := request.Context()

	tokens, err := server.generator.GenerateStream(ctx, input.Prompt)
	if err != nil {
		slog.Error("generation failed", "error", err)
		// todo: GenerateStream can fail for queue full which should be handled separately
		http.Error(writer, "generation failed", http.StatusInternalServerError)
		return
	}

	writer.Header().Set("Content-Type", "text/event-stream")
	writer.Header().Set("Cache-Control", "no-cache")

	controller := http.NewResponseController(writer)

	// commits headers and verifies this writer supports flushing
	if err := controller.Flush(); err != nil {
		slog.Error("streaming unsupported", "error", err)
		http.Error(writer, "streaming unsupported", http.StatusInternalServerError)
		return
	}

	for {
		// first check if the context has been canceled before attempting to receive the next token
		select {
		case <-ctx.Done():
			return
		default:
		}

		token, err := tokens.Recv(ctx)
		if err != nil {
			if ctx.Err() != nil { // client is gone, nothing left to write to
				return
			}
			if errors.Is(err, io.EOF) { // this indicates the stream has been closed
				writeTerminalEvent(writer, controller, "done", terminalEvent{FinishReason: tokens.Reason()})
				return
			}

			slog.Error("stream receive failed", "error", err, "finish_reason", tokens.Reason())
			writeTerminalEvent(writer, controller, "error", terminalEvent{
				FinishReason: tokens.Reason(),
				Error:        "generation failed",
			})
			return
		}

		payload, err := json.Marshal(generateResponse{GeneratedText: token})
		if err != nil {
			slog.Error("encode token failed", "error", err)
			writeTerminalEvent(writer, controller, "error", terminalEvent{
				FinishReason: "error",
				Error:        "generation failed",
			})
			return
		}

		if _, err := fmt.Fprintf(writer, "data: %s\n\n", payload); err != nil {
			return
		}
		if err := controller.Flush(); err != nil {
			return
		}
	}
}
