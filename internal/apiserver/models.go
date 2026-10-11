package apiserver

import "github.com/fqadri/blizzard/internal/chat"

type chatCompletionRequest struct {
	Model               string         `json:"model"`
	Messages            []chat.Message `json:"messages"`
	MaxCompletionTokens *int           `json:"max_completion_tokens"`
	MaxTokens           *int           `json:"max_tokens"` // deprecated by OpenAI in favor of max_completion_tokens, but still widely sent
	Stream              bool           `json:"stream"`
}

// chatCompletionChunk is one streamed chunk in the OpenAI chat completions shape
type chatCompletionChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
}

type chunkChoice struct {
	Index        int     `json:"index"`
	Delta        delta   `json:"delta"`
	FinishReason *string `json:"finish_reason"` // null until the final chunk
}

// delta is what one chunk adds to the assistant message: the role first, then content, and nothing in the final chunk
type delta struct {
	Role    string  `json:"role,omitempty"`
	Content *string `json:"content,omitempty"`
}

// errorResponse is OpenAI's error body
type errorResponse struct {
	Error errorInfo `json:"error"`
}

type errorInfo struct {
	Message string  `json:"message"`
	Type    string  `json:"type"`
	Param   *string `json:"param"`
	Code    *string `json:"code"`
}
