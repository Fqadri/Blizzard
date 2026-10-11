// Package chat holds the conversation types shared by the API server and the engine.
package chat

// Message is one turn of a conversation, as in OpenAI chat completions.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}
