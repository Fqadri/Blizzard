package apiserver

import (
	"net/http"
)

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)

	return mux
}
