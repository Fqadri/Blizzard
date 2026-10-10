package apiserver

import (
	"net/http"
)

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("POST /stream_generate", s.handleStreamGenerate)

	return mux
}
