package apiserver

import (
	"net/http"
)

func (s *Server) routes() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/stream_generate", s.handleStreamGenerate)

	return mux
}
