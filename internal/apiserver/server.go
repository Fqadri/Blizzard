package apiserver

import (
	"context"
	"net"
	"net/http"
	"time"
)

type Config struct {
	Model     string // served model; requests naming any other model get 404
	MaxTokens int    // largest max_tokens a request may ask for
}

type Server struct {
	httpServer *http.Server
	generator  Generator
	config     Config
}

// Serve accepts connections on ln until the server is shut down.
// returns
// - http.ErrServerClosed as soon as Shutdown or Close is called
// - any other error encountered while serving
func (server *Server) Serve(ln net.Listener) error {
	return server.httpServer.Serve(ln)
}

// Shutdown stops accepting connections and waits for in-flight handlers to return.
func (server *Server) Shutdown(ctx context.Context) error {
	return server.httpServer.Shutdown(ctx)
}

func New(generator Generator, config Config) *Server {
	server := &Server{generator: generator, config: config}
	server.httpServer = &http.Server{
		Handler:           server.routes(),
		ReadHeaderTimeout: 5 * time.Second,
	}
	return server
}
