package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fqadri/blizzard/internal/apiserver"
	"github.com/fqadri/blizzard/internal/llmengine"
	"github.com/fqadri/blizzard/internal/modelexecutor"
	"github.com/fqadri/blizzard/internal/modelworker"
	"github.com/fqadri/blizzard/internal/scheduler"
)

const (
	listenAddr       = ":8080"
	requestQueueSize = 100
	maxBatchSize     = 8
	maxTokens        = 1024
	startTimeout     = 5 * time.Minute // model load
	stepTimeout      = 30 * time.Second
)

func main() {
	if err := run(); err != nil {
		log.Fatalf("blizzard: %v", err)
	}
}

func run() error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// bind before loading the model so a busy port fails in seconds, not after a multi-minute load
	listener, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return fmt.Errorf("listen on %s: %w", listenAddr, err)
	}
	defer listener.Close()

	worker := modelworker.New(modelworker.Config{
		Python:       envOr("BLIZZARD_PYTHON", "python"),
		Script:       envOr("BLIZZARD_WORKER", "python/worker.py"),
		Args:         []string{"--model", envOr("BLIZZARD_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")},
		StartTimeout: startTimeout,
	})

	// blocks until the model is loaded, so no request can arrive before the worker can serve it
	if err := worker.Start(ctx); err != nil {
		return fmt.Errorf("start model worker: %w", err)
	}

	defer func() { _ = worker.Stop() }()

	queue := scheduler.New[*llmengine.Sequence](requestQueueSize)
	executor := modelexecutor.New(worker, stepTimeout)
	engine := llmengine.New(queue, executor, maxTokens)

	// this will run the engine in a separate goroutine
	go func() {
		if err := engine.Run(ctx, maxBatchSize); err != nil {
			log.Printf("blizzard: engine stopped unexpectedly: %v", err)
		}
	}()

	server := apiserver.New(engine)

	serverErr := make(chan error, 1)
	go func() { serverErr <- server.Serve(listener) }()

	log.Printf("blizzard: listening on %s", listenAddr)

	// wait for either the server
	// 1. to return an error or
	// 2. for a termination signal
	select {
	case err := <-serverErr:
		return err
	case <-ctx.Done():
		log.Print("blizzard: shutting down")
		return nil
	}
}

func envOr(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}
