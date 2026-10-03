package modelworker

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os/exec"
	"sync"
	"time"
)

var ErrNotRunning = errors.New("modelworker: worker is not running")

type Config struct {
	Python       string        // interpreter path
	Script       string        // worker script path
	Args         []string      // passed after the script, never through a shell
	StartTimeout time.Duration // bounds the wait for the ready message; zero means unbounded
	Logger       *slog.Logger
}

// Worker owns one child process and exchanges a single JSON value per call over its pipes.
type Worker struct {
	cfg Config

	mu         sync.Mutex // the wire protocol is strictly one request, one response
	cmd        *exec.Cmd
	stdin      io.WriteCloser
	enc        *json.Encoder
	dec        *json.Decoder
	stderrDone chan struct{} // used for signaling when stderr pumping is done
}

// readyMessage is sent by the model python script to indicate it is ready to accept requests.
type readyMessage struct {
	Ready bool   `json:"ready"`
	Error string `json:"error,omitempty"`
}

func New(cfg Config) *Worker {
	if cfg.Logger == nil {
		cfg.Logger = slog.Default()
	}
	return &Worker{cfg: cfg}
}

// Start launches the child and blocks until it reports readiness, which covers model load.
// ctx governs the lifetime of the child process, so it must outlive every Call.
func (w *Worker) Start(ctx context.Context) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.cmd != nil {
		return errors.New("modelworker: already started")
	}

	// -u is mandatory: a buffered child stdout deadlocks the request/response cycle
	args := append([]string{"-u", w.cfg.Script}, w.cfg.Args...)
	cmd := exec.CommandContext(ctx, w.cfg.Python, args...)
	cmd.WaitDelay = 10 * time.Second

	configureProc(cmd)

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("modelworker: stdin pipe: %w", err)
	}

	// closing stdin is the child's designed exit signal, and works the same on both platforms
	cmd.Cancel = func() error { return stdin.Close() }
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("modelworker: stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("modelworker: stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("modelworker: start: %w", err)
	}

	if err := afterStart(cmd); err != nil {
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return err
	}

	w.cmd = cmd
	w.stdin = stdin
	w.enc = json.NewEncoder(stdin)
	w.dec = json.NewDecoder(stdout)
	w.stderrDone = make(chan struct{})

	go w.pumpStderr(stderr, w.stderrDone)

	readyCtx := ctx
	if w.cfg.StartTimeout > 0 {
		var cancel context.CancelFunc
		readyCtx, cancel = context.WithTimeout(ctx, w.cfg.StartTimeout)
		defer cancel()
	}

	var ready readyMessage
	if err := w.decodeLocked(readyCtx, &ready); err != nil {
		_ = w.stopLocked()
		return fmt.Errorf("modelworker: waiting for ready: %w", err)
	}
	if !ready.Ready {
		_ = w.stopLocked()
		return fmt.Errorf("modelworker: worker failed to start: %s", ready.Error)
	}

	w.cfg.Logger.Info("model worker ready", "pid", cmd.Process.Pid)
	return nil
}

// Call writes req and decodes the reply into resp. Any failure stops the worker,
// because a half-read pipe cannot be resynchronised.
// resp must not be read when Call returns an error; an abandoned decode may still be writing to it.
func (w *Worker) Call(ctx context.Context, req, resp any) error {
	w.mu.Lock()
	defer w.mu.Unlock()

	if w.cmd == nil {
		return ErrNotRunning
	}

	if err := w.enc.Encode(req); err != nil {
		_ = w.stopLocked()
		return fmt.Errorf("modelworker: write request: %w", err)
	}

	if err := w.decodeLocked(ctx, resp); err != nil {
		_ = w.stopLocked()
		return fmt.Errorf("modelworker: read response: %w", err)
	}

	return nil
}

func (w *Worker) Running() bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.cmd != nil
}

func (w *Worker) Stop() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.stopLocked()
}

// decodeLocked abandons the decode on cancellation; the caller must stop the worker afterwards.
func (w *Worker) decodeLocked(ctx context.Context, v any) error {
	dec := w.dec // captured here because stopLocked may nil the field before the goroutine runs

	done := make(chan error, 1)
	go func() { done <- dec.Decode(v) }()

	select {
	case err := <-done:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// stopLocked closes stdin so the child sees EOF and exits on its own.
func (w *Worker) stopLocked() error {
	if w.cmd == nil {
		return nil
	}

	cmd, stdin, stderrDone := w.cmd, w.stdin, w.stderrDone
	w.cmd, w.stdin, w.enc, w.dec, w.stderrDone = nil, nil, nil, nil, nil

	_ = stdin.Close()

	done := make(chan error, 1)
	go func() {
		<-stderrDone // Wait closes the pipes, so all reads must finish first
		done <- cmd.Wait()
	}()

	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		_ = cmd.Process.Kill()
		return <-done
	}
}

func (w *Worker) pumpStderr(r io.Reader, done chan<- struct{}) {
	defer close(done)

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 1<<20) // tracebacks exceed the default 64KB line cap
	for scanner.Scan() {
		w.cfg.Logger.Info("model worker stderr", "line", scanner.Text()) // failures surface through the protocol, so stderr is plain logging
	}
}
