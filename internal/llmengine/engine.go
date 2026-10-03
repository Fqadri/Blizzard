package llmengine

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"

	stream "github.com/fqadri/blizzard/internal/stream"
)

// finish reasons reported to the client on the terminal event
const (
	FinishReasonStop      = "stop"
	FinishReasonLength    = "length"
	FinishReasonError     = "error"
	FinishReasonCancelled = "cancelled"
)

var (
	ErrExecutorFailed = errors.New("llmengine: model executor failed")
	ErrEngineShutdown = errors.New("llmengine: engine shutting down")
	ErrClientTooSlow  = errors.New("llmengine: client fell behind token stream")
)

type StepResult struct {
	ID    uint64
	Token string
	Done  bool
}

type modelExecutor interface {
	// ExecuteStep executes a single inference step for a batch and returns the result tokens.
	// To support streaming, you have to execute steps incrementally using ExecuteStep.
	ExecuteStep(ctx context.Context, seq []*Sequence) ([]StepResult, error)
}

type scheduler interface {
	Submit(seq *Sequence) error
	NextBatch(max int) ([]*Sequence, error)
	Ready() <-chan struct{} // returns a channel that is signaled when pending work is available
	Remove(seq *Sequence)
}

type Sequence struct {
	ID     uint64
	Prompt string        // original prompt
	Output []string      // only the length is used; the tokens are retained for diagnostics
	Stream stream.Stream // where the engine sends each token; the handler holds the receive side

	terminateOnce sync.Once   // ensures that the sequence is only terminated once
	closed        atomic.Bool // indicates whether the sequence has been closed
}

type engine struct {
	modelExecutor modelExecutor
	scheduler     scheduler
	seqID         atomic.Uint64 // monotonically increasing identifier for the sequence
	maxTokens     int           // maximum number of tokens to generate per sequence
}

func (e *engine) nextID() uint64 {
	return e.seqID.Add(1)
}

func New(
	scheduler scheduler,
	modelExecutor modelExecutor,
	maxTokens int) *engine {
	return &engine{scheduler: scheduler, modelExecutor: modelExecutor, maxTokens: maxTokens}
}

// GenerateStream generates a stream for the given prompt
func (e *engine) GenerateStream(ctx context.Context, prompt string) (stream.Stream, error) {
	seq := &Sequence{ // encapsulate the request prompt into an input sequence object
		ID:     e.nextID(),
		Prompt: prompt,
		Stream: stream.NewChannelStream(),
	}

	if err := e.scheduler.Submit(seq); err != nil { // put it on the scheduler
		e.terminate(seq, FinishReasonError, err)
		return nil, err
	}

	// if context is already canceled, this will run right away
	context.AfterFunc(ctx, func() {
		e.scheduler.Remove(seq)
		e.terminate(seq, FinishReasonCancelled, ctx.Err())
	})

	return seq.Stream, nil
}

// Run is a state machine that continuously fetches batches from the scheduler and executes them using the model executor
// Run returns nil when ctx is canceled, which signals application shutdown.
func (e *engine) Run(ctx context.Context, maxBatchSize int) error {
	if maxBatchSize <= 0 {
		return errors.New("llmengine: maxBatchSize must be positive")
	}

	active := []*Sequence{} // active set of sequences currently being processed
	defer func() { e.terminateAll(active, FinishReasonError, ErrEngineShutdown) }()

	for {
		if ctx.Err() != nil { // context canceled, exit the loop
			return nil
		}

		// if there is nothing to process on active set, either wait for
		// 1. scheduler to signal work or
		// 2. context to be done
		if len(active) == 0 {
			select {
			case <-e.scheduler.Ready():
			case <-ctx.Done():
				return nil
			}
		}

		// drop sequences the cancel hook terminated between steps. it means request from client side got canceled.
		kept := active[:0]
		for _, seq := range active {
			if seq.closed.Load() {
				continue
			}
			kept = append(kept, seq)
		}
		active = kept

		// continuously batch: fetch a new batch from the scheduler if there is free capacity in the active set
		if free := maxBatchSize - len(active); free > 0 {
			batch, err := e.scheduler.NextBatch(free)
			if err != nil {
				return err // future safety net
			}
			active = append(active, batch...)
		}

		if len(active) == 0 {
			continue
		}

		results, err := e.modelExecutor.ExecuteStep(ctx, active) // one step
		if err != nil {
			if ctx.Err() != nil { // shutdown abandoned the step; the deferred terminateAll reports it
				return nil
			}
			slog.Error("execute step failed, terminating active batch", "error", err, "batch_size", len(active))
			active = e.terminateAll(active, FinishReasonError, ErrExecutorFailed)
			continue
		}

		active = e.deliver(active, results) // keeps only unfinished
	}
}

// deliver processes the results of a step execution for the active sequences and streams the generated tokens to the clients.
// terminates sequences for the following conditions:
// 1. model executor returned no result for the sequence. todo: there could be legimate cases where this happens so handle them appropriately
// 2. send to client stream failed
// 3. sequence has completed normally (finished)
// 4. sequence has reached the maximum allowed token length (finished)
func (e *engine) deliver(
	active []*Sequence,
	results []StepResult) []*Sequence {

	// make map to quickly look up step result for a specific sequence ID
	byID := make(map[uint64]StepResult, len(results))
	for _, r := range results {
		byID[r.ID] = r
	}

	kept := active[:0]
	for _, seq := range active {
		res, ok := byID[seq.ID]
		if !ok {
			// 1. model executor returned no result for the sequence
			slog.Warn("executor returned no result for sequence",
				"seq_id", seq.ID, "tokens_generated", len(seq.Output), "batch_size", len(active))
			e.terminate(seq, FinishReasonError, ErrExecutorFailed)
			continue
		}

		// an empty token is the stop token or half of a multi-byte character; it still counts toward maxTokens
		if res.Token != "" {
			if err := seq.Stream.Send(res.Token); err != nil { // 2. send to client stream failed
				switch {
				case errors.Is(err, stream.ErrStreamFull):
					slog.Warn("dropping sequence, client too slow", "seq_id", seq.ID, "tokens_generated", len(seq.Output))
					e.terminate(seq, FinishReasonError, ErrClientTooSlow)
				case errors.Is(err, stream.ErrStreamClosed):
					// cancel hook already terminated it; the first reason wins
				default:
					e.terminate(seq, FinishReasonError, err) // safety net for future
				}
				continue
			}
		}

		// send is success, nowappend token to output
		seq.Output = append(seq.Output, res.Token)

		switch {
		case res.Done: // 3. sequence has completed normally
			e.terminate(seq, FinishReasonStop, nil)
		case len(seq.Output) >= e.maxTokens: // 4. sequence has reached the maximum allowed token length
			e.terminate(seq, FinishReasonLength, nil)
		default:
			kept = append(kept, seq)
		}
	}

	return kept
}

// terminate ends a sequence and ensures it is closed exactly once
// can be called either via
// 1. the cancel hook registered in the engine triggered from http request context at any point
// 2. the deliver method while processing results
func (e *engine) terminate(seq *Sequence, reason string, err error) {
	seq.terminateOnce.Do(func() {
		seq.Stream.Close(reason, err)
		seq.closed.Store(true)
	})
}

// terminateAll terminates each sequence
func (e *engine) terminateAll(active []*Sequence, reason string, err error) []*Sequence {
	for _, seq := range active {
		e.terminate(seq, reason, err)
	}
	return active[:0]
}
