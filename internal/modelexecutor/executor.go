package modelexecutor

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/fqadri/blizzard/internal/llmengine"
)

var errWorkerDown = errors.New("modelexecutor: worker is not running")

// stepRequest is one request sent to the worker. A sequence carries its prompt only on its
// first step; the worker holds state for it afterwards, and frees state for any id absent
// from seqs.
//
//	{"step":42,"seqs":[{"id":1,"prompt":"Explain gravity"},{"id":2}]}
type stepRequest struct {
	Step uint64    `json:"step"` // the current step number of the sequence batch. helps in validating desynced replies
	Seqs []wireSeq `json:"seqs"`
}

// Prompt is present only on the sequence's first step. A request carrying a prompt makes the
// worker (re)prefill that sequence from scratch, discarding any state it already held for it.
//
//	{"id":1,"prompt":"Explain gravity"}   first appearance
//	{"id":2}                              already cached
type wireSeq struct {
	ID     uint64 `json:"id"`
	Prompt string `json:"prompt,omitempty"`
}

// stepResponse is the worker's reply. Step echoes the request so a desynced transport is
// detectable; Error reports a step the worker could not run.
//
//	{"step":42,"results":[{"id":1,"token":"Hi","done":false},{"id":2,"token":".","done":true}]}
//	{"step":42,"results":[],"error":"cuda out of memory"}
type stepResponse struct {
	Step    uint64       `json:"step"` // echoes the request step. helps in validating desynced replies
	Results []wireResult `json:"results"`
	Error   string       `json:"error,omitempty"`
}

// Token is already detokenized, so the tokenizer stays entirely on the worker side.
//
//	{"id":1,"token":"Hi","done":false}   keep generating
//	{"id":2,"token":".","done":true}     model emitted a stop token
type wireResult struct {
	ID    uint64 `json:"id"`
	Token string `json:"token"`
	Done  bool   `json:"done"`
}

// worker is any backend that answers a stepRequest with a stepResponse.
type worker interface {
	Call(ctx context.Context, req, resp any) error
	Running() bool
	Stop() error
}

// Executor is not safe for concurrent use; the engine calls it from a single goroutine.
type Executor struct {
	worker  worker
	timeout time.Duration
	step    uint64
	cached  map[uint64]struct{} // sequence IDs the model worker currently holds state for
}

func New(w worker, timeout time.Duration) *Executor {
	return &Executor{worker: w, timeout: timeout, cached: make(map[uint64]struct{})}
}

// ExecuteStep runs one forward pass over the batch and returns one result per input sequence.
// A sequence carries its prompt only on its first step. After that the worker holds token ids it generated,
// keyed by sequence ID, so later steps need only the id:
//
//	step 1:  {"id":1,"prompt":"Explain gravity"}   → "Grav"   (prefill)
//	step 2:  {"id":1}                              → "ity"    (decode from cache)
//	step 3:  {"id":1}                              → " is"    (decode from cache)
//
// The worker frees state for any id absent from the batch, so a sequence that finished or was cancelled needs no explicit removal.
//
// Sending the full text on every step instead would force the worker to re-tokenize it, and
// tokenizer merges can shift at the append boundary, so the re-encoded prefix may not match
// the token ids already in the cache. Carrying only the id sidesteps that entirely.
func (e *Executor) ExecuteStep(ctx context.Context, seqs []*llmengine.Sequence) ([]llmengine.StepResult, error) {
	if len(seqs) == 0 {
		return nil, nil
	}

	if !e.worker.Running() {
		// todo: something must restart the worker
		return nil, fmt.Errorf("modelexecutor: %w", errWorkerDown)
	}

	e.step++

	req := stepRequest{Step: e.step, Seqs: make([]wireSeq, 0, len(seqs))}
	currentSequences := make(map[uint64]struct{}, len(seqs))

	// append all sequences to the request, sending the prompt only for sequences not in the cache
	for _, seq := range seqs {
		ws := wireSeq{ID: seq.ID}
		if _, ok := e.cached[seq.ID]; !ok {
			ws.Prompt = seq.Prompt // send prompt only for first occurrence
		}
		currentSequences[seq.ID] = struct{}{}
		req.Seqs = append(req.Seqs, ws)
	}

	callCtx, cancel := context.WithTimeout(ctx, e.timeout)
	defer cancel()

	var resp stepResponse
	if err := e.worker.Call(callCtx, &req, &resp); err != nil {
		clear(e.cached) // Call stops the worker on failure, taking its cached state with it
		return nil, fmt.Errorf("modelexecutor: step %d: %w", e.step, err)
	}

	if resp.Step != e.step { // replies are off by one and cannot be resynchronised
		_ = e.worker.Stop() // todo: if worker is stopped then nobody today is restarting it
		clear(e.cached)
		return nil, fmt.Errorf("modelexecutor: desync: sent step %d, got %d", e.step, resp.Step)
	}

	if resp.Error != "" {
		clear(e.cached) // the worker's state for this step is unknown, so re-prefill everything
		return nil, fmt.Errorf("modelexecutor: worker: %s", resp.Error)
	}

	// llm engine, in next execution step, can drop sequences that are no longer active, so update the cached set accordingly
	e.cached = currentSequences

	// convert wire results to engine results
	results := make([]llmengine.StepResult, 0, len(resp.Results))
	for _, r := range resp.Results {
		results = append(results, llmengine.StepResult{ID: r.ID, Token: r.Token, Done: r.Done})
	}

	return results, nil
}
