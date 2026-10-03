package stream

import (
	"context"
)

type Stream interface {
	Recv(ctx context.Context) (string, error)
	Send(msg string) error

	// Close marks the stream finished; a non-nil err makes Recv return it instead of io.EOF.
	Close(reason string, err error)

	// Reason is only meaningful once Recv has returned an error.
	Reason() string
}
