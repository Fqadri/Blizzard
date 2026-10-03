package stream

import (
	"context"
	"errors"
	"io"
	"sync"
)

var (
	ErrStreamClosed = errors.New("stream closed")
	ErrStreamFull   = errors.New("stream full")
)

var _ Stream = (*channelStream)(nil) // ensure channelStream implements the Stream interface

// channelStream is an implementation of the Stream interface that uses a Go channel to deliver messages.
type channelStream struct {
	ch        chan string   // actual data channel
	closed    chan struct{} // signal channel that means the stream has been closed
	closeOnce sync.Once     // ensures close on channel is performed only once

	reason string
	err    error
}

func NewChannelStream() Stream {
	return &channelStream{
		ch:     make(chan string, 100), // this is so that the sender can send without blocking which would otherwise block the sender if the receiver is slow
		closed: make(chan struct{}),
	}
}

func (c *channelStream) Recv(ctx context.Context) (string, error) {
	// drain buffered tokens before honoring the close signal
	select {
	case msg := <-c.ch:
		return msg, nil
	default:
	}

	select {
	case msg := <-c.ch:
		return msg, nil
	case <-c.closed:
		select { // close may have raced a final Send, so buffered data still wins over the terminal state
		case msg := <-c.ch:
			return msg, nil
		default:
			err := io.EOF
			if c.err != nil {
				err = c.err
			}
			return "", err
		}
	case <-ctx.Done():
		return "", ctx.Err()
	}
}

// Send attempts to send a message into the stream. closing the stream and coordinating with the Send is callers responsibility.
// returns
// 1. nil if the message was successfully sent
// 2. ErrStreamClosed if the stream has been closed
// 3. ErrStreamFull if the stream's buffer is full. This can happen if the receiver is too slow to consume messages.
func (c *channelStream) Send(msg string) error {
	select {
	case <-c.closed:
		return ErrStreamClosed
	default:
	}

	select {
	case c.ch <- msg: // if buffer got full, this will fall to default below to avoid head-of-line blocking
		return nil
	default:
		return ErrStreamFull
	}
}

// Close records the terminal reason and error.
// It signals both recv and send methods
// It is safe to call Close multiple times; the first call wins and subsequent ones have no effect
func (c *channelStream) Close(reason string, err error) {
	c.closeOnce.Do(func() {
		c.reason = reason
		c.err = err
		close(c.closed) // close signal channel. <-c.closed now will return the signal.
	})
}

// Reason returns the reason why the stream was closed. If the stream is not yet closed, it returns an empty string.
func (c *channelStream) Reason() string {
	select {
	case <-c.closed:
		return c.reason
	default:
		return ""
	}
}
