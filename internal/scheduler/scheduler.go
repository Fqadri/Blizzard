package scheduler

import (
	"container/list"
	"errors"
	"sync"
)

// sentinel error returned when the scheduler's queue is full
var ErrQueueFull = errors.New("scheduler: queue is full")

type Scheduler[T comparable] struct {
	mutex        sync.Mutex
	pendingQueue *list.List
	index        map[T]*list.Element
	capacity     int
	ready        chan struct{} //signal channel when new items are added to the queue
}

func New[T comparable](capacity int) *Scheduler[T] {
	if capacity <= 0 || capacity > 200 { //todo: make the upper limit configurable
		capacity = 100
	}

	return &Scheduler[T]{
		capacity:     capacity,
		pendingQueue: list.New(),
		index:        make(map[T]*list.Element, capacity),
		ready:        make(chan struct{}, 1), // 1 only to allow signal
	}
}

// ready returns a channel that is signaled whenever new items are added to the scheduler's queue.
func (s *Scheduler[T]) Ready() <-chan struct{} {
	return s.ready
}

// Submit adds a new item to the scheduler's queue.
// Submit returns:
//   - ErrQueueFull if the scheduler has reached capacity.
func (s *Scheduler[T]) Submit(value T) error {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if s.pendingQueue.Len() >= s.capacity {
		return ErrQueueFull
	}

	elem := s.pendingQueue.PushBack(value)
	s.index[value] = elem

	// signal that new items are available in the queue
	select {
	case s.ready <- struct{}{}:
	default:
	}

	return nil
}

// NextBatch retrieves up to 'max' items from the scheduler's queue in FIFO order.
// NextBatch returns:
//   - a slice of up to 'max' items from the scheduler's queue.
//   - nil if no items are available.
func (s *Scheduler[T]) NextBatch(max int) ([]T, error) {
	if max <= 0 {
		return nil, nil
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()

	limit := min(max, s.pendingQueue.Len())
	if limit == 0 {
		return nil, nil // no items available in the queue
	}

	batch := make([]T, 0, limit)

	// remove items from the front of the queue and add them to the batch
	for range limit {
		elem := s.pendingQueue.Front()
		v := elem.Value.(T)

		s.pendingQueue.Remove(elem)
		delete(s.index, v)

		batch = append(batch, v)

		// re-arm: ready is buffered 1, so bursty Submits drop signals the engine still needs
		if s.pendingQueue.Len() > 0 {
			select {
			case s.ready <- struct{}{}:
			default:
			}
		}
	}

	return batch, nil
}

func (s *Scheduler[T]) Remove(value T) {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if elem, ok := s.index[value]; ok {
		s.pendingQueue.Remove(elem)
		delete(s.index, value)
	}
}
