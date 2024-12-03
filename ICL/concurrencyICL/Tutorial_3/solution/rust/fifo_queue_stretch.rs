use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// We require the generic argument T to implement the Copy trait so that we can initialise
// `content` with a default value of type T -- we will need to be able to copy in the default
// value.
struct FifoQueueState<T: Copy> {
    content: Vec<T>,
    head: usize,
    tail: usize,
    count: usize,
}

struct FifoQueue<T: Copy> {
    state: Mutex<FifoQueueState<T>>,
    condition: Condvar,
}

impl<T: Copy> FifoQueueState<T> {
    // The `default_element_provider` function provides a default value with which to initialise
    // the vector.
    fn new(capacity: usize, default_element_provider: fn() -> T) -> Self {
        let mut result = FifoQueueState {
            content: Vec::with_capacity(capacity),
            head: 0,
            tail: 0,
            count: 0,
        };
        // Initialise the vector so that every element has a default value.
        result.content.resize(capacity, default_element_provider());
        return result;
    }
}

impl<T: Copy> FifoQueue<T> {
    fn new(capacity: usize, default_element_provider: fn() -> T) -> Self {
        FifoQueue {
            state: Mutex::new(FifoQueueState::new(capacity, default_element_provider)),
            condition: Condvar::new(),
        }
    }

    fn enq(&self, data: T) {
        let mut state = self.state.lock().unwrap();
        while state.count >= state.content.len() {
            state = self.condition.wait(state).unwrap();
        }
        let tail = state.tail;
        state.content[tail] = data;
        state.tail = (state.tail + 1) % state.content.len();
        state.count += 1;
        self.condition.notify_all();
    }

    fn deq(&self) -> T {
        let mut state = self.state.lock().unwrap();
        while state.count == 0 {
            state = self.condition.wait(state).unwrap();
        }
        let head = state.head;
        let result = state.content[head];
        state.head = (state.head + 1) % state.content.len();
        state.count -= 1;
        self.condition.notify_all();
        return result;
    }
}

pub fn main() {
    let p2c = Arc::new(FifoQueue::new(10, || {
        return 0;
    }));
    let c2p = Arc::new(FifoQueue::new(10, || {
        return 0;
    }));

    let producer_p2c = p2c.clone();
    let producer_c2p = c2p.clone();

    let num_consumers = 8;

    let producer = thread::spawn(move || {
        for _ in 0..num_consumers * 1000 {
            producer_p2c.enq(1);
        }
        let mut result = 0;
        for _ in 0..num_consumers {
            result += producer_c2p.deq();
        }
        println!("Result: {}", result);
    });

    let mut consumers = Vec::new();
    for _ in 0..num_consumers {
        let consumer_p2c = p2c.clone();
        let consumer_c2p = c2p.clone();
        consumers.push(thread::spawn(move || {
            let mut result = 0;
            for _ in 0..1000 {
                result += consumer_p2c.deq();
            }
            consumer_c2p.enq(result);
        }));
    }

    producer.join().unwrap();
    for consumer in consumers {
        consumer.join().unwrap();
    }
}
