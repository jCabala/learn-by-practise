use std::sync::{Arc, Condvar, Mutex};
use std::thread;

struct FifoQueueState {
    content: Vec<i32>,
    head: usize,
    tail: usize,
    count: usize,
}

struct FifoQueue {
    state: Mutex<FifoQueueState>,
    condition: Condvar,
}

impl FifoQueueState {
    fn new(capacity: usize) -> Self {
        let mut result = FifoQueueState {
            content: Vec::with_capacity(capacity),
            head: 0,
            tail: 0,
            count: 0,
        };
        result.content.resize(capacity, 0);
        return result;
    }
}

impl FifoQueue {
    fn new(capacity: usize) -> Self {
        FifoQueue {
            state: Mutex::new(FifoQueueState::new(capacity)),
            condition: Condvar::new(),
        }
    }

    fn enq(&self, data: i32) {
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

    fn deq(&self) -> i32 {
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
    let shared_queue = Arc::new(FifoQueue::new(10));

    let producer_shared_queue = shared_queue.clone();
    let consumer_shared_queue = shared_queue.clone();

    let producer = thread::spawn(move || {
        for i in 1..1000 {
            producer_shared_queue.enq(i);
        }
    });

    let consumer = thread::spawn(move || {
        let mut result = 0;
        for _ in 1..1000 {
            result += consumer_shared_queue.deq();
        }
        println!("{}", result);
    });

    producer.join().unwrap();
    consumer.join().unwrap();
}
