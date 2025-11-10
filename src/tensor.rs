use std::sync::{Arc, Mutex};

type BackwardFn = Box<dyn Fn(f64) + Send>;

struct TensorData {
    data: f64,
    grad: f64,
    prev: Vec<Tensor>,
    backward_fn: Option<BackwardFn>,
}

pub struct Tensor {
    inner: Arc<Mutex<TensorData>>,
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Tensor {
            inner: Arc::new(Mutex::new(TensorData {
                data,
                grad: 0.0,
                prev: Vec::new(),
                backward_fn: None,
            })),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.lock().unwrap().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.lock().unwrap().grad
    }

    fn from_op(data: f64, prev: Vec<Tensor>, backward_fn: BackwardFn) -> Self {
        Tensor {
            inner: Arc::new(Mutex::new(TensorData {
                data,
                grad: 0.0,
                prev,
                backward_fn: Some(backward_fn),
            })),
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        // compute forward pass data operation
        let data = self.data() + other.data();

        // create clones of the data that we can hand off to our backward_fn
        let self_clone = self.clone();
        let other_clone = other.clone();

        // define the backward func and box it on the heap to run later
        let backward_fn = Box::new(move |grad_output: f64| {
            // use += to accumulate bc we might run this multiple times
            self_clone.inner.lock().unwrap().grad += grad_output * 1.0;
            other_clone.inner.lock().unwrap().grad += grad_output * 1.0;
        });

        let prev = vec![self.clone(), other.clone()];

        Tensor::from_op(data, prev, backward_fn)
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        // compute forward pass data operation
        let data = self.data() * other.data();

        // create clones of the data that we can hand off to our backward_fn
        let self_clone = self.clone();
        let other_clone = other.clone();

        // define the backward func and box it on the heap to run later
        let backward_fn = Box::new(move |grad_output: f64| {
            // use += to accumulate bc we might run this multiple times
            self_clone.inner.lock().unwrap().grad += grad_output * other_clone.data();
            other_clone.inner.lock().unwrap().grad += grad_output * self_clone.data();
        });

        let prev = vec![self.clone(), other.clone()];

        Tensor::from_op(data, prev, backward_fn)
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            inner: Arc::clone(&self.inner),
        }
    }
}
