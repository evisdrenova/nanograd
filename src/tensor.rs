use std::sync::{Arc, Mutex};

type BackwardFn = Box<dyn Fn() + Send>;

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
}
