use std::sync::{Arc, Mutex};

struct TensorData {
    data: f64,
    grad: f64,
}

pub struct Tensor {
    inner: Arc<Mutex<TensorData>>,
}

impl Tensor {
    pub fn new(data: f64) -> Self {
        Tensor {
            inner: Arc::new(Mutex::new(TensorData { data, grad: 0.0 })),
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.lock().unwrap().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.lock().unwrap().grad
    }
}
