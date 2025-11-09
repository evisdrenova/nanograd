use std::sync::{Arc, Mutex};

struct TensorData {
    data: f64,
    grad: f64,
}

pub struct Tensor {
    inner: Arc<Mutex<TensorData>>,
}

impl Tensor {
    pub fn new(&self) -> Self {
        //TODO: create a new Tensor with grad and initialize it to 0
    }

    pub fn data(&self) -> f64 {
        // TODO: return the data
    }

    pub fn grad(&self) -> f64 {
        //TODO: return the grad
    }
}
