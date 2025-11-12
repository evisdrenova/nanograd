use std::{
    collections::HashSet,
    sync::{Arc, Mutex},
};

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
        let data = self.data() * other.data();

        let self_clone = self.clone();
        let other_clone = other.clone();

        let backward_fn = Box::new(move |grad_output: f64| {
            self_clone.inner.lock().unwrap().grad += grad_output * other_clone.data();
            other_clone.inner.lock().unwrap().grad += grad_output * self_clone.data();
        });

        let prev = vec![self.clone(), other.clone()];

        Tensor::from_op(data, prev, backward_fn)
    }

    pub fn pow(&self, n: f64) -> Tensor {
        let data = self.data().powf(n);

        let self_clone = self.clone();

        let backward_fn = Box::new(move |grad_output: f64| {
            self_clone.inner.lock().unwrap().grad +=
                grad_output * n * self_clone.data().powf(n - 1.0);
        });

        let prev = vec![self.clone()];

        Tensor::from_op(data, prev, backward_fn)
    }

    fn build_reverse_top_order(&self) -> Vec<Tensor> {
        // the vec that we return
        let mut topo = Vec::new();
        // stores the nodes that we have already visited
        let mut visited = HashSet::new();

        fn build_reverse_top_order_recursive(
            tensor: &Tensor,
            topo: &mut Vec<Tensor>,
            visited: &mut HashSet<*const ()>,
        ) {
            // create a unique pointer for this tensor
            let ptr = Arc::as_ptr(&tensor.inner) as *const ();

            if visited.contains(&ptr) {
                return;
            }

            visited.insert(ptr);

            let prev_unwrapped = tensor.inner.lock().unwrap().prev.clone();

            for parent in prev_unwrapped.iter() {
                build_reverse_top_order_recursive(parent, topo, visited);
            }
            topo.push(tensor.clone());
        }
        build_reverse_top_order_recursive(self, &mut topo, &mut visited);
        topo
    }

    pub fn backward(&self) {
        let topo = self.build_reverse_top_order();

        self.inner.lock().unwrap().grad = 1.0;

        for tensor in topo.iter().rev() {
            let grad = tensor.grad();

            let should_call = tensor.inner.lock().unwrap().backward_fn.is_some();

            if should_call {
                if let Some(ref func) = tensor.inner.lock().unwrap().backward_fn {
                    func(grad);
                }
            }
        }
    }

    pub fn relu(&self) -> Tensor {
        let data = if self.data() > 0.0 { self.data() } else { 0.0 };

        let self_clone = self.clone();

        let backward_fn = Box::new(move |grad_output: f64| {
            let local_grad = if self_clone.data() > 0.0 { 1.0 } else { 0.0 };

            self_clone.inner.lock().unwrap().grad += grad_output * local_grad;
        });

        Tensor::from_op(data, vec![self.clone()], backward_fn)
    }

    pub fn zero_grad(&self) {
        self.inner.lock().unwrap().grad = 0.0;
    }

    pub fn update(&self, learning_rate: f64) {
        let mut locked = self.inner.lock().unwrap();
        locked.data -= learning_rate * locked.grad;
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        // a - b = a + (-1 × b)
        let neg_one = Tensor::new(-1.0);
        self.add(&other.mul(&neg_one))
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Tensor {
        Tensor {
            inner: Arc::clone(&self.inner),
        }
    }
}
