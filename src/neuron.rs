use std::ops::Add;

use rand::Rng;

use crate::tensor::Tensor;

pub struct Neuron {
    weights: Vec<Tensor>,
    bias: Tensor,
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Self {
        let mut rng = rand::rng();

        // Initialize weights randomly between -1 and 1
        let weights = (0..num_inputs)
            .map(|_| Tensor::new(rng.random_range(-1.0..1.0)))
            .collect();

        // Initialize bias to 0
        let bias = Tensor::new(0.0);

        Neuron { weights, bias }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Tensor {
        let mut sum = Tensor::new(0.0);

        for i in 0..self.weights.len() {
            sum = sum.add(&self.weights[i].mul(&inputs[i]));
        }

        sum = sum.add(&self.bias);
        sum.relu()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        // Return all parameters (weights + bias) for training
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}
