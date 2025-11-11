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
        // return all parameters (weights + bias) for training
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(num_inputs: usize, num_outputs: usize) -> Self {
        let neurons = (0..num_outputs).map(|_| Neuron::new(num_inputs)).collect();

        Layer { neurons }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(inputs))
            .collect()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

impl Clone for Neuron {
    fn clone(&self) -> Neuron {
        Neuron {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}

#[derive(Clone)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(num_inputs: usize, layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        let mut input_size = num_inputs;

        for &output_size in layer_sizes {
            layers.push(Layer::new(input_size, output_size));
            input_size = output_size; // Output of this layer = input to next
        }

        MLP { layers }
    }

    pub fn forward(&self, inputs: &[Tensor]) -> Vec<Tensor> {
        let mut current = inputs.to_vec();

        for layer in &self.layers {
            current = layer.forward(&current);
        }

        current
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
