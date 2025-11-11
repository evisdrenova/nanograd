use nanograd::neuron::Neuron;
use nanograd::tensor::Tensor;

fn main() {
    let neuron = Neuron::new(3); // 3 inputs
    let inputs = vec![Tensor::new(1.0), Tensor::new(2.0), Tensor::new(3.0)];

    let output = neuron.forward(&inputs);
    println!("output: {}", output.data());
}
