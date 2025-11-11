use nanograd::neuron::{MLP, Neuron};
use nanograd::tensor::Tensor;

fn main() {
    let mlp = MLP::new(3, &[4, 4, 1]); // 3 → 4 → 4 → 1

    let inputs = vec![Tensor::new(1.0), Tensor::new(2.0), Tensor::new(3.0)];

    let output = mlp.forward(&inputs);

    println!("output: {}", output[0].data());
    assert_eq!(output.len(), 1);
}
