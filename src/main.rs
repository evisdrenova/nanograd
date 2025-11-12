use nanograd::neuron::MLP;
use nanograd::tensor::Tensor;

fn main() {
    let mlp = MLP::new(2, &[8, 1]);

    // let mlp = MLP::new(3, &[4, 4, 1]); // 3 → 4 → 4 → 1

    // let inputs = vec![Tensor::new(1.0), Tensor::new(2.0), Tensor::new(3.0)];

    // let output = mlp.forward(&inputs);

    // println!("output: {}", output[0].data());
    // assert_eq!(output.len(), 1); // Should have 1 output

    // Simple dataset: XOR-like problem
    // Input: [x1, x2], Target: x1 XOR x2
    let dataset = vec![
        (vec![0.0, 0.0], 0.0),
        (vec![0.0, 1.0], 1.0),
        (vec![1.0, 0.0], 1.0),
        (vec![1.0, 1.0], 0.0),
    ];

    let tensor_dataset: Vec<(Vec<Tensor>, Tensor)> = dataset
        .iter()
        .map(|(inputs, target)| {
            let tensor_inputs = inputs.iter().map(|&x| Tensor::new(x)).collect();

            let tensor_target = Tensor::new(*target);

            (tensor_inputs, tensor_target)
        })
        .collect();

    let learning_rate = 0.05;
    let epochs = 1000;

    for epoch in 0..epochs {
        // zero out gradients in between epoch runs
        for j in mlp.parameters().iter() {
            j.zero_grad();
        }

        // initialize loss
        let mut total_loss = 0.0;

        for (input, target) in &tensor_dataset {
            // call forward pass
            let prediction = mlp.forward(input);

            // calc loss
            let loss = prediction[0].sub(target).pow(2.0);

            loss.backward();

            total_loss += loss.data();
        }

        // update weights
        for param in mlp.parameters().iter() {
            param.update(learning_rate);
        }

        if epoch % 50 == 0 {
            let params = mlp.parameters();
            let grad_sum: f64 = params.iter().map(|p| p.grad().abs()).sum();
            println!(
                "Epoch {}: Loss = {:.4}, Grad sum = {:.6}",
                epoch, total_loss, grad_sum
            );
        }
    }

    println!("\nTesting trained model:");
    for (inputs, target) in &tensor_dataset {
        let prediction = mlp.forward(inputs);
        println!(
            "Input: [{:.1}, {:.1}] -> Prediction: {:.4}, Target: {:.1}",
            inputs[0].data(),
            inputs[1].data(),
            prediction[0].data(),
            target.data()
        );
    }
}
