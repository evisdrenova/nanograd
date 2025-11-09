use nanograd::tensor::Tensor;

fn main() {
    let t = Tensor::new(5.0);
    println!("data: {}", t.data());
    println!("grad: {}", t.grad());
}
