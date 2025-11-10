use nanograd::tensor::Tensor;

fn main() {
    let a = Tensor::new(3.0);
    let b = a.pow(2.0);
    println!("data: {}", b.data());
}
