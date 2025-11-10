use nanograd::tensor::Tensor;

fn main() {
    let a = Tensor::new(2.0);
    let b = Tensor::new(3.0);
    let c = a.mul(&b);
    println!("data: {}", c.data());
}
