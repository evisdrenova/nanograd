use nanograd::tensor::Tensor;

fn main() {
    let a = Tensor::new(2.0);
    let b = Tensor::new(3.0);
    let c = a.add(&b);
    let d = a.data() + b.data();
    println!("data: {}", c.data());
    println!("data: {}", d);
}
