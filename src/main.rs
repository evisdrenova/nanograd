use nanograd::tensor::Tensor;

fn main() {
    let a = Tensor::new(2.0);
    let b = Tensor::new(3.0);
    let c = a.mul(&b); // c = 6.0
    let d = c.add(&a); // d = 8.0
    let e = d.pow(2.0); // e = 64.0

    e.backward();

    println!("e.data: {}", e.data());
    println!("e.grad: {}", e.grad());
    println!("d.grad: {}", d.grad());
    println!("c.grad: {}", c.grad());
    println!("a.grad: {}", a.grad());
    println!("b.grad: {}", b.grad());
}
