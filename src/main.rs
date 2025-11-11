use nanograd::tensor::Tensor;

fn main() {
    let a = Tensor::new(3.0);
    let b = a.relu();
    b.backward();

    assert_eq!(b.data(), 3.0); // 3.0 stays 3.0
    assert_eq!(a.grad(), 1.0); // Gradient passes through

    let a = Tensor::new(-3.0);
    let b = a.relu();
    b.backward();

    assert_eq!(b.data(), 0.0); // -3.0 becomes 0.0
    assert_eq!(a.grad(), 0.0); // Gradient blocked
}
