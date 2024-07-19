use micrograd_rs::value::{Value};

#[cfg(test)]
mod value_tests {

    use super::*;

    #[test]
    fn test_mul() {
        // Your test code here
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let x = Value::mul(a, b);

        assert_eq!(x.get().borrow().data, 2.0);
    }

    #[test]
    fn test_mul_operator() {

        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let x = a * b;

        assert_eq!(x.get().borrow().data, 2.0);
    }

    #[test]
    fn test_add() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let x = Value::add(a, b);

        assert_eq!(x.get().borrow().data, 3.0);
    }

    #[test]
    fn test_add_operator() {
        let a = Value::new(1.0);
        let b = Value::new(2.0);
        let x = a + b;

        assert_eq!(x.get().borrow().data, 3.0);
    }

    #[test]
    fn test_sub() {
        let a = Value::new(5.0);
        let b = Value::new(2.0);
        let x = Value::sub(a, b);

        assert_eq!(x.get().borrow().data, 3.0);
    }

    #[test]
    fn test_sub_operator() {
        let a = Value::new(5.0);
        let b = Value::new(2.0);
        let x = a - b;

        assert_eq!(x.get().borrow().data, 3.0);
    }

    #[test]
    fn test_div() {
        let a = Value::new(6.0);
        let b = Value::new(2.0);
        let x = Value::div(a, b);

        assert_eq!(x.get().borrow().data, 3.0);
    }

    #[test]
    fn test_div_operator() {
        let a = Value::new(6.0);
        let b = Value::new(2.0);
        let x = a / b;

        assert_eq!(x.get().borrow().data, 3.0);
    }

    #[test]
    fn test_complex_expression() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = Value::new(4.0);
        let d = Value::new(5.0);

        // (2 + 3) * (4 - 5) / 2
        let x = (a + b) * (c - d) / Value::new(2.0);

        assert_eq!(x.get().borrow().data, -2.5);
    }

    #[test]
    fn test_tanh() {
        let a = Value::new(0.5);
        let x = Value::tanh(a);

        let expected = 0.5_f32.tanh();
        assert!((x.get().borrow().data - expected).abs() < 1e-6);
    }

    #[test]
    fn test_exp() {
        let a = Value::new(1.0);
        let x = Value::exp(a);
        
        let expected = 1.0_f32.exp();
        assert!((x.get().borrow().data - expected).abs() < 1e-5);
    }

    #[test]
    fn test_pow() {
        let a = Value::new(2.0);
        let x = Value::pow(a, 3.0);
        
        assert_eq!(x.get().borrow().data, 8.0);
    }

    #[test]
    fn test_backward() {
        let a = Value::new(2.0);
        let b = Value::new(3.0);
        let c = a.clone() * b.clone();
        Value::back_propagate(&c);
        
        assert_eq!(a.get().borrow().grad, 3.0);
        assert_eq!(b.get().borrow().grad, 2.0);
    }

    #[test]
    fn test_mixed_operations() {
        let a = Value::new(0.5);
        let b = Value::new(0.3);
        let c = Value::new(0.2);
        let x = Value::tanh(a * b + c);
        
        let expected = (0.5_f32 * 0.3 + 0.2).tanh();
        assert!((x.get().borrow().data - expected).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_operations() {
        let a = Value::new(2.0);
        let x = 3 + a * 4.0 - 2;
        
        assert_eq!(x.get().borrow().data, 9.0);
    }
}
