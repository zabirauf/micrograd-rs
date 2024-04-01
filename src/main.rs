mod value;
mod neuron;

use value::Value;

use crate::neuron::{Layer, MultiLayerPerceptron, Neuron};

fn main() {
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);

    // Weights w1, w2
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);

    // Bias of the neuron
    let b = Value::new(6.8813735870195432);

    // Computation
    let x1w1 = Value::mul(x1, w1);
    let x2w2 = Value::mul(x2, w2);
    let x1w1x2w2 = Value::add(x1w1, x2w2);
    let n = Value::add(x1w1x2w2, b);
    let o = Value::tanh(n);

    println!("O is {}", o);

    Value::back_propagate(&o);

    println!("O is {}", o);

    // println!("-----------");

    // // ----------

    // // Inputs x1, x2
    // let x1 = Value::new(2.0);
    // let x2 = Value::new(0.0);

    // // Weights w1, w2
    // let w1 = Value::new(-3.0);
    // let w2 = Value::new(1.0);

    // // Bias of the neuron
    // let b = Value::new(6.8813735870195432);

    // // Computation
    // let x1w1 = x1 * w1;
    // let x2w2 = x2 * w2;
    // let x1w1x2w2 = x1w1 + x2w2;
    // let n = x1w1x2w2 + b;
    // let e = Value::exp(2 * n);
    // let o = (e.clone() - 1.0) / (e.clone() + 1);

    // println!("O is {}", o);

    // Value::back_propagate(&o);

    // println!("O is {}", o);

    // println!("-----------");
    // // ----------

    // let a = Value::new(3.0);
    // let b = 1 + a.clone();
    // Value::back_propagate(&b);

    // println!("A is {}", a);

    // let c = Value::new(2.0);
    // let c = Value::exp(c);

    // let d = Value::new(1.0) + c;
    // Value::back_propagate(&d);

    // println!("D is {}", d.clone());

    // let e = Value::pow(Value::new(2.0), 3.0);
    // Value::back_propagate(&e);
    // println!("E is {}", e);

    // let f = Value::new(12.0);
    // let g = Value::new(3.0);

    // let h = f / g;
    // Value::back_propagate(&h);
    // println!("H is {}", h);

    // let x = vec![Value::new(2.0), Value::new(3.0)];
    // let n = Neuron::new(x.len());
    // let o = n.forward(&x);
    // println!("Neuron O is {}", o);
    // println!();

    // let n = Layer::new(2, 3);
    // let o = n.forward(&x);

    // println!("Layer O are");
    // for neuron in o.iter() {
    //     println!("--- {}", neuron);
    // }
    // println!();

    // let x = vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)];
    // let mlp = MultiLayerPerceptron::new(x.len(), vec![4,4,1]);
    // let o = mlp.forward(&x);

    // println!("MLP O are");
    // for neuron in o.iter() {
    //     println!("--- {}", neuron);
    // }
    // println!();

    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let ys = vec![Value::new(1.0), Value::new(-1.0), Value::new(-1.0), Value::new(1.0)];
    let mlp = MultiLayerPerceptron::new(3, vec![4,4,1]);

    println!("{}", mlp);

    mlp.train(0.01, 50, xs, ys);

    println!("{}", mlp);
}
