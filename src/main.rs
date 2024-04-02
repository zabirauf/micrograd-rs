mod value;
mod neuron;

use value::{Value, RefValue};

use crate::neuron::MultiLayerPerceptron;

fn main() {
    let xs = vec![
        vec![Value::new(2.0), Value::new(3.0), Value::new(-1.0)],
        vec![Value::new(3.0), Value::new(-1.0), Value::new(0.5)],
        vec![Value::new(0.5), Value::new(1.0), Value::new(1.0)],
        vec![Value::new(1.0), Value::new(1.0), Value::new(-1.0)],
    ];
    let ys = vec![Value::new(1.0), Value::new(-1.0), Value::new(-1.0), Value::new(1.0)];
    let mlp = MultiLayerPerceptron::new(3, vec![4,4,1]);

    let forward_and_print = |inputs: Vec<Vec<RefValue>>| {
        let mut combined_ypred = Vec::new();
        for input in inputs {
            let ypred = mlp.forward(&input).iter().map(|val| val.get().borrow().data.to_string()).collect::<Vec<String>>();
            combined_ypred.push(ypred.join(", "));
        }
        println!("Pre training forward: [{}]", combined_ypred.join(", "));
    };
    forward_and_print(xs.clone());

    mlp.train(0.01, 50, xs.clone(), ys);
    forward_and_print(xs.clone());
}
