use crate::value::{RefValue, Value};
use rand::Rng;
use std::fmt;

pub trait NetworkParameters {
    fn parameters(&self) -> Vec<RefValue>;
}

#[derive(Clone)]
pub struct Neuron {
    weights: Vec<RefValue>,
    bias: RefValue,
}

impl Neuron {
    pub fn new(len: usize) -> Neuron {
        let mut rng = rand::thread_rng();
        let mut n = Neuron {
            weights: Vec::<RefValue>::with_capacity(len),
            bias: Value::new(rng.gen_range(-1.0..1.0)),
        };

        for _ in 0..len {
            let v = Value::new(rng.gen_range(-1.0..1.0));
            n.weights.push(v);
        }

        n
    }

    pub fn forward(&self, x: &Vec<RefValue>) -> RefValue {
        let weighted_sum: RefValue = x
            .iter()
            .zip(self.weights.iter())
            .map(|(x_val, weight)| Value::mul(x_val.clone(), weight.clone()))
            .fold(self.bias.clone(), |acc, v| Value::add(v, acc));

        Value::tanh(weighted_sum)
    }
}

impl NetworkParameters for Neuron {
    fn parameters(&self) -> Vec<RefValue> {
        let mut params = vec![self.bias.clone()];
        params.extend_from_slice(&self.weights);

        params
    }
}

#[derive(Clone)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(len_in: usize, len_out: usize) -> Layer {
        let mut layer = Layer {
            neurons: Vec::with_capacity(len_out),
        };

        for _ in 0..len_out {
            layer.neurons.push(Neuron::new(len_in));
        }

        layer
    }

    pub fn forward(&self, x: &Vec<RefValue>) -> Vec<RefValue> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(x))
            .collect()
    }
}

impl NetworkParameters for Layer {
    fn parameters(&self) -> Vec<RefValue> {
        self.neurons
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

#[derive(Clone)]
pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    pub fn new(len_in: usize, len_outs: Vec<usize>) -> MultiLayerPerceptron {
        let mut mlp = MultiLayerPerceptron {
            layers: Vec::<Layer>::with_capacity(len_outs.len()),
        };

        let mut layer_sizes = vec![len_in];
        layer_sizes.extend_from_slice(&len_outs);

        for i in 0..(layer_sizes.len() - 1) {
            mlp.layers
                .push(Layer::new(layer_sizes[i], layer_sizes[i + 1]));
        }

        mlp
    }

    pub fn forward(&self, x: &Vec<RefValue>) -> Vec<RefValue> {
        let mut out = x.to_vec();
        for layer in self.layers.iter() {
            let res = layer.forward(&out);
            out = res
        }

        out
    }

    pub fn train(
        &self,
        learning_rate: f32,
        iterations: u32,
        xs: Vec<Vec<RefValue>>,
        ys: Vec<RefValue>,
    ) {
        //let mut loss: RefValue = Value::new(0.0);
        for iter in 0..iterations {
            let loss = xs
                .iter()
                .map(|x| self.forward(x))
                .map(|y| y.get(0).unwrap().clone())
                .zip(ys.iter())
                .fold(Value::new(0.0), |acc, (ypred, y)| {
                    // acc + (y-ypref)^2.0
                    Value::add(acc, Value::pow(Value::sub(y.clone(), ypred), 2.0))
                });
            let loss = Value::div(loss.clone(), Value::new(ys.len() as f32));

            Value::back_propagate(&loss);
            println!("Loss at iteration {}: {}", iter, loss);

            let params = self.parameters();

            for p in params {
                Value::backward(&p, learning_rate);
            }
        }
    }
}

impl NetworkParameters for MultiLayerPerceptron {
    fn parameters(&self) -> Vec<RefValue> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }
}
impl fmt::Display for MultiLayerPerceptron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MultiLayerPerceptron")?;
        for layer in self.layers.iter() {
            writeln!(f, "    {}", layer)?;
        }
        Ok(())
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Layer")?;
        for neuron in &self.neurons {
            writeln!(f, "        {}", neuron)?;
        }
        Ok(())
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Neuron")?;
        for param in self.parameters().iter() {
            writeln!(f, "            {}", param)?;
        }
        Ok(())
    }
}