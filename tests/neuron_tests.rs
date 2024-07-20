use env_logger;
use log::debug;
use micrograd_rs::neuron::{MultiLayerPerceptron, NetworkParameters, Neuron};
use micrograd_rs::value::{Value, RefValue};
use rand::Rng;
use rand::seq::SliceRandom;

#[cfg(test)]
mod neuron_tests {
    use super::*;

    #[ctor::ctor]
    fn init() {
        env_logger::init();
    }

    #[test]
    fn test_neuron_creation() {
        let neuron = Neuron::new(3);
        assert_eq!(neuron.weights.len(), 3);
        assert!(neuron.bias.get().borrow().data >= -1.0 && neuron.bias.get().borrow().data <= 1.0);
    }

    #[test]
    fn test_neuron_forward() {
        let neuron = Neuron::new(2);
        let input = vec![Value::new(1.0), Value::new(2.0)];
        let output = neuron.forward(&input);
        assert!(output.get().borrow().data >= -1.0 && output.get().borrow().data <= 1.0);
    }

    #[test]
    fn test_neuron_parameters() {
        let neuron = Neuron::new(2);
        let params = neuron.parameters();
        assert_eq!(params.len(), 3); // 2 weights + 1 bias
    }

    #[test]
    fn test_mlp_creation() {
        let mlp = MultiLayerPerceptron::new(2, vec![3, 1]);
        assert_eq!(mlp.layers.len(), 2);
        assert_eq!(mlp.layers[0].neurons.len(), 3);
        assert_eq!(mlp.layers[1].neurons.len(), 1);
    }

    #[test]
    fn test_mlp_forward() {
        let mlp = MultiLayerPerceptron::new(2, vec![3, 1]);
        let input = vec![Value::new(1.0), Value::new(2.0)];
        let output = mlp.forward(&input);
        assert_eq!(output.len(), 1);
        assert!(output[0].get().borrow().data >= -1.0 && output[0].get().borrow().data <= 1.0);
    }

    #[test]
    fn test_mlp_parameters() {
        let mlp = MultiLayerPerceptron::new(2, vec![3, 1]);
        let params = mlp.parameters();
        assert_eq!(params.len(), 13); // (2*3 weights + 3 biases) + (3*1 weights + 1 bias) = 9 + 4 = 13
    }

    #[test]
    fn test_mlp_train() {
        let mlp = MultiLayerPerceptron::new(2, vec![3, 1]);
        let xs = vec![
            vec![Value::new(0.0), Value::new(0.0)],
            vec![Value::new(0.0), Value::new(1.0)],
            vec![Value::new(1.0), Value::new(0.0)],
            vec![Value::new(1.0), Value::new(1.0)],
        ];

        // XOR operator
        let ys = vec![
            Value::new(0.0),
            Value::new(1.0),
            Value::new(1.0),
            Value::new(0.0),
        ];

        // Train for a few iterations
        mlp.train(0.1, 400, xs.clone(), ys.clone());

        // Check if the network has learned something
        let outputs: Vec<f32> = xs
            .iter()
            .map(|x| mlp.forward(x)[0].get().borrow().data)
            .collect();
        debug!("Outputs after training: {:?}", outputs);

        // Check if outputs are close to expected values within a certain error range
        let error_margin = 0.45; // Adjust this value based on desired accuracy
        for (output, expected) in outputs.iter().zip(ys.iter()) {
            let expected_value = expected.get().borrow().data;
            assert!(
                (output - expected_value).abs() < error_margin,
                "Output {:.2} not within {:.2} of expected {:.2}",
                output,
                error_margin,
                expected_value
            );
        }
    }

    #[test]
    fn test_binary_classification() {
        // Generate synthetic data for binary classification
        let mut rng = rand::thread_rng();
        let num_samples = 100;
        let mut x = Vec::with_capacity(num_samples);
        let mut y = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let x1 = rng.gen_range(-1.0..1.0);
            let x2 = rng.gen_range(-1.0..1.0);
            x.push(vec![Value::new(x1), Value::new(x2)]);

            // Simple decision boundary: y = 1 if x1 + x2 > 0, else -1
            let label = if x1 + x2 > 0.0 { 1.0 } else { -1.0 };
            y.push(Value::new(label));
        }

        // Create model
        let model = MultiLayerPerceptron::new(2, vec![5, 1]);

        // Define loss function
        fn loss(
            model: &MultiLayerPerceptron,
            x: &Vec<Vec<RefValue>>,
            y: &Vec<RefValue>,
            batch_size: Option<usize>,
        ) -> (RefValue, f32) {
            let (xb, yb) = if let Some(size) = batch_size {
                let mut indices: Vec<usize> = (0..x.len()).collect();
                indices.shuffle(&mut rand::thread_rng());
                let indices = &indices[0..size];
                (
                    indices.iter().map(|&i| x[i].clone()).collect(),
                    indices.iter().map(|&i| y[i].clone()).collect(),
                )
            } else {
                (x.clone(), y.clone())
            };

            let scores: Vec<RefValue> = xb
                .iter()
                .map(|input| model.forward(input)[0].clone())
                .collect();

            let losses: Vec<RefValue> = yb
                .iter()
                .zip(scores.iter())
                .map(|(yi, scorei)| Value::relu(Value::new(1.0) + (-1.0 * yi.clone() * scorei.clone())))
                .collect();

            let data_loss = losses
                .iter()
                .fold(Value::new(0.0), |acc, loss| acc + loss.clone())
                / Value::new(losses.len() as f32);

            let alpha = Value::new(1e-4);
            let reg_loss = model
                .parameters()
                .iter()
                .fold(Value::new(0.0), |acc, p| acc + p.clone() * p.clone())
                * alpha;

            let total_loss = data_loss + reg_loss;

            let accuracy = yb
                .iter()
                .zip(scores.iter())
                .map(|(yi, scorei)| {
                    (yi.get().borrow().data > 0.0) == (scorei.get().borrow().data > 0.0)
                })
                .filter(|&x| x)
                .count() as f32
                / yb.len() as f32;

            (total_loss, accuracy)
        }

        // Training loop
        let num_epochs = 100;
        let batch_size = 32;
        for epoch in 0..num_epochs {
            let (total_loss, acc) = loss(&model, &x, &y, Some(batch_size));

            Value::back_propagate(&total_loss);

            let learning_rate = 0.1;
            for p in model.parameters() {
                Value::backward(&p, learning_rate);
            }

            if epoch % 10 == 0 {
                println!(
                    "Epoch {} - Loss: {:.4}, Accuracy: {:.2}%",
                    epoch,
                    total_loss.get().borrow().data,
                    acc * 100.0
                );
            }
        }

        // Final evaluation
        let (final_loss, final_accuracy) = loss(&model, &x, &y, None);
        println!(
            "Final - Loss: {:.4}, Accuracy: {:.2}%",
            final_loss.get().borrow().data,
            final_accuracy * 100.0
        );

        // Assert that the final accuracy is above a certain threshold
        assert!(final_accuracy > 0.8, "Final accuracy should be above 80%");
    }
}
