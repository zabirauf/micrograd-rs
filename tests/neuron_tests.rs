use micrograd_rs::value::Value;
use micrograd_rs::neuron::{Neuron, MultiLayerPerceptron, NetworkParameters};
use log::debug;
use env_logger;

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
        let ys = vec![Value::new(0.0), Value::new(1.0), Value::new(1.0), Value::new(0.0)];
        
        // Train for a few iterations
        mlp.train(0.1, 400, xs.clone(), ys.clone());
        
        // Check if the network has learned something
        let outputs: Vec<f32> = xs.iter().map(|x| mlp.forward(x)[0].get().borrow().data).collect();
        debug!("Outputs after training: {:?}", outputs);
        
        // Check if outputs are close to expected values within a certain error range
        let error_margin = 0.45; // Adjust this value based on desired accuracy
        for (output, expected) in outputs.iter().zip(ys.iter()) {
            let expected_value = expected.get().borrow().data;
            assert!((output - expected_value).abs() < error_margin,
                    "Output {:.2} not within {:.2} of expected {:.2}", output, error_margin, expected_value);
        }
    }

}