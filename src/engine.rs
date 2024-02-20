use std::fmt;
use std::ops;

#[derive(Clone)]
pub struct Value {
    data: f32,
    children: Vec<Box<Value>>,
    op: Option<char>
}

impl Value {
    pub fn default() -> Value {
        Value {
            data: 0.0,
            children: vec![],
            op: None
        }
    }

    pub fn new(data: f32) -> Value {
        Value {
            data: data,
            ..Value::default()
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={:.2}, op={}, children={})", self.data, self.op.unwrap_or('?'), self.children.len())
    }
}

impl ops::Add for Value {
    type Output = Self;

    fn add(self, rhs: Value) -> Value {
        Value {
            data: self.data + rhs.data,
            op: Some('+'),
            children: vec![Box::new(self), Box::new(rhs)],
            ..Value::default()
        }
    }
}

impl ops::Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            op: Some('*'),
            children: vec![Box::new(self), Box::new(rhs)],
            ..Value::default()
        }
    }
}