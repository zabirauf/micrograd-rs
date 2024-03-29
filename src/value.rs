use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::ops;
use std::rc::Rc;

#[derive(Clone)]
pub struct Value {
    data: f32,
    children: Vec<RefValue>,
    non_chained_deps: Option<[f32; 1]>,
    grad: f32,
    op: Option<&'static str>,
}

#[derive(Clone)]
pub struct RefValue(Rc<RefCell<Value>>);

impl Value {
    pub fn new(data: f32) -> RefValue {
        RefValue(Rc::new(RefCell::new(Value {
            data: data,
            children: vec![],
            non_chained_deps: None,
            grad: 0.0,
            op: None,
        })))
    }

    pub fn back_propagate(val: &RefValue) -> () {
        {
            let mut start_borrow = val.get().borrow_mut();
            start_borrow.grad = 1.0;
        }

        let mut topo = vec![];
        let mut visited = HashSet::new();

        Self::topological_sort(&val, &mut topo, &mut visited);

        while let Some(node) = topo.pop() {
            let (data, grad, children, non_chained_deps, op) = {
                let n = node.get().borrow();
                (n.data, n.grad, n.children.clone(), n.non_chained_deps, n.op)
            };

            for child in children.iter() {
                let mut child_borrow = child.get().borrow_mut();
                match op {
                    Some("+") => {
                        child_borrow.grad += 1.0 * grad; // For addition, gradient is passed directly
                    }
                    Some("*") => {
                        let other_child_data = if child == &children[0] {
                            children[1].get().borrow().data
                        } else {
                            children[0].get().borrow().data
                        };
                        child_borrow.grad += grad * other_child_data; // For multiplication, gradient is scaled by the other operand
                    },
                    Some("tanh") => {
                        child_borrow.grad += 1.0 - data.powi(2);
                    },
                    Some("exp") => {
                        //TODO: Seems like some bug here
                        child_borrow.grad += data * grad;
                    },
                    Some("pow") => {
                        let n = non_chained_deps.unwrap()[0];
                        child_borrow.grad += n * child_borrow.data.powf(n-1.0) * grad;
                    }
                    _ => {} // No operation or unsupported operation; no gradient update
                }

                println!("{}", child_borrow);
            }
        }
    }

    fn topological_sort(
        node: &RefValue,
        topo: &mut Vec<RefValue>,
        visited: &mut HashSet<RefValue>,
    ) {
        if visited.contains(node) {
            return;
        }

        visited.insert(node.clone());
        for child in node.get().borrow().children.iter() {
            Self::topological_sort(child, topo, visited);
        }

        topo.push(node.clone());
    }

    pub fn tanh(slf: RefValue) -> RefValue {
        let x = slf.get().borrow().data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);
        RefValue(Rc::new(RefCell::new(Value {
            data: t,
            op: Some("tanh"),
            children: vec![slf],
            non_chained_deps: None,
            grad: 0.0,
        })))
    }

    pub fn exp(slf: RefValue) -> RefValue {
        let x = slf.get().borrow().data;
        RefValue(Rc::new(RefCell::new(Value{
            data: x.exp(),
            op: Some("exp"),
            children: vec![slf],
            non_chained_deps: None,
            grad: 0.0
        })))
    }

    pub fn pow(slf: RefValue, other: f32) -> RefValue {
        let x = slf.get().borrow().data;
        RefValue(Rc::new(RefCell::new(Value {
            data: x.powf(other),
            op: Some("pow"),
            children: vec![slf],
            non_chained_deps: Some([other]),
            grad: 0.0 
        })))
    }
}

impl RefValue {
    pub fn get(&self) -> &RefCell<Value> {
        self.0.borrow()
    }
}

// Implement Hash and Eq for RefValue
impl Hash for RefValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash the pointer address of the Rc
        Rc::as_ptr(&self.0).hash(state);
    }
}

impl PartialEq for RefValue {
    fn eq(&self, other: &Self) -> bool {
        // Compare the pointer addresses
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for RefValue {}

impl fmt::Display for RefValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Value(data={:.4}, op={}, grad={:.4}, children={})",
            self.get().borrow().data,
            self.get().borrow().op.unwrap_or("?"),
            self.get().borrow().grad,
            self.get().borrow().children.len()
        )
    }
}

impl fmt::Display for Value{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Value(data={:.4}, op={}, grad={:.4}, children={})",
            self.data,
            self.op.unwrap_or("?"),
            self.grad,
            self.children.len()
        )
    }
}

impl From<f32> for RefValue {
    fn from(value: f32) -> Self {
        Value::new(value)
    }
}

impl From<i32> for RefValue {
    fn from(value: i32) -> Self {
        Value::new(value as f32)
    }
}

impl<T: Into<RefValue>> ops::Add<T> for RefValue {
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        let val = RefValue(Rc::new(RefCell::new(Value {
            data: self.get().borrow().data + rhs.get().borrow().data,
            op: Some("+"),
            children: vec![],
            non_chained_deps: None,
            grad: 0.0,
        })));
        val.get().borrow_mut().children.extend(vec![self, rhs]);

        val
    }
}

impl ops::Add<RefValue> for i32 {
    type Output = RefValue;

    fn add(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value + rhs
    }
}
impl ops::Add<RefValue> for f32 {
    type Output = RefValue;

    fn add(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value + rhs
    }
}

impl<T: Into<RefValue>> ops::Sub<T> for RefValue {
    type Output = RefValue;

    fn sub(self, rhs: T) -> Self::Output {
        self + (rhs.into() * -1)
    }
}

impl ops::Sub<RefValue> for i32 {
    type Output = RefValue;

    fn sub(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value - rhs
    }
}

impl ops::Sub<RefValue> for f32 {
    type Output = RefValue;

    fn sub(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value - rhs
    }
}

impl<T: Into<RefValue>> ops::Mul<T> for RefValue {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        let val = RefValue(Rc::new(RefCell::new(Value {
            data: self.get().borrow().data * rhs.get().borrow().data,
            op: Some("*"),
            children: vec![],
            non_chained_deps: None,
            grad: 0.0,
        })));

        val.get().borrow_mut().children.extend(vec![self, rhs]);

        val
    }
}

impl ops::Mul<RefValue> for i32 {
    type Output = RefValue;

    fn mul(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value * rhs
    }
}
impl ops::Mul<RefValue> for f32 {
    type Output = RefValue;

    fn mul(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value * rhs
    }
}

impl<T: Into<RefValue>> ops::Div<T> for RefValue {
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        let rhs = rhs.into();
        self * Value::pow(rhs, -1.0)
    }
    
}

impl ops::Div<RefValue> for i32 {
    type Output = RefValue;

    fn div(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value / rhs
    }
}
impl ops::Div<RefValue> for f32 {
    type Output = RefValue;

    fn div(self, rhs: RefValue) -> Self::Output {
        let self_as_ref_value: RefValue = self.into();
        self_as_ref_value / rhs
    }
}