use std::cell::RefCell;
use std::fmt;
use std::ops;
use std::collections::VecDeque;
use std::collections::HashSet;

type PropagateFn = fn(value: RefCell<&mut Value>);
#[derive(Clone, PartialEq)]
pub struct Value {
    data: f32,
    children: Vec<Box<Value>>,
    op: Option<&'static str>,
    grad: f32,
    back_propagate: PropagateFn,
}

impl Value {
    pub fn default() -> Value {
        Value {
            data: 0.0,
            children: vec![],
            op: None,
            grad: 0.0,
            back_propagate: |_| {}
        }
    }

    pub fn new(data: f32) -> Value {
        Value {
            data: data,
            ..Value::default()
        }
    }

    pub fn back_propagate(val: &mut Value) -> () {
        // let mut queue: VecDeque<(&Value, RefCell<&mut Value>)> = VecDeque::new();
        // let mut root_val = Value::new(0.0);
        // let root = RefCell::new(&mut root_val);
        // queue.push_back((val, root));

        // while let Some((next_val, val_parent)) = queue.pop_front() {
        //     (next_val.back_propagate)(val_parent);

        //     for child in next_val.children.iter() {
        //         let parent_ref = RefCell::new(&mut *next_val);
        //         queue.push_back((child, parent_ref));
        //     }
        // };

        val.grad = 1.0;
        let mut topo : Vec<*const Value>= vec![];
        let mut visited: HashSet<*const Value> = HashSet::new();
        
        fn build_topo(v: &Value, visited: &mut HashSet<*const Value>, topo: &mut Vec<*const Value>) {
            if !visited.contains(&(v as *const Value)) {
                visited.insert(v as *const Value);
                for child in &v.children {
                    build_topo(child, visited, topo);
                }
                topo.push(v as *const Value);
            }
        }

        build_topo(&val, &mut visited, &mut topo);

        topo.reverse();
        for v in topo.iter() {
        unsafe {
            (**v).back_propagate(&mut *(v as *mut Value));
        }
        }
    }

    pub fn tanh(self) -> Value {
        let x = self.data;
        let t = ((2.0 * x).exp() - 1.0) / ((2.0 * x).exp() + 1.0);

        Value {
            data: t,
            op: Some("tanh"),
            children: vec![Box::new(self)],
            back_propagate: |val| {
                let mut val_mut = val.borrow_mut();
                if let [op] = &mut *val_mut.children {
                    op.grad = 1.0 - op.grad.powi(2);
                } else {
                    panic!("Children doesn't contain single item");
                }
            },
            ..Value::default()
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(data={:.4}, op={}, children={})", self.data, self.op.unwrap_or("?"), self.children.len())
    }
}

impl ops::Add for Value {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data + rhs.data,
            op: Some("+"),
            children: vec![Box::new(self), Box::new(rhs)],
            back_propagate: |val| {
                let mut val_mut = val.borrow_mut();
                let val_grad = val_mut.grad;
                if let [lhs, rhs] = &mut *val_mut.children {
                    lhs.grad = 1.0 * val_grad; // Increment the gradient of the left-hand side
                    rhs.grad = 1.0 * val_grad; // Increment the gradient of the right-hand side
                } else {
                    panic!("Children doesn't contain two items");
                }
            },
            ..Value::default()
        }
    }
}

impl ops::Mul for Value {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Value {
            data: self.data * rhs.data,
            op: Some("*"),
            children: vec![Box::new(self), Box::new(rhs)],
            back_propagate: |val| {
                let mut val_mut = val.borrow_mut();
                let val_grad = val_mut.grad;
                if let [lhs, rhs] = &mut *val_mut.children {
                    lhs.grad = rhs.data * val_grad; // Increment the gradient of the left-hand side
                    rhs.grad = lhs.data * val_grad; // Increment the gradient of the right-hand side
                } else {
                    panic!("Children doesn't contain two items");
                }
            },
            ..Value::default()
        }
    }
}