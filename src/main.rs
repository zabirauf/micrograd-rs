mod engine;

use engine::Value;

fn main() {
    let a = Value::new(5.0);
    let b = Value::new(-3.0);
    println!("A is {}", a);
    println!("B is {}", b);

    let c = a.clone() + b.clone();
    println!("C is {}", c);

    let d = a.clone()*b.clone();
    println!("D is {}", d);

    let e = a.clone()*b.clone() + c;
    println!("E is {}", e);
}
