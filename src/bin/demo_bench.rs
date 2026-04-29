use quiver::bench::bench;
use quiver::minidb::{QuiverDB, Value};
use rusqlite::Connection;
use serde::Serialize;
use std::env;
use std::fs::File;
use std::time::Duration;

#[derive(Serialize)]
struct DemoResult {
    quiver_ns: f64,
    sqlite_ns: f64,
    speedup: f64,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let csv_path = if args.len() > 1 { &args[1] } else { "demo/test_db.csv" };

    println!("Loading data from {}...", csv_path);
    
    // 1. Read the CSV and extract the first column as integers.
    let mut values = Vec::new();
    let mut target_val = 0;
    
    if let Ok(file) = File::open(csv_path) {
        let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(file);
        for (i, result) in rdr.records().enumerate() {
            if let Ok(record) = result {
                if let Some(val_str) = record.get(0) {
                    if let Ok(val) = val_str.parse::<i64>() {
                        values.push(val);
                        if i == values.len() / 2 {
                            target_val = val; // pick a target value from the middle
                        }
                    }
                }
            }
        }
    } else {
        println!("Could not open {}; falling back to synthetic data", csv_path);
        for i in 0..10000 {
            values.push((i % 100) as i64);
            if i == 5000 { target_val = 50; }
        }
    }

    if values.is_empty() {
        println!("No numeric data found in the first column.");
        return;
    }

    println!("Loaded {} rows. Target value for query: {}", values.len(), target_val);

    // 2. Build Quiver DB
    let mut db = QuiverDB::new();
    for val in &values {
        db.insert(vec![Value::Int(*val)]);
    }
    
    let min_dur = Duration::from_millis(500);
    
    let q_quiver = bench("Quiver", min_dur, || {
        db.query_and(0, Value::Int(target_val)).len() as u64
    });

    // 3. Build SQLite DB
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE users (col1 INTEGER)", []).unwrap();
    conn.execute("CREATE INDEX idx_col1 ON users(col1)", []).unwrap();
    
    {
        let tx = conn.transaction().unwrap();
        let mut stmt = tx.prepare("INSERT INTO users (col1) VALUES (?1)").unwrap();
        for val in &values {
            stmt.execute([*val]).unwrap();
        }
        drop(stmt);
        tx.commit().unwrap();
    }
    
    let q_sqlite = bench("SQLite", min_dur, || {
        let mut stmt = conn.prepare_cached("SELECT count(*) FROM users WHERE col1 = ?1").unwrap();
        let count: i64 = stmt.query_row([target_val], |row| row.get(0)).unwrap();
        count as u64
    });

    let speedup = q_sqlite.mean_ns / q_quiver.mean_ns;
    
    println!("Quiver is {:.1}x faster than SQLite", speedup);

    let res = DemoResult {
        quiver_ns: q_quiver.mean_ns,
        sqlite_ns: q_sqlite.mean_ns,
        speedup,
    };

    let json = serde_json::to_string_pretty(&res).unwrap();
    std::fs::write("demo_results.json", json).unwrap();
}
