use axum::{
    routing::{get, post},
    Router, Json,
    extract::Multipart,
};
use serde::Serialize;
use tower_http::services::ServeDir;
use tower_http::cors::CorsLayer;
use std::net::SocketAddr;
use csv::ReaderBuilder;

use quiver::minidb::QuiverDB;
use rusqlite::{Connection, params};

#[derive(Serialize)]
struct BenchmarkResponse {
    quiver_time_ns: u64,
    sqlite_time_ns: u64,
    speedup_multiplier: f64,
    rows_processed: usize,
}

#[tokio::main]
async fn main() {
    println!("🚀 Starting Quiver ARM Cloud Server...");
    
    let app = Router::new()
        .nest_service("/", ServeDir::new("docs"))
        .route("/api/benchmark", post(run_investor_benchmark))
        .route("/api/upload_benchmark", post(run_custom_benchmark))
        .layer(CorsLayer::permissive());

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("🌐 Server listening on http://127.0.0.1:3000");
    
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

/// Dynamic benchmark processing a user-uploaded CSV file
async fn run_custom_benchmark(mut multipart: Multipart) -> Json<BenchmarkResponse> {
    let mut data = Vec::new();

    // Read the uploaded file
    if let Some(field) = multipart.next_field().await.unwrap() {
        let bytes = field.bytes().await.unwrap();
        data.extend_from_slice(&bytes);
    }

    let mut rdr = ReaderBuilder::new().from_reader(data.as_slice());
    let mut ages = Vec::new();
    let mut depts = Vec::new();

    for result in rdr.records() {
        if let Ok(record) = result {
            if let (Ok(age), Ok(dept)) = (record[0].parse::<i64>(), record[1].parse::<i64>()) {
                ages.push(age);
                depts.push(dept);
            }
        }
    }

    let num_rows = ages.len();

    // 1. Build QuiverDB
    let mut db = QuiverDB::new();
    db.add_int_column("age", ages.clone());
    db.add_int_column("dept", depts.clone());

    // 2. Build SQLite
    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER, dept INTEGER)", []).unwrap();
    conn.execute("CREATE INDEX idx_age ON users(age)", []).unwrap();
    conn.execute("CREATE INDEX idx_dept ON users(dept)", []).unwrap();
    
    let tx = conn.transaction().unwrap();
    {
        let mut stmt = tx.prepare("INSERT INTO users (id, age, dept) VALUES (?1, ?2, ?3)").unwrap();
        for i in 0..num_rows {
            stmt.execute(params![i as i64, ages[i], depts[i]]).unwrap();
        }
    }
    tx.commit().unwrap();

    // 3. Run Benchmarks (Looking for exact match: age=30 AND dept=5)
    let quiver_start = std::time::Instant::now();
    let _ = db.count_where_and("age", &30, "dept", &5);
    let quiver_time_ns = quiver_start.elapsed().as_nanos() as u64;

    let sqlite_start = std::time::Instant::now();
    let mut stmt = conn.prepare_cached("SELECT count(*) FROM users WHERE age = 30 AND dept = 5").unwrap();
    let _: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
    let sqlite_time_ns = sqlite_start.elapsed().as_nanos() as u64;

    let speedup = sqlite_time_ns as f64 / std::cmp::max(quiver_time_ns, 1) as f64;

    Json(BenchmarkResponse {
        quiver_time_ns,
        sqlite_time_ns,
        speedup_multiplier: speedup,
        rows_processed: num_rows,
    })
}

/// Fallback standard benchmark
async fn run_investor_benchmark() -> Json<BenchmarkResponse> {
    let num_rows = 100_000;
    
    let ages: Vec<i64> = (0..num_rows as i64).map(|i| 18 + (i * 7 + 3) % 62).collect();
    let depts: Vec<i64> = (0..num_rows as i64).map(|i| (i * 13 + 5) % 20).collect();

    let mut db = QuiverDB::new();
    db.add_int_column("age", ages.clone());
    db.add_int_column("dept", depts.clone());

    let mut conn = Connection::open_in_memory().unwrap();
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, age INTEGER, dept INTEGER)", []).unwrap();
    conn.execute("CREATE INDEX idx_age ON users(age)", []).unwrap();
    conn.execute("CREATE INDEX idx_dept ON users(dept)", []).unwrap();
    
    let tx = conn.transaction().unwrap();
    {
        let mut stmt = tx.prepare("INSERT INTO users (id, age, dept) VALUES (?1, ?2, ?3)").unwrap();
        for i in 0..num_rows {
            stmt.execute(params![i as i64, ages[i], depts[i]]).unwrap();
        }
    }
    tx.commit().unwrap();

    let quiver_start = std::time::Instant::now();
    let _ = db.count_where_and("age", &30, "dept", &5);
    let quiver_time_ns = quiver_start.elapsed().as_nanos() as u64;

    let sqlite_start = std::time::Instant::now();
    let mut stmt = conn.prepare_cached("SELECT count(*) FROM users WHERE age = 30 AND dept = 5").unwrap();
    let _: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
    let sqlite_time_ns = sqlite_start.elapsed().as_nanos() as u64;

    let speedup = sqlite_time_ns as f64 / std::cmp::max(quiver_time_ns, 1) as f64;

    Json(BenchmarkResponse {
        quiver_time_ns,
        sqlite_time_ns,
        speedup_multiplier: speedup,
        rows_processed: num_rows,
    })
}
