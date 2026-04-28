use quiver::minidb::{QuiverDB, Value};
use rusqlite::{Connection, params};

fn main() {
    let num_rows: usize = 500_000;

    let ages: Vec<i64> = (0..num_rows as i64).map(|i| 18 + (i * 7 + 3) % 62).collect();
    let depts: Vec<i64> = (0..num_rows as i64).map(|i| (i * 13 + 5) % 20).collect();

    // Build QuiverDB
    let mut db = QuiverDB::new();
    db.add_int_column("age", ages.clone());
    db.add_int_column("dept", depts.clone());

    // Build SQLite
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

    // Warm up
    for _ in 0..5 {
        let _ = db.count_where_and("age", &Value::Int(30), "dept", &Value::Int(5));
    }
    for _ in 0..5 {
        let mut stmt = conn.prepare_cached("SELECT count(*) FROM users WHERE age = 30 AND dept = 5").unwrap();
        let _: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
    }

    // Benchmark Quiver (average of 50 runs)
    let mut quiver_total: u64 = 0;
    for _ in 0..50 {
        let start = std::time::Instant::now();
        let _ = db.count_where_and("age", &Value::Int(30), "dept", &Value::Int(5));
        quiver_total += start.elapsed().as_nanos() as u64;
    }
    let quiver_avg = quiver_total / 50;

    // Benchmark SQLite (average of 50 runs)
    let mut sqlite_total: u64 = 0;
    for _ in 0..50 {
        let start = std::time::Instant::now();
        let mut stmt = conn.prepare_cached("SELECT count(*) FROM users WHERE age = 30 AND dept = 5").unwrap();
        let _: i64 = stmt.query_row([], |row| row.get(0)).unwrap();
        sqlite_total += start.elapsed().as_nanos() as u64;
    }
    let sqlite_avg = sqlite_total / 50;

    let speedup = sqlite_avg as f64 / std::cmp::max(quiver_avg, 1) as f64;

    // Output JSON
    println!("{{\"quiver_time_ns\":{},\"sqlite_time_ns\":{},\"speedup_multiplier\":{:.1},\"rows_processed\":{},\"arch\":\"{}\",\"timestamp\":\"{}\"}}",
        quiver_avg,
        sqlite_avg,
        speedup,
        num_rows,
        std::env::consts::ARCH,
        chrono_lite()
    );
}

fn chrono_lite() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    format!("{}", dur.as_secs())
}
