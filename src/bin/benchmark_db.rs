use quiver::minidb::{QuiverDB, Value};
use rusqlite::{params, Connection};
use std::env;
use std::path::Path;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut db_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--db" && i + 1 < args.len() {
            db_path = Some(args[i + 1].clone());
            i += 2;
        } else {
            i += 1;
        }
    }
    let db_path = db_path.unwrap_or_else(|| {
        eprintln!("usage: benchmark_db --db <path-to-sqlite.db>");
        std::process::exit(2);
    });
    if !Path::new(&db_path).exists() {
        eprintln!("error: db file not found: {}", db_path);
        std::process::exit(2);
    }

    let conn = Connection::open(&db_path).expect("open sqlite db");

    let (table, col1, col2) = pick_table_and_columns(&conn);
    eprintln!("[demo] using table='{}' col1='{}' col2='{}'", table, col1, col2);

    let load_q = format!(
        "SELECT \"{c1}\", \"{c2}\" FROM \"{t}\" WHERE \"{c1}\" IS NOT NULL AND \"{c2}\" IS NOT NULL",
        t = table,
        c1 = col1,
        c2 = col2
    );
    let mut col1_data: Vec<i64> = Vec::new();
    let mut col2_data: Vec<i64> = Vec::new();
    {
        let mut stmt = conn.prepare(&load_q).expect("prepare load");
        let mut rows = stmt.query([]).expect("query rows");
        while let Some(row) = rows.next().unwrap() {
            let a: i64 = row.get(0).unwrap_or(0);
            let b: i64 = row.get(1).unwrap_or(0);
            col1_data.push(a);
            col2_data.push(b);
        }
    }

    let n = col1_data.len();
    if n == 0 {
        eprintln!("error: no rows in {}", table);
        std::process::exit(3);
    }

    let target1 = col1_data[n / 2];
    let target2 = col2_data[n / 2];

    let mut qdb = QuiverDB::new();
    qdb.add_int_column("c1", col1_data.clone());
    qdb.add_int_column("c2", col2_data.clone());

    let _ = conn.execute(
        &format!(
            "CREATE INDEX IF NOT EXISTS idx_demo_c1 ON \"{}\"(\"{}\")",
            table, col1
        ),
        [],
    );
    let _ = conn.execute(
        &format!(
            "CREATE INDEX IF NOT EXISTS idx_demo_c2 ON \"{}\"(\"{}\")",
            table, col2
        ),
        [],
    );
    let sql_q = format!(
        "SELECT count(*) FROM \"{t}\" WHERE \"{c1}\" = ?1 AND \"{c2}\" = ?2",
        t = table,
        c1 = col1,
        c2 = col2
    );

    for _ in 0..5 {
        let _ = qdb.count_where_and("c1", &Value::Int(target1), "c2", &Value::Int(target2));
    }
    for _ in 0..5 {
        let mut stmt = conn.prepare_cached(&sql_q).unwrap();
        let _: i64 = stmt
            .query_row(params![target1, target2], |r| r.get(0))
            .unwrap();
    }

    let mut q_total: u128 = 0;
    for _ in 0..50 {
        let start = Instant::now();
        let _ = qdb.count_where_and("c1", &Value::Int(target1), "c2", &Value::Int(target2));
        q_total += start.elapsed().as_nanos();
    }
    let q_avg = (q_total / 50) as u64;

    let mut s_total: u128 = 0;
    for _ in 0..50 {
        let start = Instant::now();
        let mut stmt = conn.prepare_cached(&sql_q).unwrap();
        let _: i64 = stmt
            .query_row(params![target1, target2], |r| r.get(0))
            .unwrap();
        s_total += start.elapsed().as_nanos();
    }
    let s_avg = (s_total / 50) as u64;

    let speedup = s_avg as f64 / std::cmp::max(q_avg, 1) as f64;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let db_basename = Path::new(&db_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("user.db");

    println!(
        "{{\"db_file\":\"{}\",\"table\":\"{}\",\"col1\":\"{}\",\"col2\":\"{}\",\"target1\":{},\"target2\":{},\"quiver_time_ns\":{},\"sqlite_time_ns\":{},\"speedup_multiplier\":{:.1},\"rows_processed\":{},\"arch\":\"{}\",\"timestamp\":\"{}\"}}",
        db_basename, table, col1, col2, target1, target2, q_avg, s_avg, speedup, n, std::env::consts::ARCH, ts
    );
}

fn pick_table_and_columns(conn: &Connection) -> (String, String, String) {
    let tables: Vec<String> = {
        let mut stmt = conn
            .prepare(
                "SELECT name FROM sqlite_master \
                 WHERE type='table' AND name NOT LIKE 'sqlite_%' \
                 ORDER BY name",
            )
            .unwrap();
        stmt.query_map([], |r| r.get::<_, String>(0))
            .unwrap()
            .flatten()
            .collect()
    };
    if tables.is_empty() {
        eprintln!("error: no user tables in this DB");
        std::process::exit(3);
    }

    let mut best: Option<(String, String, String, i64)> = None;
    for t in &tables {
        let pragma = format!("PRAGMA table_info(\"{}\")", t);
        let int_cols: Vec<String> = {
            let mut stmt = conn.prepare(&pragma).unwrap();
            stmt.query_map([], |r| {
                let name: String = r.get(1)?;
                let ty: String = r.get(2).unwrap_or_default();
                let pk: i64 = r.get(5).unwrap_or(0);
                Ok((name, ty, pk))
            })
            .unwrap()
            .flatten()
            .filter(|(_, ty, pk)| *pk == 0 && ty.to_uppercase().contains("INT"))
            .map(|(n, _, _)| n)
            .collect()
        };
        if int_cols.len() < 2 {
            continue;
        }
        let row_count: i64 = conn
            .query_row(&format!("SELECT count(*) FROM \"{}\"", t), [], |r| r.get(0))
            .unwrap_or(0);
        if let Some((_, _, _, prev)) = &best {
            if row_count <= *prev {
                continue;
            }
        }
        best = Some((t.clone(), int_cols[0].clone(), int_cols[1].clone(), row_count));
    }

    match best {
        Some((t, c1, c2, _)) => (t, c1, c2),
        None => {
            eprintln!("error: no table found with at least two non-PK INTEGER columns");
            std::process::exit(3);
        }
    }
}
