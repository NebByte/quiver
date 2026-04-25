/// QuiverDB: A minimal columnar database powered by Quiver bitmap indexes.
///
/// Demonstrates real-world usage of Quiver's ARM-optimized primitives:
/// - QuiverBitmap for bitmap indexes (uses NEON-accelerated popcount, AND, OR)
/// - Rank/select for positional access
/// - Compressed storage via bitmap containers
///
/// This is NOT a production database — it's a benchmark vehicle to prove
/// that Quiver's NEON speedups translate into real application-level gains.

use crate::bitmap::QuiverBitmap;
use std::collections::HashMap;

// ── Schema ────────────────────────────────────────────────────

/// A column value that can be indexed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Value {
    Int(i64),
    Str(String),
}

/// A single column of data.
#[derive(Clone, Debug)]
pub enum Column {
    Int(Vec<i64>),
    Str(Vec<String>),
}

/// Bitmap index: for each distinct value in a column, stores a bitmap
/// of which row IDs contain that value.
///
/// Query: "WHERE city = 'NYC'" → look up bitmap for "NYC", get all row IDs instantly.
/// Query: "WHERE age = 25 AND city = 'NYC'" → AND the two bitmaps together.
#[derive(Clone, Debug)]
pub struct BitmapIndex {
    value_bitmaps: HashMap<Value, QuiverBitmap>,
}

impl BitmapIndex {
    pub fn new() -> Self {
        BitmapIndex {
            value_bitmaps: HashMap::new(),
        }
    }

    /// Index a value at a given row ID.
    pub fn index(&mut self, value: Value, row_id: u32) {
        self.value_bitmaps
            .entry(value)
            .or_insert_with(QuiverBitmap::new)
            .insert(row_id);
    }

    /// Get the bitmap for a specific value (equality lookup).
    pub fn get(&self, value: &Value) -> Option<&QuiverBitmap> {
        self.value_bitmaps.get(value)
    }

    /// Get bitmaps for a range of integer values and OR them together.
    pub fn range_int(&self, min: i64, max: i64) -> QuiverBitmap {
        let mut result = QuiverBitmap::new();
        for (val, bitmap) in &self.value_bitmaps {
            if let Value::Int(v) = val {
                if *v >= min && *v <= max {
                    result = result.or(bitmap);
                }
            }
        }
        result
    }

    /// Number of distinct values indexed.
    pub fn distinct_count(&self) -> usize {
        self.value_bitmaps.len()
    }
}

// ── Database ──────────────────────────────────────────────────

/// A minimal columnar database with bitmap indexes.
pub struct QuiverDB {
    columns: HashMap<String, Column>,
    indexes: HashMap<String, BitmapIndex>,
    row_count: u32,
}

impl QuiverDB {
    pub fn new() -> Self {
        QuiverDB {
            columns: HashMap::new(),
            indexes: HashMap::new(),
            row_count: 0,
        }
    }

    /// Add an integer column with data and build a bitmap index on it.
    pub fn add_int_column(&mut self, name: &str, data: Vec<i64>) {
        assert!(
            self.row_count == 0 || data.len() == self.row_count as usize,
            "column length mismatch"
        );
        if self.row_count == 0 {
            self.row_count = data.len() as u32;
        }

        // Build bitmap index
        let mut index = BitmapIndex::new();
        for (row_id, &value) in data.iter().enumerate() {
            index.index(Value::Int(value), row_id as u32);
        }

        self.columns.insert(name.to_string(), Column::Int(data));
        self.indexes.insert(name.to_string(), index);
    }

    /// Add a string column with data and build a bitmap index on it.
    pub fn add_str_column(&mut self, name: &str, data: Vec<String>) {
        assert!(
            self.row_count == 0 || data.len() == self.row_count as usize,
            "column length mismatch"
        );
        if self.row_count == 0 {
            self.row_count = data.len() as u32;
        }

        // Build bitmap index
        let mut index = BitmapIndex::new();
        for (row_id, value) in data.iter().enumerate() {
            index.index(Value::Str(value.clone()), row_id as u32);
        }

        self.columns
            .insert(name.to_string(), Column::Str(data));
        self.indexes.insert(name.to_string(), index);
    }

    pub fn row_count(&self) -> u32 {
        self.row_count
    }

    // ── Queries ───────────────────────────────────────────────

    /// COUNT(*) WHERE col = value
    /// Uses: bitmap lookup + NEON-accelerated cardinality
    pub fn count_where_eq(&self, column: &str, value: &Value) -> u64 {
        match self.indexes.get(column) {
            Some(index) => match index.get(value) {
                Some(bitmap) => bitmap.cardinality(),
                None => 0,
            },
            None => 0,
        }
    }

    /// COUNT(*) WHERE col1 = val1 AND col2 = val2
    /// Uses: two bitmap lookups + NEON-accelerated AND + cardinality
    pub fn count_where_and(
        &self,
        col1: &str,
        val1: &Value,
        col2: &str,
        val2: &Value,
    ) -> u64 {
        let bm1 = self.get_bitmap(col1, val1);
        let bm2 = self.get_bitmap(col2, val2);
        match (bm1, bm2) {
            (Some(a), Some(b)) => a.and(b).cardinality(),
            _ => 0,
        }
    }

    /// COUNT(*) WHERE col1 = val1 OR col2 = val2
    /// Uses: two bitmap lookups + NEON-accelerated OR + cardinality
    pub fn count_where_or(
        &self,
        col1: &str,
        val1: &Value,
        col2: &str,
        val2: &Value,
    ) -> u64 {
        let bm1 = self.get_bitmap(col1, val1);
        let bm2 = self.get_bitmap(col2, val2);
        match (bm1, bm2) {
            (Some(a), Some(b)) => a.or(b).cardinality(),
            (Some(a), None) => a.cardinality(),
            (None, Some(b)) => b.cardinality(),
            _ => 0,
        }
    }

    /// COUNT(*) WHERE col BETWEEN min AND max (integer columns only)
    /// Uses: range scan over bitmap index + OR chain + cardinality
    pub fn count_where_range(&self, column: &str, min: i64, max: i64) -> u64 {
        match self.indexes.get(column) {
            Some(index) => index.range_int(min, max).cardinality(),
            None => 0,
        }
    }

    /// Multi-column AND: WHERE col1 = v1 AND col2 = v2 AND col3 = v3 ...
    /// Uses: chained NEON-accelerated bitmap AND operations
    pub fn count_where_multi_and(&self, filters: &[(&str, &Value)]) -> u64 {
        let mut result: Option<QuiverBitmap> = None;
        for &(col, val) in filters {
            match self.get_bitmap(col, val) {
                Some(bm) => {
                    result = Some(match result {
                        Some(acc) => acc.and(bm),
                        None => bm.clone(),
                    });
                }
                None => return 0,
            }
        }
        result.map_or(0, |bm| bm.cardinality())
    }

    /// SELECT (get matching row IDs) WHERE col = value
    /// Uses: bitmap lookup + iterator
    pub fn select_where_eq(&self, column: &str, value: &Value) -> Vec<u32> {
        match self.get_bitmap(column, value) {
            Some(bm) => bm.iter().collect(),
            None => Vec::new(),
        }
    }

    /// Helper: get bitmap for column=value
    fn get_bitmap(&self, column: &str, value: &Value) -> Option<&QuiverBitmap> {
        self.indexes.get(column)?.get(value)
    }

    // ── Naive (no-index) equivalents for comparison ───────────

    /// Naive COUNT WHERE col = value — full column scan, no bitmap index.
    pub fn naive_count_where_eq_int(&self, column: &str, target: i64) -> u64 {
        match self.columns.get(column) {
            Some(Column::Int(data)) => data.iter().filter(|&&v| v == target).count() as u64,
            _ => 0,
        }
    }

    /// Naive COUNT WHERE col1 = v1 AND col2 = v2 — full scan both columns.
    pub fn naive_count_where_and_int(
        &self,
        col1: &str,
        val1: i64,
        col2: &str,
        val2: i64,
    ) -> u64 {
        let c1 = match self.columns.get(col1) {
            Some(Column::Int(d)) => d,
            _ => return 0,
        };
        let c2 = match self.columns.get(col2) {
            Some(Column::Int(d)) => d,
            _ => return 0,
        };
        c1.iter()
            .zip(c2.iter())
            .filter(|(&a, &b)| a == val1 && b == val2)
            .count() as u64
    }

    /// Naive COUNT WHERE col BETWEEN min AND max — full column scan.
    pub fn naive_count_where_range(&self, column: &str, min: i64, max: i64) -> u64 {
        match self.columns.get(column) {
            Some(Column::Int(data)) => {
                data.iter().filter(|&&v| v >= min && v <= max).count() as u64
            }
            _ => 0,
        }
    }

    // ── Stats ─────────────────────────────────────────────────

    /// Total memory used by all bitmap indexes.
    pub fn index_memory_bytes(&self) -> usize {
        let mut total = 0;
        for (_, index) in &self.indexes {
            for (_, bitmap) in &index.value_bitmaps {
                total += bitmap.size_in_bytes();
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_queries() {
        let mut db = QuiverDB::new();
        db.add_int_column("age", vec![25, 30, 25, 40, 30, 25]);
        db.add_str_column(
            "city",
            vec![
                "NYC".into(),
                "LA".into(),
                "NYC".into(),
                "NYC".into(),
                "LA".into(),
                "SF".into(),
            ],
        );

        // Single column equality
        assert_eq!(db.count_where_eq("age", &Value::Int(25)), 3);
        assert_eq!(db.count_where_eq("age", &Value::Int(30)), 2);
        assert_eq!(db.count_where_eq("city", &Value::Str("NYC".into())), 3);

        // AND: age=25 AND city=NYC → rows 0,2
        assert_eq!(
            db.count_where_and("age", &Value::Int(25), "city", &Value::Str("NYC".into())),
            2
        );

        // OR: age=40 OR city=LA → rows 1,3,4
        assert_eq!(
            db.count_where_or("age", &Value::Int(40), "city", &Value::Str("LA".into())),
            3
        );

        // Naive equivalents match
        assert_eq!(db.naive_count_where_eq_int("age", 25), 3);
    }

    #[test]
    fn test_range_query() {
        let mut db = QuiverDB::new();
        let ages: Vec<i64> = (0..100).map(|i| 18 + (i % 50)).collect();
        db.add_int_column("age", ages);

        // Range: age between 25 and 35
        let bitmap_count = db.count_where_range("age", 25, 35);
        let naive_count = db.naive_count_where_range("age", 25, 35);
        assert_eq!(bitmap_count, naive_count);
    }
}
