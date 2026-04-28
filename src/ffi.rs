use crate::bitmap::QuiverBitmap;
use std::os::raw::c_uint;

/// Creates a new QuiverBitmap.
/// The caller is responsible for freeing it using `quiver_bitmap_free`.
#[no_mangle]
pub extern "C" fn quiver_bitmap_new() -> *mut QuiverBitmap {
    let b = Box::new(QuiverBitmap::new());
    Box::into_raw(b)
}

/// Frees a QuiverBitmap created by `quiver_bitmap_new` or other FFI functions.
#[no_mangle]
pub unsafe extern "C" fn quiver_bitmap_free(ptr: *mut QuiverBitmap) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// Inserts a value into the bitmap.
#[no_mangle]
pub unsafe extern "C" fn quiver_bitmap_insert(ptr: *mut QuiverBitmap, val: u32) {
    if let Some(bitmap) = ptr.as_mut() {
        bitmap.insert(val);
    }
}

/// Returns the number of elements in the bitmap.
#[no_mangle]
pub unsafe extern "C" fn quiver_bitmap_cardinality(ptr: *const QuiverBitmap) -> u32 {
    if let Some(bitmap) = ptr.as_ref() {
        bitmap.cardinality() as u32
    } else {
        0
    }
}

/// Performs a highly-optimized SIMD AND intersection of two bitmaps.
/// Returns a new bitmap. The caller must free it.
#[no_mangle]
pub unsafe extern "C" fn quiver_bitmap_and(
    left: *const QuiverBitmap,
    right: *const QuiverBitmap,
) -> *mut QuiverBitmap {
    if let (Some(l), Some(r)) = (left.as_ref(), right.as_ref()) {
        let result = l.and(r);
        Box::into_raw(Box::new(result))
    } else {
        std::ptr::null_mut()
    }
}

/// Performs a highly-optimized SIMD OR union of two bitmaps.
/// Returns a new bitmap. The caller must free it.
#[no_mangle]
pub unsafe extern "C" fn quiver_bitmap_or(
    left: *const QuiverBitmap,
    right: *const QuiverBitmap,
) -> *mut QuiverBitmap {
    if let (Some(l), Some(r)) = (left.as_ref(), right.as_ref()) {
        let result = l.or(r);
        Box::into_raw(Box::new(result))
    } else {
        std::ptr::null_mut()
    }
}
