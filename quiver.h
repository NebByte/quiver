#ifndef QUIVER_H
#define QUIVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque struct representing the QuiverBitmap
typedef struct QuiverBitmap QuiverBitmap;

// Creates a new, empty QuiverBitmap.
// The caller is responsible for calling quiver_bitmap_free on the returned pointer.
QuiverBitmap* quiver_bitmap_new(void);

// Frees the memory associated with a QuiverBitmap.
void quiver_bitmap_free(QuiverBitmap* ptr);

// Inserts a 32-bit integer into the bitmap.
void quiver_bitmap_insert(QuiverBitmap* ptr, uint32_t val);

// Returns the total number of elements stored in the bitmap.
uint32_t quiver_bitmap_cardinality(const QuiverBitmap* ptr);

// Performs a highly-optimized SIMD AND intersection of two bitmaps.
// Returns a new QuiverBitmap containing only the elements present in both.
// The caller is responsible for freeing the returned bitmap.
QuiverBitmap* quiver_bitmap_and(const QuiverBitmap* left, const QuiverBitmap* right);

// Performs a highly-optimized SIMD OR union of two bitmaps.
// Returns a new QuiverBitmap containing elements present in either bitmap.
// The caller is responsible for freeing the returned bitmap.
QuiverBitmap* quiver_bitmap_or(const QuiverBitmap* left, const QuiverBitmap* right);

#ifdef __cplusplus
}
#endif

#endif // QUIVER_H
