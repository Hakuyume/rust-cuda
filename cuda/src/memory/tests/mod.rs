use super::Array;

#[test]
fn malloc() {
    let m = Array::<u8>::new(16).unwrap();
    assert!(!m.as_ptr().is_null());
    assert_eq!(m.len(), 16);
}

#[test]
#[should_panic(expected = "MemoryAllocation")]
fn malloc_huge() {
    Array::<u8>::new(1 << 48).unwrap();
}

mod slice;
mod split_at;
mod memcpy;
