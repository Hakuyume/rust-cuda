type Array = super::Array<super::Owned<u8>>;

#[test]
fn malloc() {
    let m = Array::new(16).unwrap();
    assert_eq!(m.len(), 16);
}

#[test]
#[should_panic(expected = "MemoryAllocation")]
fn malloc_huge() {
    Array::new(1 << 48).unwrap();
}

mod slice;
mod split_at;
mod memcpy;
