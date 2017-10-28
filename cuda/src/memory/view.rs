use std::marker;

pub trait View<T> {
    fn as_ptr(&self) -> *const T;
    fn len(&self) -> usize;

    fn slice<'a>(&'a self, start: usize, end: usize) -> Slice<'a, T> {
        assert!(start <= end);
        assert!(end <= self.len());
        Slice {
            ptr: unsafe { self.as_ptr().offset(start as isize) },
            len: end - start,
            _dummy: marker::PhantomData::default(),
        }
    }
}

pub trait ViewMut<T>: View<T> {
    fn as_mut_ptr(&mut self) -> *mut T;

    fn slice_mut<'a>(&'a mut self, start: usize, end: usize) -> SliceMut<'a, T> {
        assert!(start <= end);
        assert!(end <= self.len());
        SliceMut {
            ptr: unsafe { self.as_mut_ptr().offset(start as isize) },
            len: end - start,
            _dummy: marker::PhantomData::default(),
        }
    }
}

pub struct Slice<'a, T: 'a> {
    ptr: *const T,
    len: usize,
    _dummy: marker::PhantomData<&'a ()>,
}

pub struct SliceMut<'a, T: 'a> {
    ptr: *mut T,
    len: usize,
    _dummy: marker::PhantomData<&'a ()>,
}

impl<'a, T: 'a> View<T> for Slice<'a, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T: 'a> View<T> for SliceMut<'a, T> {
    fn as_ptr(&self) -> *const T {
        self.ptr
    }
    fn len(&self) -> usize {
        self.len
    }
}

impl<'a, T: 'a> ViewMut<T> for SliceMut<'a, T> {
    fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}
