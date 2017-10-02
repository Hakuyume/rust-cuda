use std::mem;

pub struct Slice<T> {
    _dummy: [T],
}

#[repr(C)]
struct Repr<T> {
    ptr: *mut T,
    len: usize,
}

impl<T> Slice<T> {
    pub fn as_ptr(&self) -> *const T {
        self.repr().ptr as *const T
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.repr().ptr
    }

    pub fn len(&self) -> usize {
        self.repr().len
    }

    fn repr(&self) -> Repr<T> {
        unsafe { mem::transmute::<&Slice<T>, Repr<T>>(self) }
    }
}

pub unsafe fn from_raw_parts<'a, T>(ptr: *const T, len: usize) -> &'a Slice<T> {
    mem::transmute::<Repr<T>, &Slice<T>>(Repr {
                                             ptr: ptr as *mut T,
                                             len,
                                         })
}

pub unsafe fn from_raw_parts_mut<'a, T>(ptr: *mut T, len: usize) -> &'a mut Slice<T> {
    mem::transmute::<Repr<T>, &mut Slice<T>>(Repr { ptr, len })
}

mod index;
