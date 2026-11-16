use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    mem::size_of,
    ops::{Index, IndexMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

const ALIGN_SIMD: usize = 64; // enough to support AVX-512
pub type AlignedMemory64 = AlignedMemory<ALIGN_SIMD>;

pub struct AlignedMemory<const ALIGN: usize> {
    p: *mut u64,
    sz_u64: usize,
    layout: Layout,
}

impl<const ALIGN: usize> AlignedMemory<{ ALIGN }> {
    pub fn new(sz_u64: usize) -> Self {
        let sz_bytes = sz_u64 * size_of::<u64>();
        let layout = Layout::from_size_align(sz_bytes, ALIGN).unwrap();

        let ptr;
        unsafe {
            ptr = alloc_zeroed(layout);
        }

        Self {
            p: ptr as *mut u64,
            sz_u64,
            layout,
        }
    }

    // pub fn from(data: &[u8]) -> Self {
    //     let sz_u64 = (data.len() + size_of::<u64>() - 1) / size_of::<u64>();
    //     let mut out = Self::new(sz_u64);
    //     let out_slice = out.as_mut_slice();
    //     let mut i = 0;
    //     for chunk in data.chunks(size_of::<u64>()) {
    //         out_slice[i] = u64::from_ne_bytes(chunk);
    //         i += 1;
    //     }
    //     out
    // }

    pub fn as_slice(&self) -> &[u64] {
        unsafe { from_raw_parts(self.p, self.sz_u64) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u64] {
        unsafe { from_raw_parts_mut(self.p, self.sz_u64) }
    }

    pub unsafe fn as_ptr(&self) -> *const u64 {
        self.p
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut u64 {
        self.p
    }

    pub fn len(&self) -> usize {
        self.sz_u64
    }
}

unsafe impl<const ALIGN: usize> Send for AlignedMemory<{ ALIGN }> {}
unsafe impl<const ALIGN: usize> Sync for AlignedMemory<{ ALIGN }> {}

impl<const ALIGN: usize> Drop for AlignedMemory<{ ALIGN }> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.p as *mut u8, self.layout);
        }
    }
}

impl<const ALIGN: usize> Index<usize> for AlignedMemory<{ ALIGN }> {
    type Output = u64;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<const ALIGN: usize> IndexMut<usize> for AlignedMemory<{ ALIGN }> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<const ALIGN: usize> Clone for AlignedMemory<{ ALIGN }> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.sz_u64);
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

pub type AlignedMemory16 = AlignedMemory16Type<ALIGN_SIMD>;

pub struct AlignedMemory16Type<const ALIGN: usize> {
    p: *mut u16,
    sz_u16: usize,
    layout: Layout,
}

impl<const ALIGN: usize> AlignedMemory16Type<ALIGN> {
    pub fn new(sz_u16: usize) -> Self {
        let sz_bytes = sz_u16 * size_of::<u16>();
        let layout = Layout::from_size_align(sz_bytes, ALIGN).unwrap();

        let ptr;
        unsafe {
            ptr = alloc_zeroed(layout);
        }

        Self {
            p: ptr as *mut u16,
            sz_u16,
            layout,
        }
    }

    pub fn as_slice(&self) -> &[u16] {
        unsafe { from_raw_parts(self.p, self.sz_u16) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u16] {
        unsafe { from_raw_parts_mut(self.p, self.sz_u16) }
    }

    pub unsafe fn as_ptr(&self) -> *const u16 {
        self.p
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut u16 {
        self.p
    }

    pub fn len(&self) -> usize {
        self.sz_u16
    }

}

unsafe impl<const ALIGN: usize> Send for AlignedMemory16Type<ALIGN> {}
unsafe impl<const ALIGN: usize> Sync for AlignedMemory16Type<ALIGN> {}

impl<const ALIGN: usize> Drop for AlignedMemory16Type<ALIGN> {
    fn drop(&mut self) {
        if self.sz_u16 != 0 {
            unsafe {
                dealloc(self.p as *mut u8, self.layout);
            }
        }
    }
}

impl<const ALIGN: usize> Index<usize> for AlignedMemory16Type<ALIGN> {
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<const ALIGN: usize> IndexMut<usize> for AlignedMemory16Type<ALIGN> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<const ALIGN: usize> Clone for AlignedMemory16Type<ALIGN> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.sz_u16);
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}

pub type AlignedMemory32 = AlignedMemory32Type<ALIGN_SIMD>;

pub struct AlignedMemory32Type<const ALIGN: usize> {
    p: *mut u32,
    sz_u32: usize,
    layout: Layout,
}

impl<const ALIGN: usize> AlignedMemory32Type<ALIGN> {
    pub fn new(sz_u32: usize) -> Self {
        let sz_bytes = sz_u32 * size_of::<u32>();
        let layout = Layout::from_size_align(sz_bytes, ALIGN).unwrap();

        let ptr;
        unsafe {
            ptr = alloc_zeroed(layout);
        }

        Self {
            p: ptr as *mut u32,
            sz_u32,
            layout,
        }
    }

    pub fn as_slice(&self) -> &[u32] {
        unsafe { from_raw_parts(self.p, self.sz_u32) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u32] {
        unsafe { from_raw_parts_mut(self.p, self.sz_u32) }
    }

    pub unsafe fn as_ptr(&self) -> *const u32 {
        self.p
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut u32 {
        self.p
    }

    pub fn len(&self) -> usize {
        self.sz_u32
    }
}

unsafe impl<const ALIGN: usize> Send for AlignedMemory32Type<ALIGN> {}
unsafe impl<const ALIGN: usize> Sync for AlignedMemory32Type<ALIGN> {}

impl<const ALIGN: usize> Drop for AlignedMemory32Type<ALIGN> {
    fn drop(&mut self) {
        if self.sz_u32 != 0 {
            unsafe {
                dealloc(self.p as *mut u8, self.layout);
            }
        }
    }
}

impl<const ALIGN: usize> Index<usize> for AlignedMemory32Type<ALIGN> {
    type Output = u32;

    fn index(&self, index: usize) -> &Self::Output {
        &self.as_slice()[index]
    }
}

impl<const ALIGN: usize> IndexMut<usize> for AlignedMemory32Type<ALIGN> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.as_mut_slice()[index]
    }
}

impl<const ALIGN: usize> Clone for AlignedMemory32Type<ALIGN> {
    fn clone(&self) -> Self {
        let mut out = Self::new(self.sz_u32);
        out.as_mut_slice().copy_from_slice(self.as_slice());
        out
    }
}