//! Storage buffers: arrays a compute kernel reads (`shader.buffer`) or writes (`shader.buffer_rw`),
//! of `stride`-byte elements (`compute.rs`). The CPU copy is uploaded when it changes, and again
//! after a device reset; what a kernel writes stays on the GPU.
use {
    crate::{cx::Cx, id_pool::*, makepad_error_log::*},
    std::rc::Rc,
};

#[derive(Debug, Clone, PartialEq)]
pub struct StorageBuffer(Rc<PoolId>);

#[derive(Clone, Debug, PartialEq, Copy)]
pub struct StorageBufferId(pub(crate) usize, u64);

impl StorageBufferId {
    /// The pool generation of the slot this id names.
    pub(crate) fn generation(self) -> u64 {
        self.1
    }
}

impl StorageBuffer {
    /// `len` elements of `stride` bytes, zeroed.
    pub fn new(cx: &mut Cx, stride: usize, len: usize) -> Self {
        let buffer = cx.storage_buffers.alloc();
        let generation = cx.next_uniform_gen();
        let b = &mut cx.storage_buffers[buffer.storage_buffer_id()];
        b.stride = stride.max(1);
        b.data = vec![0; stride.max(1) * len.max(1)];
        b.generation = generation;
        buffer
    }

    /// One element per item of `data`, its stride `T`'s size.
    pub fn new_with_data<T: Copy>(cx: &mut Cx, data: &[T]) -> Self {
        let buffer = Self::new(cx, std::mem::size_of::<T>(), data.len());
        buffer.set_data(cx, data);
        buffer
    }

    pub fn storage_buffer_id(&self) -> StorageBufferId {
        StorageBufferId(self.0.id, self.0.generation)
    }

    /// Replaces its contents (and its length, at its stride); uploaded before the next dispatch.
    pub fn set_data<T: Copy>(&self, cx: &mut Cx, data: &[T]) {
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        let generation = cx.next_uniform_gen();
        let b = &mut cx.storage_buffers[self.storage_buffer_id()];
        if bytes.len() % b.stride != 0 {
            error!("StorageBuffer::set_data: {} bytes is not a whole number of {}-byte elements", bytes.len(), b.stride);
            return;
        }
        b.data.clear();
        b.data.extend_from_slice(bytes);
        b.generation = generation;
    }

    /// How many elements it holds.
    pub fn len(&self, cx: &Cx) -> usize {
        let b = &cx.storage_buffers[self.storage_buffer_id()];
        b.data.len() / b.stride
    }

    pub fn is_empty(&self, cx: &Cx) -> bool {
        self.len(cx) == 0
    }

    /// The bytes of one element.
    pub fn stride(&self, cx: &Cx) -> usize {
        cx.storage_buffers[self.storage_buffer_id()].stride
    }
}

#[derive(Default)]
pub struct CxStorageBufferPool(pub(crate) IdPool<CxStorageBuffer>);

impl CxStorageBufferPool {
    pub fn alloc(&mut self) -> StorageBuffer {
        let (new_id, _previous_item) = self.0.alloc_with_reuse_filter(|_| true, CxStorageBuffer::default());
        StorageBuffer(Rc::new(new_id))
    }
}

impl std::ops::Index<StorageBufferId> for CxStorageBufferPool {
    type Output = CxStorageBuffer;
    fn index(&self, index: StorageBufferId) -> &Self::Output {
        let d = &self.0.pool[index.0];
        if d.generation != index.1 {
            error!("StorageBuffer id generation wrong {} {} {}", index.0, d.generation, index.1)
        }
        &d.item
    }
}

impl std::ops::IndexMut<StorageBufferId> for CxStorageBufferPool {
    fn index_mut(&mut self, index: StorageBufferId) -> &mut Self::Output {
        let d = &mut self.0.pool[index.0];
        if d.generation != index.1 {
            error!("StorageBuffer id generation wrong {} {} {}", index.0, d.generation, index.1)
        }
        &mut d.item
    }
}

#[derive(Default)]
pub struct CxStorageBuffer {
    /// bumped whenever `data` changes, so it is uploaded again
    pub generation: u64,
    pub stride: usize,
    pub data: Vec<u8>,
}
