// original: cuda/samples/0_Simple/vectorAdd/vectorAdd.cu

extern crate rand;
extern crate cuda;

use std::os::raw::{c_float, c_int, c_void};

use rand::Rng;

extern "C" {
    fn vectorAdd(A: *const c_float, B: *const c_float, C: *mut c_float, numElements: c_int);
}

fn vector_add<A, B, C>(grid_dim: cuda::misc::Dim3,
                       block_dim: cuda::misc::Dim3,
                       a: &cuda::memory::Array<A>,
                       b: &cuda::memory::Array<B>,
                       c: &mut cuda::memory::Array<C>,
                       num: usize,
                       stream: &cuda::stream::Handle)
                       -> cuda::Result<()>
    where A: cuda::memory::Ptr<Type = c_float>,
          B: cuda::memory::Ptr<Type = c_float>,
          C: cuda::memory::PtrMut<Type = c_float>
{
    assert_eq!(a.len(), num);
    assert_eq!(b.len(), num);
    assert_eq!(c.len(), num);

    unsafe {
        cuda::misc::launch_kernel(vectorAdd as *const c_void,
                                  grid_dim.into(),
                                  block_dim.into(),
                                  &mut [&mut a.as_ptr(),
                                        &mut b.as_ptr(),
                                        &mut c.as_mut_ptr(),
                                        &mut (num as c_int)],
                                  0,
                                  stream)
    }
}

fn main() {
    const NUM: usize = 50000;
    println!("[Vector addition of {} elements]", NUM);

    {
        let mut rng = rand::thread_rng();
        let h_a: Vec<_> = (0..NUM).map(|_| rng.gen()).collect();
        let h_b: Vec<_> = (0..NUM).map(|_| rng.gen()).collect();
        let mut h_c: Vec<_> = vec![0.; NUM];

        let mut d_a = cuda::memory::Array::new(NUM).unwrap();
        let mut d_b = cuda::memory::Array::new(NUM).unwrap();
        let mut d_c = cuda::memory::Array::new(NUM).unwrap();

        println!("Copy input data from the host memory to the CUDA device");
        cuda::memory::memcpy(&mut d_a, &h_a).unwrap();
        cuda::memory::memcpy(&mut d_b, &h_b).unwrap();

        const THREADS_PER_BLOCK: usize = 256;
        const BLOCKS_PER_GRID: usize = (NUM + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        println!("CUDA kernel launch with {} blocks of {} threads",
                 BLOCKS_PER_GRID,
                 THREADS_PER_BLOCK);

        let stream = cuda::stream::Stream::default();
        stream.with(|stream| {
            vector_add(BLOCKS_PER_GRID.into(),
                       THREADS_PER_BLOCK.into(),
                       &d_a,
                       &d_b,
                       &mut d_c,
                       NUM,
                       stream)
                    .unwrap();
        });

        println!("Copy output data from the CUDA device to the host memory");
        cuda::memory::memcpy(&mut h_c, &d_c).unwrap();

        for i in 0..NUM {
            if (h_a[i] + h_b[i] - h_c[i]).abs() > 1e-5 {
                panic!("Result verification failed at element {}!", i);
            }
        }
        println!("Test PASSED");
    }

    println!("Done");
}
