// original: cuda/samples/0_Simple/vectorAdd

extern crate rand;
extern crate cuda;

use std::os::raw::{c_float, c_void};

use rand::Rng;

extern "C" {
    fn vector_add(a: *const c_float, b: *const c_float, c: *mut c_float, num: usize);
}

fn main() {
    const NUM: usize = 50000;
    println!("[Vector addition of {} elements]", NUM);

    {
        let mut rng = rand::thread_rng();
        let h_a: Vec<c_float> = (0..NUM).map(|_| rng.gen()).collect();
        let h_b: Vec<c_float> = (0..NUM).map(|_| rng.gen()).collect();
        let mut h_c: Vec<c_float> = vec![0.; NUM];

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
        stream.with(|stream| unsafe {
            cuda::misc::launch_kernel(vector_add as *const c_void,
                                      BLOCKS_PER_GRID.into(),
                                      THREADS_PER_BLOCK.into(),
                                      &mut [&mut d_a.as_ptr(),
                                            &mut d_b.as_ptr(),
                                            &mut d_c.as_mut_ptr(),
                                            &mut NUM],
                                      0,
                                      stream)
                    .unwrap()
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
