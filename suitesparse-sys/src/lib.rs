#![allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    improper_ctypes,
    clippy::all
)]

use std::default::Default;
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

impl Default for klu_common {
    fn default() -> Self {
        unsafe {
            let mut common = std::mem::zeroed::<klu_common_struct>();
            klu_defaults(&raw mut common);
            common
        }
    }
}

impl Default for klu_l_common {
    fn default() -> Self {
        unsafe {
            let mut common = std::mem::zeroed::<klu_l_common_struct>();
            klu_l_defaults(&raw mut common);
            common
        }
    }
}

#[cfg(feature = "klu")]
use crate::{
    klu_common as klu_common_, klu_l_common as klu_l_common_, klu_l_numeric as klu_l_numeric_,
    klu_l_symbolic as klu_l_symbolic_, klu_numeric as klu_numeric_, klu_symbolic as klu_symbolic_,
};

#[cfg(feature = "klu")]
unsafe extern "C" {
    pub fn klu_analyze(
        n: i32,
        Ap: *const i32,
        Ai: *const i32,
        Common: *mut klu_common_,
    ) -> *mut klu_symbolic_;

    pub fn klu_factor(
        Ap: *const i32,
        Ai: *const i32,
        Ax: *const f64,
        Symbolic: *mut klu_symbolic_,
        Common: *mut klu_common_,
    ) -> *mut klu_numeric_;

    pub fn klu_z_factor(
        Ap: *const i32,
        Ai: *const i32,
        Ax: *const f64,
        Symbolic: *mut klu_symbolic_,
        Common: *mut klu_common_,
    ) -> *mut klu_numeric_;

    pub fn klu_l_analyze(
        n: i64,
        Ap: *const i64,
        Ai: *const i64,
        Common: *mut klu_l_common_,
    ) -> *mut klu_l_symbolic_;

    pub fn klu_l_factor(
        Ap: *const i64,
        Ai: *const i64,
        Ax: *const f64,
        Symbolic: *mut klu_l_symbolic_,
        Common: *mut klu_l_common_,
    ) -> *mut klu_l_numeric_;

    pub fn klu_zl_factor(
        Ap: *const i64,
        Ai: *const i64,
        Ax: *const f64,
        Symbolic: *mut klu_l_symbolic_,
        Common: *mut klu_l_common_,
    ) -> *mut klu_l_numeric_;

}

#[cfg(test)]
mod tests {
    #![allow(clippy::cast_sign_loss)]
    use super::*;

    #[test]
    fn klu_simple() {
        let n = 5i32;
        let Ap = vec![0, 2, 5, 9, 10, 12];
        let Ai = vec![0, 1, 0, 2, 4, 1, 2, 3, 4, 2, 1, 4];
        let Ax = vec![2., 3., 3., -1., 4., 4., -3., 1., 2., 2., 6., 1.];
        let mut b = vec![8., 45., -3., 3., 19.];

        let mut common = klu_common::default();
        let mut symbolic = unsafe { klu_analyze(n, Ap.as_ptr(), Ai.as_ptr(), &raw mut common) };
        let mut numeric = unsafe {
            klu_factor(
                Ap.as_ptr(),
                Ai.as_ptr(),
                Ax.as_ptr(),
                symbolic,
                &raw mut common,
            )
        };
        unsafe { klu_solve(symbolic, numeric, n, 1, b.as_mut_ptr(), &raw mut common) };
        unsafe { klu_free_symbolic(&raw mut symbolic, &raw mut common) };
        unsafe { klu_free_numeric(&raw mut numeric, &raw mut common) };

        let expect = vec![1., 2., 3., 4., 5.];
        for i in 0..(n as usize) {
            assert!((b[i] - expect[i]).abs() < 1e-10);
        }
    }
}
