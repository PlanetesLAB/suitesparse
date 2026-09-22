use std::ffi::c_void;

use suitesparse_sys::{
    UMFPACK_A, UMFPACK_OK, umfpack_dl_free_numeric, umfpack_dl_free_symbolic, umfpack_dl_numeric,
    umfpack_dl_solve, umfpack_dl_symbolic,
};

use crate::sparse::{CscMatrix, SparseError, SparseTriplet};

use super::{UmfpackControl, UmfpackInfo};

/// Owns a UMFPACK numeric factorization and the matrix used to build it.
///
/// Keeping the matrix here ensures iterative refinement cannot read a
/// different or already-dropped CSC matrix during a later solve.
#[derive(Debug)]
pub struct UmfpackLU {
    numeric: *mut c_void,
    matrix: Option<CscMatrix>,
}

impl Default for UmfpackLU {
    fn default() -> Self {
        Self::new()
    }
}

impl UmfpackLU {
    #[must_use]
    pub fn new() -> Self {
        Self {
            numeric: std::ptr::null_mut(),
            matrix: None,
        }
    }

    /// Assemble and factorize triplets, reusing CSC storage across calls.
    ///
    /// # Errors
    /// Returns the UMFPACK status from symbolic or numeric factorization.
    pub fn factorize(
        &mut self,
        n: usize,
        triplet: &SparseTriplet,
        control: &UmfpackControl,
        info: &mut UmfpackInfo,
    ) -> Result<(), UmfpackError> {
        self.clear();
        if self
            .matrix
            .as_ref()
            .is_none_or(|matrix| matrix.dimension() != n)
        {
            self.matrix = Some(CscMatrix::new(n, triplet.len())?);
        }
        let matrix = self.matrix.as_mut().expect("matrix was just initialized");
        matrix.assemble(triplet)?;
        let n = i64::try_from(n).map_err(|_| UmfpackError::DimensionOutOfRange)?;
        let (ap, ai, ax) = matrix.raw_parts();
        let mut symbolic = std::ptr::null_mut();
        let symbolic_status = unsafe {
            umfpack_dl_symbolic(
                n,
                n,
                ap,
                ai,
                ax,
                &raw mut symbolic,
                control.as_ptr(),
                info.as_mut_ptr(),
            )
        };
        if symbolic_status != UMFPACK_OK as i32 {
            if !symbolic.is_null() {
                unsafe { umfpack_dl_free_symbolic(&raw mut symbolic) };
            }
            return Err(UmfpackError::UmfpackStatus {
                operation: "symbolic factorization",
                code: symbolic_status,
            });
        }

        let mut numeric = std::ptr::null_mut();
        let numeric_status = unsafe {
            umfpack_dl_numeric(
                ap,
                ai,
                ax,
                symbolic,
                &raw mut numeric,
                control.as_ptr(),
                info.as_mut_ptr(),
            )
        };
        unsafe { umfpack_dl_free_symbolic(&raw mut symbolic) };
        if numeric_status != UMFPACK_OK as i32 {
            if !numeric.is_null() {
                unsafe { umfpack_dl_free_numeric(&raw mut numeric) };
            }
            return Err(UmfpackError::UmfpackStatus {
                operation: "numeric factorization",
                code: numeric_status,
            });
        }

        self.numeric = numeric;
        Ok(())
    }

    /// Solve Ax = b into `solution`; `rhs` is not modified.
    ///
    /// # Errors
    /// Returns an error if not factorized, vector lengths are wrong, or
    /// UMFPACK reports a solve failure.
    pub fn solve(
        &self,
        rhs: &[f64],
        solution: &mut [f64],
        control: &UmfpackControl,
        info: &mut UmfpackInfo,
    ) -> Result<(), UmfpackError> {
        if self.numeric.is_null() {
            return Err(UmfpackError::NotFactorized);
        }
        let matrix = self.matrix.as_ref().ok_or(UmfpackError::NotFactorized)?;
        let n = matrix.dimension();
        if rhs.len() != n {
            return Err(UmfpackError::InvalidVectorLength {
                vector: "rhs",
                expected: n,
                actual: rhs.len(),
            });
        }
        if solution.len() != n {
            return Err(UmfpackError::InvalidVectorLength {
                vector: "solution",
                expected: n,
                actual: solution.len(),
            });
        }
        let (ap, ai, ax) = matrix.raw_parts();
        let status = unsafe {
            umfpack_dl_solve(
                UMFPACK_A as i32,
                ap,
                ai,
                ax,
                solution.as_mut_ptr(),
                rhs.as_ptr(),
                self.numeric,
                control.as_ptr(),
                info.as_mut_ptr(),
            )
        };
        if status == UMFPACK_OK as i32 {
            Ok(())
        } else {
            Err(UmfpackError::UmfpackStatus {
                operation: "solve",
                code: status,
            })
        }
    }

    /// Release the factorization while retaining CSC buffer capacity for reuse.
    pub fn clear(&mut self) {
        if !self.numeric.is_null() {
            unsafe { umfpack_dl_free_numeric(&raw mut self.numeric) };
        }
    }
}

impl Drop for UmfpackLU {
    fn drop(&mut self) {
        self.clear();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UmfpackError {
    DimensionOutOfRange,
    Sparse(SparseError),
    NotFactorized,
    InvalidVectorLength {
        vector: &'static str,
        expected: usize,
        actual: usize,
    },
    UmfpackStatus {
        operation: &'static str,
        code: i32,
    },
}

impl std::fmt::Display for UmfpackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for UmfpackError {}

impl From<SparseError> for UmfpackError {
    fn from(value: SparseError) -> Self {
        Self::Sparse(value)
    }
}
