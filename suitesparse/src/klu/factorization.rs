use suitesparse_sys::{
    KLU_OK, klu_l_analyze, klu_l_factor, klu_l_free_numeric, klu_l_free_symbolic, klu_l_numeric,
    klu_l_rcond, klu_l_refactor, klu_l_solve, klu_l_symbolic,
};

use crate::sparse::{CscMatrix, SparseError, SparseTriplet};

use super::KluCommon;

/// Owns a KLU symbolic analysis, its numeric factorization and the matrix
/// used to build them.
#[derive(Debug)]
pub struct KluLU {
    symbolic: *mut klu_l_symbolic,
    numeric: *mut klu_l_numeric,
    matrix: Option<CscMatrix>,
}

impl Default for KluLU {
    fn default() -> Self {
        Self::new()
    }
}

impl KluLU {
    #[must_use]
    pub fn new() -> Self {
        Self {
            symbolic: std::ptr::null_mut(),
            numeric: std::ptr::null_mut(),
            matrix: None,
        }
    }

    /// Assemble and factorize triplets, reusing CSC storage across calls.
    ///
    /// # Errors
    /// Returns the KLU status from symbolic or numeric factorization.
    pub fn factorize(
        &mut self,
        n: usize,
        triplet: &SparseTriplet,
        common: &mut KluCommon,
    ) -> Result<(), KluError> {
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
        let n = i64::try_from(n).map_err(|_| KluError::DimensionOutOfRange)?;
        // KLU reads the matrix without modifying it.
        let (ap, ai, ax) = matrix.raw_parts();
        let (ap, ai, ax) = (ap.cast_mut(), ai.cast_mut(), ax.cast_mut());

        self.symbolic = unsafe { klu_l_analyze(n, ap, ai, common.as_mut_ptr()) };
        if self.symbolic.is_null() {
            return Err(KluError::KluStatus {
                operation: "symbolic factorization",
                code: common.options().status,
            });
        }
        self.numeric = unsafe { klu_l_factor(ap, ai, ax, self.symbolic, common.as_mut_ptr()) };
        if self.numeric.is_null() || common.options().status != KLU_OK as i32 {
            self.clear();
            return Err(KluError::KluStatus {
                operation: "numeric factorization",
                code: common.options().status,
            });
        }
        Ok(())
    }

    /// The factorized matrix, if any.
    #[must_use]
    pub fn matrix(&self) -> Option<&CscMatrix> {
        self.matrix.as_ref()
    }

    /// Values of the factorized matrix in CSC order, to be changed before
    /// [`refactor`](Self::refactor). The sparsity pattern cannot change.
    pub fn values_mut(&mut self) -> Option<&mut [f64]> {
        self.matrix.as_mut().map(CscMatrix::values_mut)
    }

    /// Factorize the current values with the pivot order of the last
    /// [`factorize`](Self::factorize). A failure releases the factorization.
    ///
    /// # Errors
    /// Returns an error if not factorized or KLU reports a failure.
    pub fn refactor(&mut self, common: &mut KluCommon) -> Result<(), KluError> {
        if self.numeric.is_null() {
            return Err(KluError::NotFactorized);
        }
        let matrix = self.matrix.as_ref().ok_or(KluError::NotFactorized)?;
        // KLU reads the matrix without modifying it.
        let (ap, ai, ax) = matrix.raw_parts();
        let ok = unsafe {
            klu_l_refactor(
                ap.cast_mut(),
                ai.cast_mut(),
                ax.cast_mut(),
                self.symbolic,
                self.numeric,
                common.as_mut_ptr(),
            )
        };
        if ok != 0 {
            Ok(())
        } else {
            self.clear();
            Err(KluError::KluStatus {
                operation: "refactorization",
                code: common.options().status,
            })
        }
    }

    /// Cheap reciprocal condition estimate, min |diag(U)| / max |diag(U)|.
    ///
    /// # Errors
    /// Returns an error if not factorized or KLU reports a failure.
    pub fn rcond(&mut self, common: &mut KluCommon) -> Result<f64, KluError> {
        if self.numeric.is_null() {
            return Err(KluError::NotFactorized);
        }
        if unsafe { klu_l_rcond(self.symbolic, self.numeric, common.as_mut_ptr()) } != 0 {
            Ok(common.options().rcond)
        } else {
            Err(KluError::KluStatus {
                operation: "condition estimate",
                code: common.options().status,
            })
        }
    }

    /// Solve Ax = b into `solution`; `rhs` is not modified.
    ///
    /// # Errors
    /// Returns an error if not factorized, vector lengths are wrong, or
    /// KLU reports a solve failure.
    pub fn solve(
        &mut self,
        rhs: &[f64],
        solution: &mut [f64],
        common: &mut KluCommon,
    ) -> Result<(), KluError> {
        if self.numeric.is_null() {
            return Err(KluError::NotFactorized);
        }
        let matrix = self.matrix.as_ref().ok_or(KluError::NotFactorized)?;
        let n = matrix.dimension();
        if rhs.len() != n {
            return Err(KluError::InvalidVectorLength {
                vector: "rhs",
                expected: n,
                actual: rhs.len(),
            });
        }
        if solution.len() != n {
            return Err(KluError::InvalidVectorLength {
                vector: "solution",
                expected: n,
                actual: solution.len(),
            });
        }
        solution.copy_from_slice(rhs);
        let n = i64::try_from(n).map_err(|_| KluError::DimensionOutOfRange)?;
        let solved = unsafe {
            klu_l_solve(
                self.symbolic,
                self.numeric,
                n,
                1,
                solution.as_mut_ptr(),
                common.as_mut_ptr(),
            )
        };
        if solved != 0 {
            Ok(())
        } else {
            Err(KluError::KluStatus {
                operation: "solve",
                code: common.options().status,
            })
        }
    }

    /// Release the factorization while retaining CSC buffer capacity for reuse.
    pub fn clear(&mut self) {
        // KLU frees nothing without a Common; its memory statistics are not needed here.
        let mut common = KluCommon::new();
        if !self.numeric.is_null() {
            unsafe { klu_l_free_numeric(&raw mut self.numeric, common.as_mut_ptr()) };
        }
        if !self.symbolic.is_null() {
            unsafe { klu_l_free_symbolic(&raw mut self.symbolic, common.as_mut_ptr()) };
        }
    }
}

impl Drop for KluLU {
    fn drop(&mut self) {
        self.clear();
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KluError {
    DimensionOutOfRange,
    Sparse(SparseError),
    NotFactorized,
    InvalidVectorLength {
        vector: &'static str,
        expected: usize,
        actual: usize,
    },
    KluStatus {
        operation: &'static str,
        code: i32,
    },
}

impl std::fmt::Display for KluError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for KluError {}

impl From<SparseError> for KluError {
    fn from(value: SparseError) -> Self {
        Self::Sparse(value)
    }
}
