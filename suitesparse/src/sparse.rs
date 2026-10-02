use suitesparse_sys::{UMFPACK_OK, umfpack_dl_triplet_to_col};

/// A validated square compressed-sparse-column matrix.
#[derive(Debug, Clone)]
pub struct CscMatrix {
    n: usize,
    col_ptr: Vec<i64>,
    row_ind: Vec<i64>,
    values: Vec<f64>,
}

impl CscMatrix {
    /// Reserve CSC buffers for a square matrix. Repeated assembly reuses them.
    ///
    /// # Errors
    /// Returns an error for a zero or unrepresentable dimension.
    pub fn new(n: usize, capacity: usize) -> Result<Self, SparseError> {
        if n == 0 {
            return Err(SparseError::EmptyMatrix);
        }
        i64::try_from(n).map_err(|_| SparseError::DimensionOutOfRange)?;
        i64::try_from(capacity).map_err(|_| SparseError::DimensionOutOfRange)?;
        Ok(Self {
            n,
            col_ptr: vec![0; n + 1],
            row_ind: Vec::with_capacity(capacity),
            values: Vec::with_capacity(capacity),
        })
    }

    /// Convert a triplet matrix to CSC, coalescing duplicate entries.
    ///
    /// # Errors
    /// Returns an error for an invalid size or index, or if UMFPACK rejects
    /// the conversion.
    pub fn from_triplet(n: usize, triplet: &SparseTriplet) -> Result<Self, SparseError> {
        let mut matrix = Self::new(n, triplet.len())?;
        matrix.assemble(triplet)?;
        Ok(matrix)
    }

    /// Rebuild this matrix from triplets, reusing allocated buffers.
    ///
    /// # Errors
    /// Returns an error for an invalid index or failed UMFPACK conversion.
    pub fn assemble(&mut self, triplet: &SparseTriplet) -> Result<(), SparseError> {
        let n_c = i64::try_from(self.n).map_err(|_| SparseError::DimensionOutOfRange)?;
        let nnz = triplet.len();
        if triplet.rows.len() != nnz || triplet.cols.len() != nnz {
            return Err(SparseError::InvalidTriplet);
        }
        let nnz_c = i64::try_from(nnz).map_err(|_| SparseError::DimensionOutOfRange)?;
        for (entry, (&row, &col)) in triplet.rows.iter().zip(&triplet.cols).enumerate() {
            if row < 0 || col < 0 || row >= n_c || col >= n_c {
                return Err(SparseError::IndexOutOfBounds {
                    entry,
                    row,
                    col,
                    n: self.n,
                });
            }
        }

        self.row_ind.resize(nnz, 0);
        self.values.resize(nnz, 0.0);
        let status = unsafe {
            umfpack_dl_triplet_to_col(
                n_c,
                n_c,
                nnz_c,
                triplet.rows.as_ptr(),
                triplet.cols.as_ptr(),
                triplet.vals.as_ptr(),
                self.col_ptr.as_mut_ptr(),
                self.row_ind.as_mut_ptr(),
                self.values.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != UMFPACK_OK as i32 {
            return Err(SparseError::ConversionFailed(status));
        }

        // UMFPACK sums duplicates, so the CSC result may have fewer entries.
        let used = usize::try_from(self.col_ptr[self.n]).map_err(|_| SparseError::InvalidOutput)?;
        if used > nnz {
            return Err(SparseError::InvalidOutput);
        }
        self.row_ind.truncate(used);
        self.values.truncate(used);
        Ok(())
    }

    #[must_use]
    pub fn dimension(&self) -> usize {
        self.n
    }

    #[must_use]
    pub fn nonzeros(&self) -> usize {
        self.values.len()
    }

    #[must_use]
    pub fn column_pointers(&self) -> &[i64] {
        &self.col_ptr
    }

    #[must_use]
    pub fn row_indices(&self) -> &[i64] {
        &self.row_ind
    }

    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    #[cfg(feature = "klu")]
    pub(crate) fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    pub(crate) fn raw_parts(&self) -> (*const i64, *const i64, *const f64) {
        (
            self.col_ptr.as_ptr(),
            self.row_ind.as_ptr(),
            self.values.as_ptr(),
        )
    }
}

#[derive(Debug, Default, Clone)]
pub struct SparseTriplet {
    rows: Vec<i64>,
    cols: Vec<i64>,
    vals: Vec<f64>,
}

impl SparseTriplet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.vals.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vals.is_empty()
    }

    pub fn clear(&mut self) {
        self.rows.clear();
        self.cols.clear();
        self.vals.clear();
    }

    /// Add an entry. Matrix bounds are checked during CSC conversion.
    ///
    /// # Panics
    /// Panics if a row or column index exceeds UMFPACK's 64-bit index range.
    pub fn add(&mut self, row: usize, col: usize, val: f64) {
        let row = i64::try_from(row).expect("row index exceeds i64");
        let col = i64::try_from(col).expect("column index exceeds i64");
        self.rows.push(row);
        self.cols.push(col);
        self.vals.push(val);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseError {
    EmptyMatrix,
    DimensionOutOfRange,
    IndexOutOfBounds {
        entry: usize,
        row: i64,
        col: i64,
        n: usize,
    },
    ConversionFailed(i32),
    InvalidTriplet,
    InvalidOutput,
}

impl std::fmt::Display for SparseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SparseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reassembly_reuses_csc_buffers() {
        let mut triplet = SparseTriplet::new();
        triplet.add(0, 0, 1.0);
        triplet.add(1, 1, 2.0);
        let mut matrix = CscMatrix::new(2, 2).unwrap();
        let row_buffer = matrix.row_ind.as_ptr();
        let value_buffer = matrix.values.as_ptr();
        matrix.assemble(&triplet).unwrap();

        triplet.clear();
        triplet.add(0, 1, 3.0);
        matrix.assemble(&triplet).unwrap();
        assert_eq!(matrix.row_ind.as_ptr(), row_buffer);
        assert_eq!(matrix.values.as_ptr(), value_buffer);
        assert_eq!(matrix.nonzeros(), 1);
    }
}
