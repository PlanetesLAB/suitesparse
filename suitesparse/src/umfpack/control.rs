use suitesparse_sys::{
    UMFPACK_ALLOC_INIT, UMFPACK_CONTROL, UMFPACK_FIXQ, UMFPACK_INFO, UMFPACK_IRSTEP,
    UMFPACK_ORDERING, UMFPACK_ORDERING_CHOLMOD, UMFPACK_PIVOT_TOLERANCE, UMFPACK_SCALE,
    UMFPACK_SCALE_NONE, UMFPACK_STRATEGY, UMFPACK_STRATEGY_SYMMETRIC, umfpack_dl_defaults,
};

/// UMFPACK options, initialized with the project's stiff-kinetics settings.
#[derive(Debug)]
pub struct UmfpackControl {
    inner: [f64; UMFPACK_CONTROL as usize],
}

impl Default for UmfpackControl {
    fn default() -> Self {
        Self::new()
    }
}

impl UmfpackControl {
    #[must_use]
    pub fn new() -> Self {
        let mut inner = [0.0; UMFPACK_CONTROL as usize];
        unsafe { umfpack_dl_defaults(inner.as_mut_ptr()) };

        inner[UMFPACK_SCALE as usize] = f64::from(UMFPACK_SCALE_NONE);
        inner[UMFPACK_PIVOT_TOLERANCE as usize] = 1.0;
        inner[UMFPACK_ORDERING as usize] = f64::from(UMFPACK_ORDERING_CHOLMOD);
        inner[UMFPACK_IRSTEP as usize] = 5.0;
        inner[UMFPACK_ALLOC_INIT as usize] = 0.9;
        inner[UMFPACK_STRATEGY as usize] = f64::from(UMFPACK_STRATEGY_SYMMETRIC);
        inner[UMFPACK_FIXQ as usize] = 1.0;
        Self { inner }
    }

    #[must_use]
    pub fn options(&self) -> &[f64; UMFPACK_CONTROL as usize] {
        &self.inner
    }

    pub fn options_mut(&mut self) -> &mut [f64; UMFPACK_CONTROL as usize] {
        &mut self.inner
    }

    pub(crate) fn as_ptr(&self) -> *const f64 {
        self.inner.as_ptr()
    }
}

/// Diagnostics filled by UMFPACK operations.
#[derive(Debug)]
pub struct UmfpackInfo {
    inner: [f64; UMFPACK_INFO as usize],
}

impl Default for UmfpackInfo {
    fn default() -> Self {
        Self {
            inner: [0.0; UMFPACK_INFO as usize],
        }
    }
}

impl UmfpackInfo {
    #[must_use]
    pub fn values(&self) -> &[f64; UMFPACK_INFO as usize] {
        &self.inner
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut f64 {
        self.inner.as_mut_ptr()
    }
}
