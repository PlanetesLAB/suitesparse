use suitesparse_sys::{klu_l_common, klu_l_defaults};

/// KLU parameters and the statistics of the last KLU call, initialized by
/// `klu_l_defaults`.
#[derive(Debug)]
pub struct KluCommon {
    inner: klu_l_common,
}

impl Default for KluCommon {
    fn default() -> Self {
        Self::new()
    }
}

impl KluCommon {
    #[must_use]
    pub fn new() -> Self {
        let mut inner = unsafe { std::mem::zeroed() };
        unsafe { klu_l_defaults(&raw mut inner) };
        Self { inner }
    }

    #[must_use]
    pub fn options(&self) -> &klu_l_common {
        &self.inner
    }

    pub fn options_mut(&mut self) -> &mut klu_l_common {
        &mut self.inner
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut klu_l_common {
        &raw mut self.inner
    }
}
