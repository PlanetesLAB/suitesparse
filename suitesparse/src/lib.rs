#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::inline_always)]

#[cfg(feature = "umfpack")]
pub mod sparse;

#[cfg(feature = "umfpack")]
pub mod umfpack;
