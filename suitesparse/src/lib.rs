#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::inline_always)]

#[cfg(feature = "umfpack")]
pub mod sparse;

#[cfg(feature = "umfpack")]
pub mod umfpack;

// CSC assembly in `sparse` uses UMFPACK's triplet conversion.
#[cfg(all(feature = "klu", feature = "umfpack"))]
pub mod klu;
