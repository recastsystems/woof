//! Sea-level pressure over high terrain, smoothed the way operational map graphics smooth it: the filter now lives
//! in rustwx-calc ([`rustwx_calc::mslp_smoothing`], the full description there) so the any-point column store's
//! sea-level pressure plane is the map's own, word for word.
#[allow(unused_imports)]
pub(crate) use rustwx_calc::mslp_smoothing::*;
