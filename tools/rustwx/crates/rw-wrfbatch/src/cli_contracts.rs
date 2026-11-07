// Shared command contracts, used by the implementation and its launcher.
pub const ENSBATCH_STATIC: &str = "gpuwm-ensemble-diagnostic-reduce.v1\tf32-words\tf64-two-pass\tmember-order\tsha256\tmember-weights-v1";
pub const COMPARE_STATIC: &str = "gpuwm.reference-input.v1";
pub const ENSBATCH: &str = "gpuwm-rw-ensbatch-products-v1\tmean\tspread\tprob\tpmm\tpaintball\tmax\t\
gpuwm-rw-ensbatch-fields-v1\tFIELD\tname\ttitle\tunits\tdefault_threshold\tunit_slug\t\
gpuwm-rw-ensbatch-events-v1\tRENDERED\tSKIPPED\tFAILED\t\
gpuwm-rw-ensbatch-vocabulary-v1\tMEMBERS\tCOVERAGE\tTIES";
pub const COMPARE: &str = "gpuwm-rw-compare-references-v1\tREFERENCE\tname\tlabel\t\
gpuwm-rw-compare-products-v1\tPRODUCT\tname\tslug\ttitle\tdifference\t\
gpuwm-rw-compare-events-v1\tRENDERED\tSKIPPED\tFAILED\tSTATS\tMATCH\tSOURCE\tFINISHED\t\
gpuwm-rw-compare-presentation-v1\t--theme";
pub const VERIFY: &str = "gpuwm.verify-visuals.request.v1\tgpuwm.verify-visuals.receipt.v1\tgpuwm.verify-visuals.inventory.v1\tstations\tradar\tartifacts\tnative-scoring\tnative-rendering\tcase-scorecard-inputs-v1";
pub const SCORECARD: &str = "gpuwm.obs-scorecard.manifest.v1\tgpuwm.obs-scorecard.receipt.v1\tnative-paired-scoring\tcase-block-bootstrap";
pub const NOWCAST_FRAMES: &str = "gpuwm-rw-nowcast-frames-v1\tfrom-netcdf\t--netcdf\t--out\t--source-id\t--lattice\t\
--variable\t--lat\t--lon\t--issue\t--causal\t--receipt\t--source-kind\t--package\t--revision";
