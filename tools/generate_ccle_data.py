"""Script to run at VM start to create CCLE data if absent."""

from omics_rpz.data.loading.ccle import load_rnaseq_ccle

ccle_exp_tpm = load_rnaseq_ccle(normalization_rnaseq="TPM")
ccle_exp_raw = load_rnaseq_ccle(normalization_rnaseq="RAW")

print(ccle_exp_tpm.shape, ccle_exp_raw.shape)
