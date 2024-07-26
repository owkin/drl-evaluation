if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if (!require("TCGAbiolinks", quietly = TRUE))
  BiocManager::install("TCGAbiolinks")
if (!requireNamespace("data.table", quietly = TRUE))
  install.packages("data.table")

library(TCGAbiolinks)
library(SummarizedExperiment)
library(dplyr)
library(tidyr)
library(tibble)
library(data.table)

library(optparse)
option_list <- list(
    make_option(c("-p", "--project"), type="character",
        dest="project", help="TCGA indication"),
    make_option(c("-f", "--min_freq"), type="double", default=0.03,
        dest="min_freq", help="Minimum mutation frequency for the gene to be included (default 0.03)"),
    make_option(c("-o", "--out"), type="character",
        dest="out", help="Output directory")
    )
args = parse_args(OptionParser(option_list=option_list))

args$project = toupper(args$project)

query <- GDCquery(
     project = paste0("TCGA-", args$project),
     data.category = "Simple Nucleotide Variation",
     access = "open",
     data.type = "Masked Somatic Mutation",
     workflow.type = "Aliquot Ensemble Somatic Variant Merging and Masking"
)
GDCdownload(query)
mut <- GDCprepare(query)

mut = mut %>%
    dplyr::mutate(barcode = substr(Tumor_Sample_Barcode, 1, 19),
                  sample_type = as.integer(substr(Tumor_Sample_Barcode, 14,15))) %>%
    filter(sample_type < 10 & IMPACT != 'LOW') %>%
    group_by(barcode, Hugo_Symbol, IMPACT) %>%
    select()

mut_matrix = mut %>%
    mutate(IMPACT = IMPACT != 'LOW') %>%
    distinct() %>%
    pivot_wider(names_from=barcode, values_from=IMPACT) %>%
    column_to_rownames(var = "Hugo_Symbol")

mut_matrix[is.na(mut_matrix)] = 0

frequencies = apply(mut_matrix, 1, sum) / ncol(mut_matrix)

mut_matrix = mut_matrix[frequencies > args$min_freq,]
mut_matrix = mut_matrix %>%
  rownames_to_column(var='Hugo')

fwrite(mut_matrix, paste0(args$out,"/filtered_mutations.tsv.gz"), sep="\t", quote=F, row.names=F)
