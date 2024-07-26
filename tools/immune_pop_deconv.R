library(MCPcounter)
library(data.table)
library(ComplexHeatmap)
library(circlize)

library(optparse)
option_list <- list(
    make_option(c("-c", "--counts"), type="character",
        dest="counts", help="Normalized counts file"),
    make_option(c("-o", "--out"), type="character",
        dest="out", help="Output directory")
    )
args = parse_args(OptionParser(option_list=option_list))

# Import data
cts  <- fread(args$counts)
#cts = fread("rna-seq/results/LUAD/LUAD_norm_counts.tsv") ### TO CHANGE
genes = cts[["Hugo"]]
cts = cts[, Hugo:=NULL] ### obsolete? quand on partait des hugo?

############ MCP-counter #############
if(all(grepl("^ENSG", genes))){ ## To finish to define for probes and maybe entrezid
    featype="ENSEMBL_ID"
}else{
    featype="HUGO_symbols"
}

cts = as.data.frame(cts)
rownames(cts) = genes

probesets=read.table(curl:::curl("https://raw.githubusercontent.com/ebecht/MCPcounter/master/Signatures/probesets.txt"),sep="\t",stringsAsFactors=FALSE,colClasses="character")
genes=read.table(curl:::curl("https://raw.githubusercontent.com/ebecht/MCPcounter/master/Signatures/genes.txt"),sep="\t",stringsAsFactors=FALSE,header=TRUE,colClasses="character",check.names=FALSE)

if(featype=="affy133P2_probesets"){
    features=probesets
    markers.names = unique(features[, 2])
    features=split(features[,1],features[,2])
    features=lapply(features,intersect,x=rownames(cts))
    features=features[sapply(features,function(x)length(x)>0)]
    missing.populations=setdiff(markers.names,names(features))
    features=features[intersect(markers.names,names(features))]
} else {
    markersG=genes
}

if(featype=="HUGO_symbols"){
    features=subset(markersG,get("HUGO symbols")%in%rownames(cts))
    markers.names = unique(features[, "Cell population"])
    features=split(features[,"HUGO symbols"],features[,"Cell population"])
    missing.populations=setdiff(markers.names,names(features))
    features=features[intersect(markers.names,names(features))]
}

if(featype=="ENTREZ_ID"){
    features=subset(markersG,ENTREZID%in%rownames(cts))
    markers.names = unique(features[, "Cell population"])
    features=split(features[,"ENTREZID"],features[,"Cell population"])
    missing.populations=setdiff(markers.names,names(features))
    features=features[intersect(markers.names,names(features))]
}

if(featype=="ENSEMBL_ID"){
    features=subset(markersG,get("ENSEMBL ID")%in%rownames(cts))
    markers.names = unique(features[, "Cell population"])
    features=split(features[,"ENSEMBL ID"],features[,"Cell population"])
    missing.populations=setdiff(markers.names,names(features))
    features=features[intersect(markers.names,names(features))]
}


if(length(missing.populations)>0){
    warning(paste("Found no markers for population(s):",paste(missing.populations,collapse=", ")))
}

row_mcp = t(appendSignatures(cts,features))

#mcp = MCPcounter::MCPcounter.estimate(cts, featuresType="ENSEMBL_ID",probesets=probesets, genes=genes)


mcp = data.table(Signature = rownames(row_mcp), row_mcp)
mcp_scaled = data.table(Signature = rownames(row_mcp), t(scale(t(row_mcp))))

print(dim(cts))
print(dim(mcp))

fwrite(mcp, file = paste0(args$out, "/mcp_deconv.tsv"), sep = "\t")
fwrite(mcp_scaled, file = paste0(args$out, "/mcp_deconv_scaled.tsv"), sep = "\t")


png(paste0(args$out, "/mcp_scaled.png"), width=700, height=700)
H = Heatmap(
      t(scale(t(row_mcp))),
      clustering_method_rows = "ward.D2",
      clustering_distance_rows = "pearson",
      clustering_method_columns = "ward.D2",
      clustering_distance_columns = "pearson",
      show_column_names=F,
      col = colorRamp2(c(-2, 0, 2), c("blue", "white", "red")),
      height = unit(8, "cm"),
      show_row_names = T,
      column_title = "MCPcounter heatmap result")
print(H)
dev.off()
